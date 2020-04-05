import numpy as np
import pandas as pd
import re
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping


def get_data():
    """Load, clean, and perform feature engineering on counsel chat data.

    Returns:
        pd.DataFrame: Data ready for modeling.
    """

    # load and prepare datatypes
    df = pd.read_csv(Path('.')/ 'data' / '20200325_counsel_chat.csv')
    for c in ['questionTitle', 'questionText', 'answerText']: df[c] = df[c].astype(str)

    # mine the first sentences from answers for reflections
    sent_split_regex = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    df['first_an_sent'] = df.answerText.apply(lambda x: sent_split_regex.split(x)[0])

    # mine "seems like" and "sounds like" sentences from answers
    seem_sounds_regex = re.compile(r'^.{0,50}?(?:seems\slike)|(?:sounds\slike)')
    def extract_seems_sounds(s):
        sents = sent_split_regex.split(s)
        for sent in sents:
            match = seem_sounds_regex.match(sent.lower())
            if match is not None: return sent
    df['seems_sounds_sents'] = df.answerText.apply(extract_seems_sounds)

    # construct reflections
    df['reflection'] = df.first_an_sent
    seems_sounds_mask = df.seems_sounds_sents.notnull()
    df.loc[seems_sounds_mask, 'reflection'] = df.loc[seems_sounds_mask, 'seems_sounds_sents']

    # truncate long reflections to put a bandaid on the cases when sentence splitting fails
    df.reflection = df.reflection.apply(lambda x: x[:250])

    return df


def toggle_model_freeze(model, frozen: bool):
    for p in model.parameters(): p.requires_grad = not frozen


def print_frozenness(model):
    x = [p.requires_grad for p in model.parameters()]
    n_frozen = len(x) - sum(x)
    n_params = len(x)
    print(f'{n_frozen} / {n_params} ({n_frozen / n_params * 100:.3f})% parameters are frozen')


class VarLenLSTM(nn.Module):
    """A generic LSTM module which efficiently handles batches of variable length (ie padded) sequences using
    packing and unpacking
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        """
        Args:
            input_size (int): Dimensionality of LSTM input vector
            hidden_size (int): Dimensionality of LSTM hidden state and cell state
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)

    def forward(self, x, x_lens, h_0, c_0):
        """
        Most of the code here handles the packing and unpacking that's required to efficiently
        perform LSTM calculations for variable length sequences.

        Args:
            x: Distributed input tensors, ready for direct input to LSTM
            x_lens: Sequence lengths of examples in the batch
            h_0: Tensor to initialize LSTM hidden state
            c_0: Tensor to initialize LSTM cell state

        Returns:
            out_padded, (h_n, c_n)

            It returns the same as the underlying PyTorch LSTM; see PyTorch docs.
        """
        max_seq_len = x.size(1)
        sorted_lens, idx = x_lens.sort(dim=0, descending=True)
        x_sorted = x[idx]
        x_packed = pack_padded_sequence(x_sorted, lengths=sorted_lens, batch_first=True)
        out_packed, (h_n, c_n) = self.lstm(x_packed, (h_0, c_0))
        out_padded, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=max_seq_len)
        _, reverse_idx = idx.sort(dim=0, descending=False)
        out_padded = out_padded[reverse_idx]
        h_n = h_n[:, reverse_idx]
        c_n = c_n[:, reverse_idx]
        return out_padded, (h_n, c_n)


class ReflectionDataset(Dataset):
    """PyTorch dataset for designed for the Youper challenge, but is structured to support
    generic seq2seq learning.
    """
    def __init__(self, question_text, reflection, tokenizer, max_seq_len=512):
        """
        Args:
            question_text (list-like): List of questions.
            reflection (list-like): List of reflections.
            tokenizer (HuggingFace tokenizer): Pretrained tokenizer.
            max_seq_len (int): Max sequence length for the encoder and decoder.
        """
        self.question_text, self.reflection = question_text, reflection
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self): return len(self.question_text)

    def __getitem__(self, i):
        """`input_ids_re` and `target_ids` are nearly identical, except they are both shifted-by-one
        versions of the original set of token IDs for the reflection. This is required for the
        autoregressive setup of the decoder.

        Returns:
            (input_ids_qu, attn_mask_qu, input_ids_re, attn_mask_re), target_ids

            input_ids_qu (tensor): Token IDs for the question.
            attn_mask_qu (tensor): Attention mask for the question.
            input_ids_re (tensor): Token IDs for the reflection. Does not include the last token.
            seq_len_re (tensor): Reflection length.
            target_ids (tensor): Token IDs for the question. Does not include the first token.

        """
        input_ids_qu, attn_mask_qu = self.tokenize(
            self.question_text.iloc[i],
            self.tokenizer,
            self.max_seq_len)
        input_ids_re, attn_mask_re = self.tokenize(
            self.reflection.iloc[i],
            self.tokenizer,
            self.max_seq_len)

        # shift inputs and targets by 1
        target_ids = input_ids_re[1:]
        input_ids_re = input_ids_re[:-1]
        seq_len_re = attn_mask_re.sum(0) - 1

        return (input_ids_qu, attn_mask_qu, input_ids_re, seq_len_re), target_ids

    @staticmethod
    def tokenize(s, tokenizer, max_seq_len):
        """General function to tokenize text with a HuggingFace tokenizer.
        """
        encoded_dict = tokenizer.encode_plus(s, max_length=max_seq_len, pad_to_max_length=True)
        input_ids = torch.tensor(encoded_dict['input_ids'], dtype=torch.long)
        attn_mask = torch.tensor(encoded_dict['attention_mask'], dtype=torch.uint8)
        return input_ids, attn_mask


class ReflectionModel(LightningModule):
    def __init__(self, roberta, enc_hidden_sz=768, dec_hidden_sz=256, dec_input_sz=768, dec_num_layers=1,
                 addl_state=None):
        """

        Args:
            roberta:
            enc_hidden_sz:
            dec_hidden_sz:
            dec_input_sz:
            addl_state (dict): A dictionary of additional state necessary for this LightningModule to function
                properly. Required keys are 'df', 'train_mask', 'valid_mask', and 'tokenizer'.
        """
        super().__init__()
        self.enc = roberta
        self.addl_state = addl_state

        # map encoder output to decoder initialization
        self.h_transform = nn.Linear(enc_hidden_sz, dec_hidden_sz, bias=True)
        self.c_transform = nn.Linear(enc_hidden_sz, dec_hidden_sz, bias=True)

        # decoder word embeddings
        emb = roberta.embeddings.word_embeddings.weight.detach().clone()
        self.dec_emb = nn.Embedding.from_pretrained(
            embeddings=emb,
            freeze=False,
            padding_idx=1)
        dec_vocab_sz = emb.shape[0]

        # LSTM decoder
        self.dec = VarLenLSTM(
            input_size=dec_input_sz,
            hidden_size=dec_hidden_sz,
            num_layers=dec_num_layers)

        # project encoder hidden states as part of decoder-encoder attention
        self.proj_enc = nn.Linear(enc_hidden_sz, dec_hidden_sz, bias=False)

        # token-level classification MLP
        self.token_pred_mlp = nn.Sequential(
            nn.Linear(enc_hidden_sz + dec_hidden_sz, 1000),
            nn.ReLU(),
            nn.Linear(1000, dec_vocab_sz))

        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.addl_state['tokenizer'].convert_tokens_to_ids('<pad>'),
            reduction='mean')

    def encode(self, input_ids, attn_mask):
        # forward pass through the encoder
        last_hidden_state, _ = self.enc(input_ids=input_ids, attention_mask=attn_mask)

        # aggregate encoder outputs with max pooling
        last_hidden_state = last_hidden_state.masked_fill((attn_mask == 0).unsqueeze(2), -1e18)
        #                                                                  (bs, enc_len, enc_hidden)
        encoder_max_pool = F.adaptive_max_pool1d(last_hidden_state.transpose(1, 2), 1).squeeze(2)

        # compute decoder initialization
        h = self.h_transform(encoder_max_pool).unsqueeze(0)
        c = self.c_transform(encoder_max_pool).unsqueeze(0)

        return last_hidden_state, (h, c)

    def decode(self, last_hidden_state, h, c, input_ids_re, seq_len_re, attn_mask_qu):
        # forward pass through the decoder
        x = self.dec_emb(input_ids_re)
        x_lens = seq_len_re
        dec_out, (h, c) = self.dec(x, x_lens, h, c)  # (bs, dec_len, dec_hidden)

        proj_enc = self.proj_enc(last_hidden_state)  # (bs, enc_len, dec_hidden)
        attn_matrix = torch.bmm(dec_out, proj_enc.transpose(1, 2))  # (bs, dec_len, enc_len)
        # rows (dim1) are attn nrgs
        # compute attention over the encoder outputs
        mask = (attn_mask_qu == 0).unsqueeze(1)
        attn_matrix = attn_matrix.masked_fill(mask, -1e18)
        attn_nrgs = F.softmax(attn_matrix, dim=-1)  # (bs, dec_len, enc_len)
        attn_out = torch.bmm(attn_nrgs, last_hidden_state)  # (bs, dec_len, enc_hidden)

        # token-level classification (next word prediction)
        mlp_in = torch.cat([dec_out, attn_out], dim=2)  # (bs, dec_len, dec_hidden+enc_hidden)
        logits = self.token_pred_mlp(mlp_in).transpose(1, 2)  # (bs, vocab_sz, dec_len)

        return logits, (h, c)

    def forward(self, input_ids_qu, attn_mask_qu, input_ids_re, seq_len_re):
        # encode
        last_hidden_state, (h, c) = self.encode(input_ids_qu, attn_mask_qu)

        # decode
        logits, (h, c) = self.decode(last_hidden_state, h, c, input_ids_re, seq_len_re,
                                     attn_mask_qu)

        return logits

    def train_dataloader(self):
        df = self.addl_state['df']
        train_mask = self.addl_state['train_mask']
        tokenizer = self.addl_state['tokenizer']
        train_ds = ReflectionDataset(
            question_text=df[train_mask]['questionText'],
            reflection=df[train_mask]['reflection'],
            tokenizer=tokenizer)
        train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
        return train_dl

    def val_dataloader(self):
        df = self.addl_state['df']
        valid_mask = self.addl_state['valid_mask']
        tokenizer = self.addl_state['tokenizer']
        valid_ds = ReflectionDataset(
            question_text=df[valid_mask]['questionText'],
            reflection=df[valid_mask]['reflection'],
            tokenizer=tokenizer)
        valid_dl = DataLoader(valid_ds, batch_size=8, shuffle=False)
        return valid_dl

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        self.enc = self.enc.eval()
        x, y = batch
        input_ids_qu, attn_mask_qu, input_ids_re, seq_len_re = x
        logits = self.forward(input_ids_qu, attn_mask_qu, input_ids_re, seq_len_re)
        loss = self.loss(logits, y)
        log = {'train_loss': loss}
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        input_ids_qu, attn_mask_qu, input_ids_re, seq_len_re = x
        logits = self.forward(input_ids_qu, attn_mask_qu, input_ids_re, seq_len_re)
        loss = self.loss(logits, y)
        log = {'val_loss_batch': loss}
        return {**log, 'log': log}

    def validation_epoch_end(self, outputs):
        if len(outputs) == 0: return {'val_loss': 15, 'log': {'val_loss': 15}}
        avg_loss = torch.stack([x['val_loss_batch'] for x in outputs]).mean()
        perp = torch.exp(avg_loss)
        log = {'val_loss': avg_loss, 'val_perp': perp}
        return {**log, 'log': log}


def generate_reflection_from_tensors(input_ids_qu, attn_mask_qu, model, start_token_id, end_token_id,
                                     max_len=500, use_gpu=True):
    """

    Args:
        input_ids_qu (tensor): Token IDs for the question.
        attn_mask_qu (tensor): Attention mask for the question.
        model (nn.Module): Trained model.
        start_token_id (int): Token ID of the start token (eg "<s>") used during training.

    Returns:
        pred_toks (list[int]):
    """

    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    model = model.to(device)
    input_ids_qu = input_ids_qu.to(device)
    attn_mask_qu = attn_mask_qu.to(device)

    model = model.eval()

    last_hidden_state, (h, c) = model.encode(input_ids_qu, attn_mask_qu)

    input_ids = torch.tensor(start_token_id).unsqueeze(0).unsqueeze(0).to(device)
    seq_len_re = torch.tensor(1).unsqueeze(0).to(device)

    pred_toks = []
    for i in range(max_len):
        logits, (h, c) = model.decode(last_hidden_state, h, c, input_ids, seq_len_re, attn_mask_qu)
        logits = logits.squeeze(2)
        _, idxs = logits.max(dim=1)
        pred_tok = idxs.item()
        if pred_tok == end_token_id: break
        pred_toks.append(pred_tok)
        input_ids = idxs.unsqueeze(0)

    return pred_toks


def generate_reflection_from_string(model, tokenizer, string, start_token_id, end_token_id, use_gpu=True):
    input_ids_qu, attn_mask_qu = ReflectionDataset.tokenize(string, tokenizer, 512)
    input_ids_qu = input_ids_qu.unsqueeze(0)
    attn_mask_qu = attn_mask_qu.unsqueeze(0)
    pred_toks = generate_reflection_from_tensors(input_ids_qu, attn_mask_qu, model, start_token_id, end_token_id,
                                                 use_gpu=use_gpu)
    return pred_toks


def observe_reflection(model, tokenizer, start_token_id, end_token_id, input_ids_qu=None, attn_mask_qu=None,
                       input_ids_re=None, string=None):
    """Wrapper around `generate_reflection_from_tensors` for printing for human observation.
    """
    if string is not None:
        pred_toks = generate_reflection_from_string(model, tokenizer, string, start_token_id, end_token_id)
    else:
        pred_toks = generate_reflection_from_tensors(input_ids_qu, attn_mask_qu, model, start_token_id, end_token_id)

    print('Question:')
    print(tokenizer.decode(input_ids_qu.squeeze(0).tolist()))

    if input_ids_re is not None:
        print('\nHuman Generated Reflection:')
        print(tokenizer.decode(input_ids_re.squeeze(0).tolist()))

    print('\nMachine Generated Reflection:')
    print(tokenizer.decode(pred_toks))
