# Youper Challenge - NLP Engineer

## Overview
The system is a generative model, consisting of an encoder-decoder seq2seq stack. The encoder is a pretained roBERTa transformer (12-layer, 768-hidden, 12-heads, 125M parameters), and the decoder is a randomly initialized LSTM (1-layer, 256-hidden, 1M parameters, 1-direction). The encoder and decoder are connected in two ways:
1. The decoder LSTM hidden state is initialized with the result of a max pooling operation across the encoder outputs.
1. When making next-token predictions at each timestep, the decoder performs a multiplicative attention operation across the encoder outputs.
The same pretrained tokenizer is used for the encoder and decoder stacks. The encoder uses a frozen set of pretrained word embeddings. The decoder uses a cloned copy of these word embeddings, which are fine-tuned during training.

![Encoder Decoder Architecture](images/enc_dec_architecture.png)

## Project Structure and Source Code

The source code is located in the file `youper.py`. The following notebooks import this code and step through the development and demonstrate some of the generation capabilities:
1. **`1.0-eda-feature-engr.ipynb`**: Loads, preps, cleans, and explores the data. Extracts reflections from the answer text.
2. **`2.0-model-training.ipynb`**: Trains the system described.
3. **`3.0-model-inference.ipynb`**: Generates reflections for every example in the dataset using the trained model. Prints several examples for manual observation.  

The Python programming language and PyTorch deep learning framework were used. The python environment used to perform this work is represented in the included `requirements.txt` file. 

## Data
As suggested in the challenge, reflections are mined from the therapist answers in the following ways:
1. The first sentence in the answer.
1. Sentences that have the language "seems like" or "sounds like" near the beginning of the sentence.
These mined reflections are used for supervised learning.

## Training
During training, the question text is passed through the encoder, and the decoder is trained to generate the associated reflections in a supervised manner.
The system is trained using the Adam optimizer, a learning rate of 3e-4, and 100% teacher forcing. The entire encoder is frozen during training, whereas the decoder is not frozen. Training is stopped automatically when the validation loss stops improving. The Pytorch Lightning framework is used to facilitate the training process. Training take approximately 15-20 minutes on a single Nvidia GeForce GTX 1080 Ti GPU. 

## Generation
At inference time, the question text is passed through the encoder and a simple greedy decoding scheme is used to generate reflections.

## Results and Analysis


## Transfer Learning
Recent advances in NLP research have introduced the transformer architecture, which is commonly pretrained as a language model on a massive corpus. These pretrained models have shown great power in their world knowledge and text generation capabilities, among many others. Leveraging these pretrained models is an effective technique for building custom systems trained on smaller amounts of domain-specific data.
There are several transfer learning components present in the system:
1. Pretrained roBERTa transformer used as the encoder
1. Pretrained roBERTa subword tokenizer
1. Pretrained roBERTa token embeddings

## Evaluation
Perplexity on the validation set is used to evaluate model performance.

## Future work

There are several alternative systems which would likely outperform the one developed in this exercise. These more powerful approaches were not pursued due to time constraints imposed by the challenge. In a real-world scenario I would recommend pursuing and experimenting with the following approache: a seq2seq architecture consisting of large transformers (eg, roBERTa, T5, etc) for both the encoder and decoder, each initialized from LM pretraining.


Possible improvements on the existing system:
1. Collect more data. More web scraping, crowd-sourcing, etc.
1. Utilize weak supervision and active learning techniques to collect more labeled examples
1. Explore various ways to improve reflection mining in order to improve label quality
1. Replace existing attention mechanism with multi-headed attention
1. Experiment with varying degrees of teach forcing
1. Experiment with a sampling-based generation scheme with varying degrees of sampling temperature
1. Experiment with more sophisitcated algorithms for generation, such as beam search
1. Adding additional layers to the LSTM
1. Increasing the size of the LSTM
1. There are some inefficiencies in the existing attention mechanism that could be resolved to reduce computation/memory complexity
1. As a post-processing step, run a sentence parser over the output to detect poor quality generations before displaying to user 
