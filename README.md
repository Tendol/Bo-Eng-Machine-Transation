# Tibetan To Englsih Machine Translation 

This project is the first step in the long-term project to create a Tibetan to English machine translation. The project aims to contribute to the preservation of Tibetan language and Tibetan Buddhism.

We used SentencePiece, a language independent subword tokenizer, to tokenize Tibetan and English dataset, and T5 encoder-decoder model to train Tibetan to English Machine Translation.

We simultaneously ran three different approaches to train our data: 
 
 * We imported a transformer model from FairSeq, an open-source sequence modeling toolkit, and fed it with our own data and hyperparameters for training. 

 * We fine-tuned a pre-trained T5 transformer model provided by `Huggingface Transformers` library with our data. 

 * We built a transformer from scratch with Pytorch. Tutorial link: https://lionbridge.ai/articles/transformers-in-nlp-creating-a-translator-model-from-scratch/ 

