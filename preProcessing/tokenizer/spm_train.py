#!/usr/bin/env python

import sys

import sentencepiece as spm


if __name__ == "__main__":
    spm.SentencePieceTrainer.Train(" ".join(sys.argv[1:]))
