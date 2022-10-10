import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from torch import optim

class Config(object):
    use_gpu = False
    device = "cpu"
    poetrydata = "./data/song_4.txt"
    worddata = "./data/text8.train.txt"
    dataPath = "./data/two.txt"
    npData = "bigdata/word2vec/data/song.npz.npy"
    modelPath = "bigdata/word2vec/checkpoint2/embedding-50000.th"
    # 窗口大小
    C = 3
    # 负采样样本倍数
    K = 15
    # 训练轮数
    epochs = 3
    MAX_POETRY_VOCAB_SIZE = 10000
    MAX_WORD_VOCAB_SIZE = 10000
    EMBEDDING_SIZE = 500
    batch_size = 128
    lr = 0.2
    momentum = 0.9