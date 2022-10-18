import pandas as pd
import numpy as np
from collections import Counter

import scipy
import torch
from gensim.models import Word2Vec

from bigdata.word2vec.Config import Config
from bigdata.word2vec.model import EmbeddingModel

# ----超参数-----------------------------------------
# 窗口大小
C = Config.C
# 负采样样本倍数
K = Config.K
# 训练轮数
epochs = Config.epochs
MAX_VOCAB_SIZE = Config.MAX_WORD_VOCAB_SIZE
EMBEDDING_SIZE = Config.EMBEDDING_SIZE
batch_size = Config.batch_size
lr = Config.lr
momentum = Config.momentum
# ---------------------------------------------------

mp = np.load(Config.npData, allow_pickle=True)
# text = mp["text"]
# word2idx = mp["word2idx"]
# idx2word = mp['idx2word']
# word_freqs = mp['word_freqs']
# word_counts = mp['word_counts']

text = mp[0]
word2idx = mp[1]
idx2word = mp[2]
word_freqs = mp[3]
word_counts = mp[4]
# model = EmbeddingModel(len(word_freqs), Config.EMBEDDING_SIZE)
# weight = torch.load(Config.modelPath, map_location='cpu')
# model.load_state_dict(weight)

model = Word2Vec.load('./data/gensim_w2v_perpoetry_model')  # 调用模型


def find_nearest(embedding_weights, word, nums=10):
    index = word2idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [(idx2word[i], round(cos_dis[i], 3)) for i in cos_dis.argsort()[:nums]]


def test():
    model = EmbeddingModel(len(word_freqs), Config.EMBEDDING_SIZE)
    weight = torch.load("./checkpoint2/embedding-50000.th")
    model.load_state_dict(weight)
    embedding_weights = model.input_embedding()
    for word in ["喜", '怒', "哀", "乐"]:
        print(word, find_nearest(embedding_weights, word))
    # for word in ["one", 'second',"computer"]:
    #     print(word, find_nearest(embedding_weights, word))


# def query_nearestWord(wordList, num=10):
#     embedding_weights = model.input_embedding()
#     res = {}
#     for word in wordList:
#         res[word] = find_nearest(embedding_weights, word, num + 1)[1:-1]
#
#     return res
def query_nearestWord(word, num=10):
    # 计算某个词的相关词列表
    top_n = model.wv.most_similar(word, topn=num)  # 最相关的几个词
    return top_n

if __name__ == '__main__':
    print(query_nearestWord('喜'))
