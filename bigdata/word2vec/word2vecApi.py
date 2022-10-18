from gensim.models.word2vec import Word2Vec

# model = Word2Vec.load('./checkpoint2/gensim_w2v_sg0_model')  # 调用模型
model = Word2Vec.load("bigdata/word2vec/checkpoint2/gensim_w2v_sg0_model")  # 调用模型


def query_nearestWord(word, num=10):
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    # 计算某个词的相关词列表
    top_n = model.wv.most_similar(word, topn=num)  # 最相关的几个词
    res = []
    for item in top_n:
        temp = [item[0], round(item[1], 3)]
        res.append(temp)
    return res


if __name__ == '__main__':
    top_n = query_nearestWord("喜")
    print("top_n格式:", top_n)
