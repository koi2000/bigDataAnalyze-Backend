from gensim.similarities import WmdSimilarity
import jieba
from gensim.models import Word2Vec

corpus = []
documents = []
# 停用词载入
stopwords = []
# stopword = open('data/stopword.txt', 'r', encoding='utf-8')
# for line in stopword:
#     stopwords.append(line.strip())
dataPath = "bigdata/WMD/data/song.txt"
# dataPath = "./data/song.txt"
modelPath = "bigdata/WMD/checkpoint/song.model"
# modelPath = "./checkpoint/song.model"
# 加载模型
model = Word2Vec.load(modelPath)
# 已经分好词并且已经去掉停用词的训练集文件
# f = open(r'./data/song.txt_cut.txt', 'r', encoding='utf-8')
# lines = f.readlines()
# # 建立语料库list文件（list中是已经分词后的）
# for each in lines:
#     text = list(each.replace('\n', '').split(' '))
#     # pri
#     nt(text)
#     corpus.append(text)
# print(len(corpus))l
#
# 未分词的原始train文件
# 建立相对应的原始语料库语句list文件（未分词）
with open(dataPath, 'r', encoding='utf-8') as f_1:
    f_1.readline()
    for test_line in f_1:
        # print(test_line)
        test_line = test_line.replace("\n", "")
        documents.append(test_line)


def userQuery(sentence, precision, numbest=10):
    mp = {}
    for sen in documents[:precision]:
        distance = model.wv.wmdistance(sentence, sen)
        mp[sen] = distance

    after = sorted(mp.items(), key=lambda item: item[1])
    return after[:numbest]


if __name__ == '__main__':
    ret = userQuery("漫卷诗书喜欲狂", 100)
    print(ret)
