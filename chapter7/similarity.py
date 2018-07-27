# -*- coding: utf-8 -*-
# @Time         : 2018-07-26 17:46
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : similarity.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import codecs
import jieba
import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from chapter7.lib.keyword_extract import get_keyword

wordvec_size = 192  # 每个词向量的大小


# noinspection PyBroadException
def get_char_pos(string, char):
    """
    找出string中char的索引
    :param string: 待查找字符串
    :param char: 目标字符
    :return: 索引列表
    """
    chPos = []
    try:
        chPos = list((pos for pos, val in enumerate(string) if (val == char)))
    except:
        pass
    return chPos


def word2vec(file_name, model_):
    """
    计算file_name中的词的词向量。
    如果在model_中能找到就设置为model_中相应的词向量，否则设置为默认的全0向量
    :param file_name:
    :param model_:
    :return:
    """
    with codecs.open(file_name, 'r', encoding='utf8') as f:
        word_vec_all = np.zeros(wordvec_size)
        for data in f:
            # 通过提前设置一个假的空格索引 -1 来处理第一个词的索引
            space_pos = [-1]
            space_pos.extend(get_char_pos(data, ' '))

            # 书上的代码有错。
            # 已改成从space_pos[i]+1开始切片，否则会包含空格，
            # 使得在model中查找不到这个词，导致相似度偏低
            for i in range(len(space_pos)-1):
                word = data[space_pos[i]+1: space_pos[i+1]]

                if model_.__contains__(word):
                    # print("No.%d word: %s" % (i, word))
                    word_vec_all += model_[word]

        return word_vec_all


def cal_similarity(v1, v2):
    """
    计算v1和v2的余弦相似性
    :param v1:
    :param v2:
    :return:
    """
    v1_mod = np.sqrt(v1.dot(v1))
    v2_mod = np.sqrt(v2.dot(v2))
    if v1_mod * v2_mod != 0:
        sim_ = v1.dot(v2) / (v1_mod * v2_mod)
    else:
        sim_ = 0
    return sim_


def doc2vec(file_name, model_):
    """
    读入file_name中的句子然后利用jieba分词，最后利用训练好的model推断分词列表的
    向量表示
    :param file_name: 待处理文档
    :param model_: 训练好的doc2vec模型
    :return:
    """
    doc = [w for x in codecs.open(file_name, 'r', encoding='utf8').readlines()
           for w in jieba.cut(x.strip())]
    doc_vec_all = model_.infer_vector(doc, alpha=0.01, steps=1000)
    return doc_vec_all


if __name__ == '__main__':
    model1 = Word2Vec.load('./data/zhwiki_news.word2vec')
    p1 = './data/P1.txt'
    p2 = './data/P2.txt'
    p1_keywords = './data/P1_keywords.txt'
    p2_keywords = './data/P2_keywords.txt'
    get_keyword(p1, p1_keywords)  # 提取p1和p2的关键词，保存在keywords路径中
    get_keyword(p2, p2_keywords)

    p1_vec = word2vec(p1_keywords, model1)  # 获取p1和p2的词向量
    p2_vec = word2vec(p2_keywords, model1)

    sim = cal_similarity(p1_vec, p2_vec)  # 利用生成的词向量计算p1和p2的相似度
    print("word2vec similarity: %f" % sim)

    model2 = Doc2Vec.load('./data/zhwiki_news.doc2vec')
    p1_doc_vec = doc2vec(p1, model2)
    p2_doc_vec = doc2vec(p2, model2)
    sim = cal_similarity(p1_doc_vec, p2_doc_vec)
    print("doc2vec similarity: %f" % sim)
