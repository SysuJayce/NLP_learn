# -*- coding: utf-8 -*-
# @Time         : 2018-07-26 17:46
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : word2vec_sim.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import codecs
import numpy as np
from gensim.models import Word2Vec
from chapter7.lib.keyword_extract import get_keyword

wordvec_size = 192


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
    with codecs.open(file_name, 'r', encoding='utf8') as f:
        word_vec_all = np.zeros(wordvec_size)
        for data in f:
            space_pos = [-1]
            space_pos.extend(get_char_pos(data, ' '))

            for i in range(len(space_pos)-1):
                word = data[space_pos[i]+1: space_pos[i+1]]

                if model_.__contains__(word):
                    # print("No.%d word: %s" % (i, word))
                    word_vec_all += model_[word]

        return word_vec_all


def similarity(v1, v2):
    v1_mod = np.sqrt(v1.dot(v1))
    v2_mod = np.sqrt(v2.dot(v2))
    if v1_mod * v2_mod != 0:
        sim_ = v1.dot(v2) / (v1_mod * v2_mod)
    else:
        sim_ = 0
    return sim_


if __name__ == '__main__':
    model = Word2Vec.load('./data/zhwiki_news.word2vec')
    p1 = './data/P1.txt'
    p2 = './data/P2.txt'
    p1_keywords = './data/P1_keywords.txt'
    p2_keywords = './data/P2_keywords.txt'
    get_keyword(p1, p1_keywords)
    get_keyword(p2, p2_keywords)

    p1_vec = word2vec(p1_keywords, model)
    p2_vec = word2vec(p2_keywords, model)

    sim = similarity(p1_vec, p2_vec)
    print(sim)
