# -*- coding: utf-8 -*-
# @Time         : 2018-07-23 16:04
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : keyword_extract.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import math
import jieba
import jieba.posseg as psg
from jieba import analyse
import numpy as np
from gensim import corpora, models
import functools


def get_stopword_list(stopword_path='./data/stopword.txt'):
    result = []
    with open(stopword_path, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            result.append(line)
    return result


def seg_to_list(sentence, pos=False):
    if not pos:
        seg_list = jieba.cut(sentence)
    else:
        seg_list = psg.cut(sentence)
    return seg_list


def word_filter(seg_list, pos=False):
    stopword_list = get_stopword_list()
    filter_list = []
    for seg in seg_list:
        if not pos:
            word = seg
        else:
            if not seg.flag.startswith('n'):
                continue
            word = seg.word
        if not word in stopword_list and len(word) > 1:
            filter_list.append(word)

    return filter_list


def load_data(pos=False, corpus_path='./data/corpus.txt'):
    doc_list = []
    with open(corpus_path, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            seg_list = seg_to_list(line, pos)
            filter_list = word_filter(seg_list, pos)
            doc_list.append(filter_list)

    return doc_list


def train_idf(doc_list):
    idf_dic = {}
    for doc in doc_list:
        for word in set(doc):
            idf_dic[word] = idf_dic.get(word, 0) + 1.0

    for k, v in idf_dic.items():
        idf_dic[k] = math.log(len(doc_list) / (1.0 + v))

    default_idf = math.log(len(doc_list))
    return idf_dic, default_idf


def cmp(e1, e2):
    """
    比较tf-idf值的大小，如果一样大的时候就通过比较对应的词字典序
    :param e1:
    :param e2:
    :return:
    """
    res = np.sign(e1[1] - e2[1])  # 返回+1 / -1 / 0，代表括号内参数运算后的符号
    if res != 0:
        return res
    else:  # tf-idf值相同时比较词的字典序
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1
        elif a == b:
            return 0
        else:
            return -1


class TfIdf:
    def __init__(self, idf_dic, default_idf, word_list, keyword_num):
        self.word_list = word_list
        self.idf_dic = idf_dic
        self.default_idf = default_idf
        self.tf_dic = self.get_tf_dic()
        self.keyword_num = keyword_num

    def get_tf_dic(self):
        tf_dic = {}
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0) + 1.0

        for k, v in tf_dic.items():
            tf_dic[k] = float(v) / len(self.word_list)

        return tf_dic

    def get_tfidf(self):
        tfidf_dic = {}
        for word in self.word_list:
            idf = self.idf_dic.get(word, self.default_idf)
            tf = self.tf_dic.get(word, 0.0)

            tfidf = tf * idf
            tfidf_dic[word] = tfidf

        for k, v in sorted(tfidf_dic.items(), key=functools.cmp_to_key(cmp),
                           reverse=True)[:self.keyword_num]:
            print(k + '/ ', end='')
        print()


class TopicModel:
    def __init__(self, doc_list, keyword_num, model='LSI', num_topics=4):
        self.dictionary = corpora.Dictionary(doc_list)
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]

        self.keyword_num = keyword_num
        self.num_topics = num_topics

        if model == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()

        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    def train_lsi(self):
        return models.LsiModel(self.corpus_tfidf, id2word=self.dictionary,
                               num_topics=self.num_topics)

    def train_lda(self):
        return models.LdaModel(self.corpus_tfidf, id2word=self.dictionary,
                               num_topics=self.num_topics)

    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}
        for word in word_dic:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic

        return wordtopic_dic

    @staticmethod
    def word_dictionary(doc_list):
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)

        dictionary = list(set(dictionary))

        return dictionary

    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        def cal_sim(A, B):
            sum_a, sum_b, product = 0.0, 0.0, 0.0
            for a, b in zip(A, B):
                Ai = a[1]
                Bi = b[1]
                sum_a += Ai * Ai
                sum_b += Bi * Bi
                product += Ai * Bi

            sim = 0.0
            if sum_a * sum_b != 0.0:
                sim = product / math.sqrt(sum_a * sum_b)

            return sim

        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = cal_sim(v, senttopic)
            sim_dic[k] = sim

        for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(cmp),
                           reverse=True):
            print(k + '/ ', end='')
        print()


def tfidf_extract(word_list, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    idf_dic, default_idf = train_idf(doc_list)
    train_model = TfIdf(idf_dic, default_idf, word_list, keyword_num)
    train_model.get_tfidf()


def textrank_extract(text, pos=False, keyword_num=10):
    textrank = analyse.textrank
    keywords = textrank(text, keyword_num)
    for keyword in keywords:
        print(keyword + '/ ', end='')
    print()


def topic_extract(word_list, model, pos=False, keyword_num=10):
    doc_list = load_data(pos)
    topic_model = TopicModel(doc_list, keyword_num, model=model)
    topic_model.get_simword(word_list)


if __name__ == '__main__':
    with open('./data/test.txt', encoding='utf8') as f:
        text = f.read()
        pos = True
        seg_list = seg_to_list(text, pos)
        filter_list = word_filter(seg_list, pos)

        print("TF-IDF模型结果：")
        tfidf_extract(filter_list)
        print("TextRank模型结果：")
        textrank_extract(text)
        print("LSI模型结果：")
        topic_extract(filter_list, 'LSI', pos)
        print("LDA模型结果：")
        topic_extract(filter_list, 'LDA', pos)
