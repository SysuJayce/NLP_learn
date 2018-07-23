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
import functools  # 用于在python3的sorted中使用自定义排序函数


def get_stopword_list(stopword_path='./data/stopword.txt'):
    """
    停用词数据集中每一行就是一个停用词
    :param stopword_path: 停用词路径
    :return: 停用词列表
    """
    result = []
    with open(stopword_path, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            result.append(line)
    return result


def seg_to_list(sentence, pos=False):
    """
    使用jieba进行分词。
    :param sentence: 待分词的语句
    :param pos: 是否基于词性标注进行分词
    :return: 分词结果
    """
    if not pos:
        seg_list = jieba.cut(sentence)
    else:
        seg_list = psg.cut(sentence)
    return seg_list


def word_filter(seg_list, pos=False):
    """
    分词过滤器，把停用词过滤掉
    :param seg_list: 分词列表
    :param pos: 该分词列表是否含有词性标注
    :return: 过滤掉停用词之后的分词列表
    """
    stopword_list = get_stopword_list()
    filter_list = []
    for seg in seg_list:
        if not pos:
            word = seg
        else:
            # 如果带有词性标注，那么只保留与名词有关的部分
            if not seg.flag.startswith('n'):
                continue
            word = seg.word
        if word not in stopword_list and len(word) > 1:  # 一个词的长度需>=2
            filter_list.append(word)

    return filter_list


def load_data(pos=False, corpus_path='./data/corpus.txt'):
    """
    加载语料库
    :param pos: 是否使用词性标注
    :param corpus_path: 语料库路径
    :return: 过滤后的语料库分词结果
    """
    doc_list = []
    with open(corpus_path, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            seg_list = seg_to_list(line, pos)
            filter_list = word_filter(seg_list, pos)
            doc_list.append(filter_list)

    return doc_list


def train_idf(doc_list):
    """
    计算每个词对应的idf
    :param doc_list: 文档列表
    :return: 文档中的所有词的idf以及生词的idf(默认idf)
    """
    idf_dic = {}
    # 计算每个词出现在多少个文档中
    for doc in doc_list:
        for word in set(doc):
            idf_dic[word] = idf_dic.get(word, 0) + 1.0

    # idf计算公式： idf(x) = log(文档总数/含有x的文档数)
    for k, v in idf_dic.items():
        idf_dic[k] = math.log(len(doc_list) / (1.0 + v))  # 分母加一平滑

    default_idf = math.log(len(doc_list))  # 设置默认idf值用来作为生词的idf
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
        """
        TF-IDF类的构造函数
        :param idf_dic: 训练好的idf词典
        :param default_idf: 默认idf值
        :param word_list: 经处理的(过滤停用词)待提取关键词文本列表
        :param keyword_num: 关键词数量
        """
        self.word_list = word_list
        self.idf_dic = idf_dic
        self.default_idf = default_idf
        self.tf_dic = self.get_tf_dic()
        self.keyword_num = keyword_num

    def get_tf_dic(self):
        """
        计算输入的文本列表所有词的tf值
        :return:
        """
        tf_dic = {}
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0) + 1.0

        for k, v in tf_dic.items():
            tf_dic[k] = float(v) / len(self.word_list)

        return tf_dic

    def get_tfidf(self):
        """
        计算输入的文本列表所有词的tf-idf值：tf * idf
        :return: 打印前keyword_num个待提取关键词文本的提取关键词结果
        """
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
        """
        :param doc_list: 经处理(去除停用词)的待提取文档列表。
        一个列表就是一个文档，列表元素为文档中的词
        :param keyword_num: 关键词数量
        :param model: 具体模型： LSI, LDA
        :param num_topics: 主题数量
        """
        # 生成词典，用于将列表向量化
        self.dictionary = corpora.Dictionary(doc_list)
        # 使用BOW模型，利用上面生成的词典将输入列表向量化
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]

        # 使用向量化后的语料生成TF-IDF模型
        self.tfidf_model = models.TfidfModel(corpus)

        # 利用上面生成的TF-IDF模型将向量化后的语料转为TF-IDF模式表示的向量
        # 也就是原本的0/1表示变成用每个词的TF-IDF值来表示
        self.corpus_tfidf = self.tfidf_model[corpus]
        self.keyword_num = keyword_num  # 关键词数量
        self.num_topics = num_topics  # 主题数量

        if model == 'LSI':  # 加载所选模型
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()

        # 获得词典中所有词对应的主题
        self.wordtopic_dic = self.get_wordtopic(doc_list)

    def train_lsi(self):
        """
        使用构造好的TF-IDF模式的向量以及原始词典来生成一个LSI模型
        :return:
        """
        return models.LsiModel(self.corpus_tfidf, id2word=self.dictionary,
                               num_topics=self.num_topics)

    def train_lda(self):
        """
        使用构造好的TF-IDF模式的向量以及原始词典来生成一个LDA模型
        :return:
        """
        return models.LdaModel(self.corpus_tfidf, id2word=self.dictionary,
                               num_topics=self.num_topics)

    def get_wordtopic(self, input_doc_list):
        """
        需要注意主题不等于关键词，关键词是从主题中选取的。
        文本与词的主题分布相似度最高的前keyword_num个就是关键词
        :param input_doc_list: 待提取主题的词典。
        :return: 词典中所有词对应的主题
        {word1: [topic1 topic2], word2: [topic1...]...}
        """
        def word_dictionary(doc_list):
            """
            从处理后的数据集构造一个词典。功能类似用gensim的corpora.Dictionary()
            :param doc_list: 处理后的数据集
            :return: 数据集含有的词的集合
            """
            dictionary = []
            for doc in doc_list:
                dictionary.extend(doc)

            dictionary = list(set(dictionary))

            return dictionary

        word_dic = word_dictionary(input_doc_list)
        wordtopic_dic = {}
        for word in word_dic:
            single_list = [word]  # gensim中利用BOW模型向量化必须使用列表
            # 将word向量化，且用TF-IDF模式表示
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            # gensim中model的需要使用以TF-IDF模式表示的向量化的数据来生成主题
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic

        return wordtopic_dic

    def get_simword(self, word_list):
        """
        计算词分布和文档分布的相似度，取相似度最高的keyword_num个词作为关键词
        :param word_list: 待提取关键词的文档列表
        :return: 打印keyword_num个关键词
        """
        # 先利用gensim把输入文档转化为向量形式
        # 再使用gensim的TF-IDF模型把向量形式转化成以TF-IDF模式表示的向量
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        # 最后用所选模型计算输入文档对应的主题
        senttopic = self.model[sentcorpus]  # 文档生成的主题senttopic

        def cal_sim(A, B):
            """
            计算余弦相似性： cos_sim(A, B) = A·B / sqrt(||A|| * ||B||)
            即：A和B的点积(对应元素相乘)再除以他们的模的积
            :param A:
            :param B:
            :return:
            """
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

        # 计算"词典中的词"生成的主题 和 "输入文档生成的主题" 的余弦相似性
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():  # 词生成的主题wordtopic_dic
            if k not in word_list:
                continue
            sim = cal_sim(v, senttopic)
            sim_dic[k] = sim

        # 上面计算的相似度排序后选keyword_num对应的词作为关键词
        for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(cmp),
                           reverse=True)[:self.keyword_num]:
            print(k + '/ ', end='')
        print()


def tfidf_extract(word_list, pos=False, keyword_num=10):
    """
    使用TF-IDF模型进行关键词提取
    :param word_list: 待提取关键词的句子
    :param pos: 是否使用词性标注
    :param keyword_num: 欲提取的关键词数量
    :return:
    """
    doc_list = load_data(pos)  # 加载语料库
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
