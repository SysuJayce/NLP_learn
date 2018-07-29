# -*- coding: utf-8 -*-
# @Time         : 2018-07-28 23:25
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : feature_extractors.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,\
                                                TfidfVectorizer


def bow_extractor(corpus, ngram_range=(1, 1)):
    """
    使用词袋模型对语料的分词结果进行向量化。这里仅仅统计词频
    :param corpus:
    :param ngram_range:
    :return:
    """
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def tfidf_transformer(bow_matrix):
    """
    将词袋模型的输出矩阵按照TF-IDF进行修正
    :param bow_matrix:
    :return:
    """
    transformer = TfidfTransformer(norm='l2', smooth_idf=True, use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix


def tfidf_extractor(corpus, ngram_range=(1, 1)):
    """
    基于TF-IDF模型对语料的分词结果进行向量化
    :param corpus:
    :param ngram_range:
    :return:
    """
    vectorizer = TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True,
                                 use_idf=True, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features
