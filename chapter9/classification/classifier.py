# -*- coding: utf-8 -*-
# @Time         : 2018-07-28 23:31
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : classifier.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import gensim
import jieba
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression

from chapter9.classification.normalization import normalize_corpus
from chapter9.classification.feature_extractors import bow_extractor,\
                                                        tfidf_extractor


def get_data():
    """
    将正常的邮件和垃圾邮件合并成一个语料，合并label
    :return:
    """
    with open('./data/ham_data.txt', 'r', encoding='utf8') as ham_f,\
            open('./data/spam_data.txt', 'r', encoding='utf8') as spam_f:
        ham_data = ham_f.readlines()
        spam_data = spam_f.readlines()

        ham_label = np.ones(len(ham_data)).tolist()
        spam_label = np.zeros(len(spam_data)).tolist()

        corpus = ham_data + spam_data
        labels = ham_label + spam_label

    return corpus, labels


def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    """
    使用sklearn中自带的train_test_split函数对数据集进行划分，得到训练集和测试集
    :param corpus: 总数据特征集
    :param labels: 总数据label
    :param test_data_proportion: 测试集所占比例
    :return: 划分结果
    """
    train_X, test_X, train_Y, test_Y = train_test_split(
        corpus, labels, test_size=test_data_proportion, random_state=42)
    return train_X, train_Y, test_X, test_Y


def remove_empty_docs(corpus, labels):
    """
    移除语料库中空的文档
    :param corpus:
    :param labels:
    :return:
    """
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():  # 判断该文档是否为空
            filtered_corpus.append(doc)
            filtered_labels.append(label)

    return filtered_corpus, filtered_labels


def get_metrics(true_labels, predicted_labels):
    """
    分别计算预测结果的准确率、精确率、召回率、F1值，直接打印出这些结果
    :param true_labels: 真实label
    :param predicted_labels: 预测结果
    :return:
    """
    print("accuracy:", np.round(metrics.accuracy_score(true_labels,
                                                       predicted_labels), 2))
    print("precision:", np.round(metrics.precision_score(
        true_labels, predicted_labels, average='weighted'), 2))
    print("recall:", np.round(metrics.recall_score(
        true_labels, predicted_labels, average='weighted'), 2))
    print("f1 score:", np.round(metrics.f1_score(
        true_labels, predicted_labels, average='weighted'), 2))


def train_predict_evaluate_model(classifier, train_features, train_labels,
                                 test_features, test_labels):
    """
    训练、预测、评估 模型
    :param classifier: 模型
    :param train_features: 训练集特征
    :param train_labels: 训练集label
    :param test_features: 测试集特征
    :param test_labels: 测试集label
    :return: 预测结果
    """
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    get_metrics(true_labels=test_labels, predicted_labels=predictions)
    return predictions


def main():
    corpus, labels = get_data()
    print("total data size:", len(labels))
    corpus, labels = remove_empty_docs(corpus, labels)
    print("sample:", corpus[10])
    print("label of sample:", labels[10])
    label_name_map = ['spam', 'normal']  # 0代表spam，1代表normal
    print("actual type:", label_name_map[int(labels[10])])

    # 划分数据集
    train_corpus, train_labels, test_corpus, test_labels =\
        prepare_datasets(corpus, labels)
    # 对语料进行预处理
    norm_train_corpus = normalize_corpus(train_corpus)
    norm_test_corpus = normalize_corpus(test_corpus)

    # 词袋模型
    bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)
    bow_test_features = bow_vectorizer.transform(norm_test_corpus)

    # tfidf模型
    tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
    tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)

    # 对处理后的语料进行分词
    tokenized_train = [jieba.lcut(text) for text in norm_train_corpus]
    tokenized_test = [jieba.lcut(text) for text in norm_test_corpus]

    # 词向量Word2Vec
    model = gensim.models.Word2Vec(tokenized_train, size=500, window=100,
                                   min_count=30, sample=1e-3)

    # 分别以多项分布朴素贝叶斯、SVM、逻辑回归算法训练分类器并评估各个分类器性能
    mnb = MultinomialNB()
    svm = SGDClassifier()
    lr = LogisticRegression()

    print("\nNavie Bayes based on BOW")
    mnb_bow_predictions = train_predict_evaluate_model(
        classifier=mnb,
        train_features=bow_train_features, train_labels=train_labels,
        test_features=bow_test_features, test_labels=test_labels)

    print("\nLogistic Regression based on BOW")
    lr_bow_predictions = train_predict_evaluate_model(
        classifier=lr,
        train_features=bow_train_features, train_labels=train_labels,
        test_features=bow_test_features, test_labels=test_labels)

    print("\nSVM based on BOW")
    svm_bow_predictions = train_predict_evaluate_model(
        classifier=svm,
        train_features=bow_train_features, train_labels=train_labels,
        test_features=bow_test_features, test_labels=test_labels)

    print("\nNavie Bayes based on tfidf")
    mnb_tfidf_predictions = train_predict_evaluate_model(
        classifier=mnb,
        train_features=tfidf_train_features, train_labels=train_labels,
        test_features=tfidf_test_features, test_labels=test_labels)

    print("\nLogistic Regression based on tfidf")
    lr_tfidf_predictions = train_predict_evaluate_model(
        classifier=lr,
        train_features=tfidf_train_features, train_labels=train_labels,
        test_features=tfidf_test_features, test_labels=test_labels)

    print("\nSVM based on tfidf")
    svm_tfidf_predictions = train_predict_evaluate_model(
        classifier=svm,
        train_features=tfidf_train_features, train_labels=train_labels,
        test_features=tfidf_test_features, test_labels=test_labels)


if __name__ == '__main__':
    main()
