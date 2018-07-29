# -*- coding: utf-8 -*-
# @Time         : 2018-07-28 22:50
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : normalization.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import re
import string
import jieba

with open('./dict/stop_words.utf8', 'r', encoding='utf8') as f:
    stopword_list = f.readlines()


def tokenize_text(text):
    """
    把text分词然后加入一个list
    :param text: 待分词的文本
    :return:
    """
    tokens = jieba.cut(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def remove_special_characters(text):
    """
    对分词后得到的token列表进行过滤，过滤其中的标点符号
    :param text: 待处理的文本
    :return: 过滤掉标点符号之后的token列表
    """
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_stopwords(text):
    """
    一般在去除标点符号之后再去除停用词，这时把空格也去掉
    :param text:
    :return:
    """
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text


def normalize_corpus(corpus, tokenize=False):
    """
    输入一个语料，对语料中的每一行删除标点符号、去除停用词后拼接成一句连续的话
    :param corpus:
    :param tokenize:
    :return:
    """
    normalized_corpus = []
    for text in corpus:
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        normalized_corpus.append(text)

        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)

    return normalized_corpus
