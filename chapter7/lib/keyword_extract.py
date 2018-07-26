# -*- coding: utf-8 -*-
# @Time         : 2018-07-26 17:40
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : keyword_extract.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

from jieba import analyse


def keyword_extract(data):
    """
    使用jieba的analyse库来提取关键词
    :param data:
    :return:
    """
    tfidf = analyse.extract_tags
    keywords = tfidf(data)
    return keywords


def get_keyword(doc_path, save_path):
    """
    按行提取doc_path中的关键词并保存在save_path中。
    保存时每个关键词后接一个空格
    :param doc_path:
    :param save_path:
    :return:
    """
    with open(doc_path, 'r', encoding='utf8') as docf,\
            open(save_path, 'w', encoding='utf8') as outf:
        for data in docf:
            data = data[: len(data)-1]
            keyword = keyword_extract(data)
            for word in keyword:
                outf.write(word + ' ')
            outf.write('\n')
