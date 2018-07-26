# -*- coding: utf-8 -*-
# @Time         : 2018-07-26 17:40
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : keyword_extract.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

from jieba import analyse


def keyword_extract(data):
    tfidf = analyse.extract_tags
    keywords = tfidf(data)
    return keywords


def get_keyword(doc_path, save_path):
    with open(doc_path, 'r', encoding='utf8') as docf,\
            open(save_path, 'w', encoding='utf8') as outf:
        for data in docf:
            data = data[: len(data)-1]
            keyword = keyword_extract(data)
            for word in keyword:
                outf.write(word + ' ')
            outf.write('\n')
