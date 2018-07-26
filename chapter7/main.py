# -*- coding: utf-8 -*-
# @Time         : 2018-07-25 23:23
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : main.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import jieba
from gensim.corpora import WikiCorpus
from chapter7.lib.langconv import *
import logging
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def preprocess():
    count = 0
    zhwiki_path = './data/zhwiki-latest-pages-articles.xml.bz2'
    f = open('./data/reduced_zhwiki.txt', 'w', encoding='utf8')
    wiki = WikiCorpus(zhwiki_path, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        word_list = []
        for sentence in text:
            sentence = Converter('zh-hans').convert(sentence)
            seg_list = jieba.cut(sentence)
            for seg in seg_list:
                word_list.append(seg)
        f.write(' '.join(word_list) + '\n')
        count += 1
        if count % 200 == 0:
            print("Saved " + str(count) + ' articles')

    f.close()


def train():
    with open('./data/reduced_zhwiki.txt', 'r', encoding='utf8') as f:
        model = Word2Vec(LineSentence(f), sg=0, size=192, window=5,
                         min_count=5, workers=4)
        model.save('./data/zhwiki_news.word2vec')


def test():
    model = Word2Vec.load('./data/zhwiki_news.word2vec')
    # print(model.similarity('大数据', '人工智能'))
    # print(model.similarity('滴滴', '共享单车'))
    print(model.similarity('西红柿', '番茄'))  # 相似度为0.63
    print(model.similarity('西红柿', '香蕉'))  # 相似度为0.44

    word = '中国'
    if word in model.wv.index2word:
        print(model.most_similar(word))


if __name__ == '__main__':
    # preprocess()
    # train()
    test()
