# -*- coding: utf-8 -*-
# @Time         : 2018-07-25 23:23
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : main.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import jieba
from gensim.corpora import WikiCorpus
from chapter7.lib.langconv import *  # 开源的繁体转简体库，配合zh_wiki.py使用
import logging
from gensim.models import Word2Vec, Doc2Vec, doc2vec
from gensim.models.word2vec import LineSentence

# 设置log的格式
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class TaggedWikiDocument:
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True

    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield doc2vec.LabeledSentence(
                # 1. 对content中的每一个c，
                # 2. 转换成简体中文之后用jieba分词
                # 3. 加入到words列表中
                words=[w for c in content
                       for w in jieba.cut(Converter('zh-hans').convert(c))],
                tags=[title])


def preprocess():
    """
    使用gensim中的WikiCorpus库提取wiki的中文语料，并将繁体转成简体中文。
    然后利用jieba的分词工具将转换后的语料分词并写入一个txt
    每个wiki文档的分词结果写在新txt中的一行，词与词之间用空格隔开
    :return:
    """
    count = 0
    zhwiki_path = './data/zhwiki-latest-pages-articles.xml.bz2'
    f = open('./data/reduced_zhwiki.txt', 'w', encoding='utf8')
    wiki = WikiCorpus(zhwiki_path, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        word_list = []
        for sentence in text:
            sentence = Converter('zh-hans').convert(sentence)  # 繁体转简体
            seg_list = jieba.cut(sentence)
            for seg in seg_list:
                word_list.append(seg)
        f.write(' '.join(word_list) + '\n')
        count += 1
        if count % 200 == 0:
            print("Saved " + str(count) + ' articles')

    f.close()


def train_w2v():
    with open('./data/reduced_zhwiki.txt', 'r', encoding='utf8') as f:
        # 使用gensim的Word2Vec类来生成词向量
        model = Word2Vec(LineSentence(f), sg=0, size=192, window=5,
                         min_count=5, workers=4)
        model.save('./data/zhwiki_news.word2vec')


def test_w2v():
    model = Word2Vec.load('./data/zhwiki_news.word2vec')
    # print(model.similarity('大数据', '人工智能'))
    # print(model.similarity('滴滴', '共享单车'))
    print(model.similarity('西红柿', '番茄'))  # 相似度为0.63
    print(model.similarity('西红柿', '香蕉'))  # 相似度为0.44

    word = '中国'
    if word in model.wv.index2word:
        print(model.most_similar(word))


def train_d2v():
    """
    训练doc2vec
    :return:
    """
    docvec_size = 192
    zhwiki_path = './data/zhwiki-latest-pages-articles.xml.bz2'
    wiki = WikiCorpus(zhwiki_path, lemmatize=False, dictionary={})
    documents = TaggedWikiDocument(wiki)

    model = Doc2Vec(documents, dm=0, dbow_words=1, size=docvec_size,
                    window=8, min_count=19, iter=5, workers=4)
    model.save('./data/zhwiki_news.doc2vec')


if __name__ == '__main__':
    # preprocess()
    # train_w2v()
    # test_w2v()
    train_d2v()
