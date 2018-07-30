# -*- coding: utf-8 -*-
# @Time         : 2018-07-30 21:30
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : preprocessing.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import jieba

stopword_file = './data/stop_words.utf8'


class Processor:
    __PAD__ = 0
    __GO__ = 1
    __EOS__ = 2
    __UNK__ = 3
    vocab = ['__PAD__', '__GO__', '__EOS__', '__UNK__']

    def __init__(self):
        self.encoder_file = './data/Q.txt'
        self.decoder_file = './data/A.txt'

    def word_to_vocabulary(self, origin_file, vocab_file, segment_file):
        """
        从语料库中提取词典
        :param origin_file:
        :param vocab_file:
        :param segment_file:
        :return:
        """
        vocabulary = []
        seg_f = open(segment_file, 'w', encoding='utf8')
        with open(stopword_file, 'r', encoding='utf8') as f:
            stopwords = [i.strip() for i in f.readlines()]

        with open(origin_file, 'r', encoding='utf8') as en:
            for sent in en.readlines():
                if 'enc' in segment_file:
                    sentence = sent.strip()
                    words = jieba.lcut(sentence)
                else:
                    words = jieba.lcut(sent.strip())

                # 去除停用词(停用词中包含了标点符号)
                words = [word for word in words if word not in stopwords]
                vocabulary.extend(words)
                for word in words:
                    seg_f.write(word + ' ')
                seg_f.write('\n')

        seg_f.close()

        # 对词典去重并排序，然后写入词典文件中
        vocab_f = open(vocab_file, 'w', encoding='utf8')
        _vocabulary = list(set(vocabulary))
        _vocabulary.sort(key=vocabulary.index)
        _vocabulary = self.vocab + _vocabulary
        for index, word in enumerate(_vocabulary):
            vocab_f.write(word + '\n')

        vocab_f.close()

    @staticmethod
    def to_vec(segment_file, vocab_file, done_file):
        """
        将分词结果转换成向量形式，每个词对应词典中的下标。即将词转换成数字
        :param segment_file: 分词结果，待向量化的文本
        :param vocab_file: 词典
        :param done_file: 向量化结果的保存路径
        :return:
        """
        word_dicts = {}
        vec = []
        with open(vocab_file, 'r', encoding='utf8') as f:
            for index, word in enumerate(f.readlines()):
                word_dicts[word.strip()] = index

        f = open(done_file, 'w', encoding='utf8')
        with open(segment_file, 'r', encoding='utf8') as seg_f:
            for sent in seg_f.readlines():
                # 最后一个字符为空格所以排除
                sents = [i.strip() for i in sent.split(' ')[: -1]]
                vec.extend(sents)
                for word in sents:
                    f.write(str(word_dicts.get(word, -1)) + ' ')
                f.write('\n')

        f.close()

    def run(self):
        # 构造编码、解码词典
        self.word_to_vocabulary(self.encoder_file, './data/enc.vocab',
                                './data/enc.segment')
        self.word_to_vocabulary(self.decoder_file, './data/dec.vocab',
                                './data/dec.segment')

        # 将Q和A的语料转换成向量形式
        self.to_vec('./data/enc.segment', './data/enc.vocab',
                    './data/enc.vec')
        self.to_vec('./data/dec.segment', './data/dec.vocab',
                    './data/dec.vec')


process = Processor()
process.run()
