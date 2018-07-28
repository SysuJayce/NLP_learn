# -*- coding: utf-8 -*-
# @Time         : 2018-07-27 21:34
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : main.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import os
import re
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def clean_sentence(string):
    """
    将单词变成小写，并将换行符替换为空格
    :param string:
    :return:
    """
    # 正则表达式中'^'如果放在中括号里面就是排除，其他时候表示句子开头
    # 只保留字母数字和空格开头的单词，删除其他字符
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    string = string.lower().replace("<br/>", " ")
    # re.sub从字符串的左边开始匹配pattern，只要
    return re.sub(strip_special_chars, "", string.lower())


def gen_data():
    """
    生成一个评论中用到的词在词汇表中的索引的矩阵ids并保存下来
    :return:
    """
    words_list = np.load('./data/wordsList.npy').tolist()  # 读取词汇表
    # 将词汇表转换成list
    words_list = [word.decode('UTF-8') for word in words_list]
    # 分别获取积极和消极评价的文件列表。
    # 一个文件代表一个评价，文件名格式为：id_rating
    pos_files = ['./data/pos/' + f for f in os.listdir('./data/pos/')
                 if os.path.isfile(os.path.join('./data/pos/', f))]
    neg_files = ['./data/neg/' + f for f in os.listdir('./data/neg/') if
                 os.path.isfile(os.path.join('./data/neg/', f))]

    num_words = []  # 记录每个评价有多少个词
    for pf in pos_files:
        with open(pf, 'r', encoding='utf8') as f:
            # 其实每个文件包含一个评价，只有一句话
            line = f.readline()
            counter = len(line.split())
            num_words.append(counter)
    print("finished positive")

    for nf in neg_files:
        with open(nf, 'r', encoding='utf8') as f:
            line = f.readline()
            counter = len(line.split())
            num_words.append(counter)
    print("finished negative")

    num_files = len(num_words)
    print("total file num: %d" % num_files)
    print("total word num: %d" % sum(num_words))
    print("average num of word in a file:", sum(num_words) / len(num_words))

    draw_plt(num_words)  # 使用matplotlib绘制每条评论包含的单词个数

    # 从matplotlib绘制的图中看出绝大多数评论包含的单词个数在300个以内
    max_seq_num = 300
    # 生成索引矩阵
    # 每一行代表一条评论
    # 行中记录这条评论中的每个词在词典中的索引
    ids = np.zeros((num_files, max_seq_num), dtype='int32')
    file_count = 0
    unknown = 399999  # 生词的标记
    for pf in pos_files:
        with open(pf, 'r', encoding='utf8') as f:
            index_counter = 0
            line = f.readline()
            # 删除评论中的标点，只留下单词和空格
            cleaned_line = clean_sentence(line)
            split = cleaned_line.split()
            for word in split:
                try:
                    ids[file_count][index_counter] = words_list.index(word)
                except ValueError:
                    ids[file_count][index_counter] = unknown
                index_counter += 1
                if index_counter >= max_seq_num:  # 只记录前300个单词的索引
                    break
            file_count += 1

    for nf in neg_files:
        with open(nf, 'r', encoding='utf8') as f:
            index_counter = 0
            line = f.readline()
            cleaned_line = clean_sentence(line)
            split = cleaned_line.split()
            for word in split:
                try:
                    ids[file_count][index_counter] = words_list.index(word)
                except ValueError:
                    ids[file_count][index_counter] = unknown
                index_counter += 1
                if index_counter >= max_seq_num:
                    break
            file_count += 1

    np.save('./data/idsMatrix', ids)


def draw_plt(num_words_):
    import matplotlib
    matplotlib.use('qt4agg')
    import matplotlib.pyplot as plt

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    plt.hist(num_words_, 50, facecolor='g')
    plt.xlabel('文本长度')
    plt.ylabel('频次')
    plt.axis([0, 1200, 0, 8000])
    plt.show()


if __name__ == '__main__':
    gen_data()
