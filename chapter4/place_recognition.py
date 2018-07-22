# -*- coding: utf-8 -*-
# @Time         : 2018-07-22 18:03
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : place_recognition.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import os
# import CRFPP


def f1(path):
    """
    计算F1值
    F1 = 2 * precision * recall / (precision + recall)
    :param path: 测试集和测试结果的保存路径
    :return:
    """
    total_tag_num = 0   # 测试集中标记总个数
    total_loc_tag_num = 0  # 测试集中属于地理位置的标记总个数
    pred_loc_tag_num = 0  # 预测为地理位置的标记个数
    correct_tag_num = 0  # 正确预计的标记个数(包括地理位置的和非地理位置的标记)
    correct_loc_tag_num = 0  # 正确预计的地理位置标记个数
    states = ['B', 'M', 'E', 'S']

    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            total_tag_num += 1  # 每一行就有一个真实标记和预测标记
            _, real, pred = line.split()

            if pred == real:
                correct_tag_num += 1  # 预测准确
                if real in states:
                    correct_loc_tag_num += 1  # 预测的地理位置准确
            if real in states:
                total_loc_tag_num += 1  # 总的地理位置标记数
            if pred in states:
                pred_loc_tag_num += 1  # 预测的地理位置标记数

        precision = 1.0 * correct_loc_tag_num / pred_loc_tag_num  # 准确率
        recall = 1.0 * correct_loc_tag_num / total_loc_tag_num  # 召回率
        f1_score = 2 * precision * recall / (precision + recall)  # F1值
        print("Precision:\t{0}, Recall:\t{1}, F1 Score:\t{2}".format(
            precision, recall, f1_score))


def load_model(path):
    # 在windows下要自行变异CRFPP，太麻烦所以就不搞了
    # -v 3: access deep information like alpha,beta,prob
    # -nN: enable nbest output. N should be >= 2
    if os.path.exists(path):
        return CRFPP.Tagger('-m {0} -v 3 -n2'.format(path))
    return None


def locationNER(text):
    tagger = load_model('./model')
    for c in text:
        tagger.add(c)

    result = []
    # parse and change internal stated as 'parsed'
    tagger.parse()
    word = ''
    for i in range(0, tagger.size()):
        for j in range(0, tagger.xsize()):
            ch = tagger.x(i, j)
            tag = tagger.y2(i)
            if tag == 'B':
                word = ch
            elif tag == 'M':
                word += ch
            elif tag == 'E':
                word += ch
                result.append(word)
            elif tag == 'S':
                word = ch
                result.append(word)

    return result


if __name__ == '__main__':
    f1('./data/test.rst')
    # text = '我中午要去北京饭店，下午去中山公园，晚上回亚运村。'
    # print(text, locationNER(text), sep='==> ')
    #
    # text = '我去回龙观，不去南锣鼓巷'
    # print(text, locationNER(text), sep='==> ')
    #
    # text = '打的去北京南站地铁站'
    # print(text, locationNER(text), sep='==> ')
