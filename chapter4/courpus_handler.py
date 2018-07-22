# -*- coding: utf-8 -*-
# @Time         : 2018-07-22 20:14
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : courpus_handler.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce


def tag_line(words, mark):
    chars = []
    tags = []
    temp_word = ''  # 用于合并组合词
    for word in words:
        word = word.strip('\t ')
        w, h = word.split('/')
        if len(w) == 0:
            continue
        if temp_word == '':
            bracket_start = word.find('[')
            if bracket_start == -1:  # 未找到括号[，说明不是组合词
                chars.extend(w)
                if h == 'ns':
                    tags += ['S'] if len(w) == 1\
                        else ['B'] + ['M'] * (len(w)-2) + ['E']
                else:
                    tags += ['O'] * len(w)
            else:  # 找到了左括号[，进入组合词
                w = w[bracket_start+1:]
                temp_word += w
        else:
            bracket_end = word.find(']')
            if bracket_end == -1:  # 未找到右括号，仍在组合词中
                temp_word += w
            else:
                w = temp_word + w
                h = word[bracket_end+1:]  # 组合词结束之后会有标注
                chars.extend(w)
                if h == 'ns':
                    tags += ['S'] if len(w) == 1\
                        else ['B'] + ['M'] * (len(w)-2) + ['E']
                else:
                    tags += ['O'] * len(w)
                temp_word = ''

    assert temp_word == ''
    return chars, tags


def corpus_handler(corpus_path):
    train_path = './data/train.txt'
    with open(corpus_path, encoding='utf8') as corpus_f,\
            open('./data/train.txt', 'w', encoding='utf8') as train_f,\
            open('./data/test.txt', 'w', encoding='utf8') as test_f:
        pos = 0  # 用于划分训练集和测试集
        for line in corpus_f:
            line = line.strip('\r\n\t ')
            if line == '':
                continue
            is_test = True if (pos % 5 == 0) else False  # 20%作为测试集
            words = line.split()[1:]  # 第一列为编号，去掉
            if len(words) == 0:
                continue
            line_chars, line_tags = tag_line(words, pos)
            save_obj = test_f if is_test else train_f
            for k, v in enumerate(line_chars):
                save_obj.write(v + '\t' + line_tags[k] + '\n')
            save_obj.write('\n')
            pos += 1


if __name__ == '__main__':
    corpus_handler(r'D:\codes\python\learning-nlp\chapter-4\data'
                   r'\people-daily.txt')
