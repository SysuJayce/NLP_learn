# -*- coding: utf-8 -*-
import os
import pickle
import jieba
import glob
import random
import jieba.analyse as aly


class IMM:
    def __init__(self, dic_path):
        self.dictionary = set()
        self.maximum = 0

        # 读取词典
        with open(dic_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()  # 去除空白行
                if not line:
                    continue
                self.dictionary.add(line)
                self.maximum = max(self.maximum, len(line))

    def cut(self, text):
        result = []  # 分词结果
        index = len(text)  # 待分词的字符串长度
        while index > 0:  # 循环指导整个字符串分词完毕
            word = None
            # 在待分词字符串中找出一个最大的长度进行尝试
            for size in range(self.maximum, 0, -1):
                if index - size < 0:
                    continue
                # 按当前字符串的最大可行长度尝试划分
                piece = text[index-size: index]
                # 如果划分出来的部分在词典中，那么本轮尝试结束，跳到下一轮
                # 因为要重新根据新索引找合适的size，所以需要break这个for循环
                if piece in self.dictionary:
                    word = piece
                    result.append(word)
                    # 更新下一轮的末尾索引
                    index -= size
                    break
            # 如果在该index下分词都失败，那就只能跳过这个index，往前挪一步
            if word is None:
                result.append(text[index])
                index -= 1
        # 如果index <= 0，说明只剩一个或者不剩未分词的字符串，分词结束
        return result[::-1]  # 双冒号表示步长，负数表示反向


class MM:
    def __init__(self, dic_path):
        self.dictionary = set()
        self.maximum = 0
        with open(dic_path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if line is None:
                    continue
                self.dictionary.add(line)
                self.maximum = max(self.maximum, len(line))

    def cut(self, text):
        result = []
        index = 0
        length = len(text)

        while index < length - 1:
            # 每次分词前需要设置一个flag，如果本轮分词失败就跳过一个索引
            word = None
            for size in range(self.maximum, 0, -1):
                if index + size > length:
                    continue
                piece = text[index: index+size]
                if piece in self.dictionary:
                    word = piece
                    result.append(word)
                    index += size
                    break
            if word is None:
                result.append(text[index])
                index += 1

        return result


class BiMM:
    def __init__(self, dic_path):
        self.mm = MM(dic_path)
        self.imm = IMM(dic_path)

    def cut(self, text):
        mm_result = self.mm.cut(text)
        imm_result = self.imm.cut(text)
        if len(mm_result) < len(imm_result):
            return mm_result
        else:
            return imm_result


class HMM:
    def __init__(self):
        self.model_file = './data/hmm_model.pkl'  # 保存训练结果的路径
        self.state_list = ['B', 'M', 'E', 'S']  # 隐藏状态列表
        self.load_para = False  # 是否需要重新加载model_file

        self.A_dic = {}  # 状态转移概率(隐藏状态->隐藏状态)
        self.B_dic = {}  # 发射概率(隐藏状态->词语/观察状态)
        self.Pi_dic = {}  # 隐藏状态的初始概率

    def try_load_model(self, trained):
        if trained:
            # 二进制模式的文件IO不指定编码方式
            with open(self.model_file, 'rb') as f:
                self.A_dic = pickle.load(f)  # 若之前有dump多次，可以load多次
                self.B_dic = pickle.load(f)  # load操作不影响f
                self.Pi_dic = pickle.load(f)
                self.load_para = True

    def train(self, path):
        # self.try_load_model(False)  # 重置概率矩阵，因为要重新训练
        count_dic = {}  # 统计状态出现的次数，用于求P(o)

        def init_parameter():
            for state in self.state_list:
                # 从状态state出发，到各个状态s的转移概率
                self.A_dic[state] = {s: 0.0 for s in self.state_list}
                # 从状态state发射到输出符号的概率，由于输出的词未定，所以空字典
                self.B_dic[state] = {}
                # 初始概率向量。字典中的元素为数，而不是字典。与上面两个矩阵不同
                self.Pi_dic[state] = 0.0

                count_dic[state] = 0

        def make_label(word):
            label = []
            if len(word) == 1:
                label.append('S')
            else:
                label += ['B'] + ['M'] * (len(word)-2) + ['E']
            return label

        init_parameter()  # 初始化A,B,Pi以及count
        line_num = 0  # 用于计算概率时作为分母，书中为0是错误的
        last_label = None
        with open(path, encoding='utf8') as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue

                # 把每一个字加入到字的集合中，在计算发射概率时有用
                word_list = [i for i in line if i != ' ']
                # 词典中的词的列表，是待标记的观察序列
                line_list = line.split()

                # 记录每一行标记结果(隐藏状态序列)
                line_state = []
                for w in line_list:
                    line_state.extend(make_label(w))

                # 确保所有字都做上标记
                assert len(line_state) == len(word_list)

                # 更新每一行的最后一个标记，为之后的计算转移概率做准备
                last_label = line_state[-1]

                for k, v in enumerate(line_state):
                    if k < (len(line_state) - 1):
                        count_dic[v] += 1.0
                    # 记录每一个隐藏状态的出现次数
                    # 由于初始化时不知道有什么词，所以发射概率矩阵并未对
                    # 每一个词所在项初始化为0，因此不能简单+=，需要手动+1
                    self.B_dic[v][word_list[k]] =\
                        self.B_dic[v].get(word_list[k], 0) + 1.0
                    if k == 0:  # 初始状态
                        self.Pi_dic[v] += 1
                    else:
                        # 由于初始化时有把转移概率矩阵的每一项初始化为0，所以+=
                        self.A_dic[line_state[k-1]][v] += 1.0

        # 根据频度计算概率
        self.Pi_dic = {k: v*1.0 / line_num for k, v in self.Pi_dic.items()}

        # 用于计算发射概率。记录每个隐藏状态发射到输出符号(词)的总数
        sum_of_B_dic = {}
        for k, v in self.B_dic.items():
            sum_of_B_dic[k] = sum(v1 for k1, v1 in v.items())

        temp = {k: {k1: v1 / count_dic[k] for k1, v1 in v.items()}
                for k, v in self.A_dic.items() if k != last_label}
        for k, v in temp.items():
            self.A_dic[k] = v
        # 对于隐藏结尾状态，应该将分母减一，否则概率和不为1
        self.A_dic[last_label] = {k: v / (count_dic[last_label] - 1.0)
                                  for k, v in self.A_dic[last_label].items()
                                  if len(self.A_dic[last_label].items()) > 0}

        # 拉普拉斯平滑，分子加一，分母加 分子个数*1
        # 书中分母不对，应该是每个隐藏状态有自己的分母，分母是由该状态发射出去的
        # 总数
        self.B_dic = {k: {k1: (v1 + 1) / (sum_of_B_dic[k] + len(v.items()))
                          for k1, v1 in v.items()}
                      for k, v in self.B_dic.items()}

        with open(self.model_file, 'wb') as f:
            pickle.dump(self.A_dic, f)
            pickle.dump(self.B_dic, f)
            pickle.dump(self.Pi_dic, f)

        return self

    def viterbi(self, text, start_p, trans_p, emit_p):
        """
        维特比算法，用于根据观测序列推测最有可能的隐藏状态序列。
        这里概率最大的路径就是最优路径
        :param text: 观测序列
        :param start_p: 初始概率向量
        :param trans_p: 状态转移概率矩阵
        :param emit_p: 发射概率矩阵
        :return: 对应观测序列的最有可能的隐藏状态序列
        """
        # 列表由字典组成，每个字典保存从起点到该时刻的所有状态的最大概率
        V = [{}]
        path = {}  # 从起点到时刻t各个状态的最优路径
        # 初始化
        for y in self.state_list:
            V[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]

        for t in range(1, len(text)):
            V.append({})  # 用于保存当前时刻各个状态的概率
            new_path = {}  # 用于保存从起点到当前时刻各个状态的最优路径

            # 检验发射概率矩阵中是否有这个字
            never_seen = (text[t] not in self.B_dic['S'].keys()) and\
                         (text[t] not in self.B_dic['B'].keys()) and\
                         (text[t] not in self.B_dic['M'].keys()) and\
                         (text[t] not in self.B_dic['E'].keys())

            for y in self.state_list:
                # 计算状态y的发射概率
                emit_P = emit_p[y].get(text[t], 0) if not never_seen else 1.0
                # 遍历前一时刻的所有状态。从前一时刻所有可达状态(概率大于0)出发
                # 到当前状态y的最优路径(概率最大的路径)
                # prob是从起点到状态y的最大概率，state保存的是从起点到状态y的最
                # 优路径下状态y的前一时刻状态。
                # 也就是说，对于当前时刻的状态y，从起点到y的最优路径概率为prob，
                # 经过前一时刻的状态state
                prob, state = max([(V[t-1][y0] * trans_p[y0].get(y, 0)
                                    * emit_P, y0)
                                   for y0 in self.state_list if V[t-1][y0] > 0])
                V[t][y] = prob  # 从起点到t时刻状态y的最大概率为prob
                # t时刻状态y的最优路径是从前一时刻状态state的路径出发到当前时刻
                # 状态y
                new_path[y] = path[state] + [y]

            path = new_path  # 结束循环时path为从起点到终点时刻各个状态的最优路径

        # 找出最优路径：对比终点时刻的各个状态对应的最优路径的概率，概率最大的
        # 状态就是终点应该到达的状态，对应的路径就是全局最优路径
        # prob, state = max([V[len(text) - 1][y], y] for y in self.state_list)
        # 如果最后一个字由M发射的概率大于由S发射的，那么从E和M中找结束状态。
        # 这里我没搞懂作者为什么要这样。
        if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
            prob, state = max([V[len(text) - 1][y], y] for y in ('E', 'M'))
        else:
            prob, state = max([V[len(text) - 1][y], y] for y in self.state_list)

        return prob, path[state]

    def cut(self, text, dic_path):
        self.try_load_model(os.path.exists(self.model_file))
        if not self.load_para:  # 没load成功就重新训练
            print("Model file not found, now training...")
            self.train(dic_path)

        prob, pos_list = self.viterbi(text, self.Pi_dic, self.A_dic, self.B_dic)
        begin, next_pos = 0, 0
        for i in range(len(text)):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin: i+1]
                next_pos = i + 1
            elif pos == 'S':
                yield text[i]
                next_pos = i + 1
        if next_pos < len(text):
            yield text[next_pos:]


class JB:
    def __init__(self):
        pass

    @classmethod
    def get_content(cls, path):
        """
        返回路径下的内容
        :param self:
        :param path:
        :return: 字符串
        """
        with open(path, 'r', encoding='gbk', errors='ignore') as f:
            content = ''
            for line in f:
                line = line.strip()
                content += line
            return content

    @classmethod
    def get_TF(cls, words, topK=10):
        tf_dic = {}
        for w in words:
            tf_dic[w] = tf_dic.get(w, 0) + 1
        return sorted(tf_dic.items(), key=lambda x: x[1], reverse=True)[:topK]

    @classmethod
    def stop_words(cls, path):
        with open(path, encoding='utf8') as f:
            return [line.strip() for line in f]


def main():
    dic_dir = r'D:\codes\python\learning-nlp\chapter-3\data\\'
    # text = "南京市长江大桥"
    # text = "这是一个非常棒的方案！"
    # text = "中国卫生部官员24日说，截至2005年底！"

    # tokenizer = IMM(dic_path)
    # tokenizer = MM(dic_path)
    # tokenizer = BiMM(dic_path)
    # tokenizer = HMM()

    # dic_path = dic_dir + r'trainCorpus.txt_utf8'
    # tokenizer.train(dic_path)
    # result = tokenizer.cut(text, dic_path)
    # print(text)
    # print(str(list(result)))
    # print(tokenizer.cut(text))

    # seg_list = jieba.cut(text)
    # print("default: ", '/'.join(seg_list))
    # seg_list = jieba.cut(text, cut_all=True)
    # print("cut_all: ", '/'.join(seg_list))
    # seg_list = jieba.cut(text, cut_all=False)
    # print("precise: ", '/'.join(seg_list))
    # seg_list = jieba.cut_for_search(text)
    # print("cut for search: ", '/'.join(seg_list))

    files = glob.glob(dic_dir + r'news\C000013\*.txt')
    stop_words_path = dic_dir + r'stop_words.utf8'
    corpus = [JB.get_content(x) for x in files]

    sample_inx = random.randint(0, len(corpus))
    split_words = [x for x in jieba.cut(corpus[sample_inx])
                   if x not in JB.stop_words(stop_words_path)]
    print("Sample 1: ", corpus[sample_inx])
    print("\nAfter Cutting: ", '/'.join(split_words))
    print("\nTopK(10) words: ", str(JB.get_TF(split_words)))


main()
