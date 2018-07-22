# -*- coding: utf-8 -*-
# @Time         : 2018-07-21 23:30
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : datetime_recognition.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import re
import jieba.posseg as psg
from _datetime import datetime, timedelta
import locale

locale.setlocale(locale.LC_CTYPE, 'chinese')

UTIL_CN_NUM = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
               '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
               '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
               '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

UTIL_CN_UNIT = {'十': 10, '百': 100, '千': 1000, '万': 10000}


def cn2dig(src):
    """
    从中文中提取数字
    :param src:
    :return:
    """
    if src == "":
        return None
    # 匹配src中的数字，从字符串的起点开始匹配，遇到非数字就结束。
    # 也就是说只有当string是由数字开始的才可能匹配成功
    # qqq123：m为None
    # 123qqq444：匹配成功，m.group()或m.group(0)为123，其中444被忽略，
    # 因为遇到q就停止了匹配
    m = re.match('\d+', src)
    if m:
        return int(m.group(0))
    rsl = 0
    unit = 1  # 单位：个十百千万对应的数字1, 10, 100, 1000...
    # python切片中[start:end:step]，其中step默认为1，这里忽略start和end意思是全
    # 选中，step为-1代表反向
    # 查找输入的src中与数字有关的内容
    for item in src[::-1]:
        if item in UTIL_CN_UNIT.keys():
            unit = UTIL_CN_UNIT[item]
        elif item in UTIL_CN_NUM.keys():
            num = UTIL_CN_NUM[item]
            rsl += num * unit  # 由计数单位乘数字得到数值
        else:
            return None

    # 处理单位前面没有数字的情况，例如十来天，当成10天
    if rsl < unit:
        rsl += unit
    return rsl


def year2dig(year):
    res = ''  # 由于年份有4位，所以用字符串
    # 对于每一个输入的年份字符串，如果是数字那就保持，如果是汉字就改成对应的数字
    for item in year:
        if item in UTIL_CN_NUM.keys():
            res += str(UTIL_CN_NUM[item])
        else:
            res += item
    m = re.match('\d+', res)
    if m:
        # 如果是遇到了简写，例如02年，那么应该添加2000年变成2002年
        if len(m.group(0)) == 2:
            # 处理19xx年的简写
            if int(m.group(0)) > (datetime.today().year -
                                  int(datetime.today().year/100)*100):
                return int(datetime.today().year/100-1)*100 + int(m.group(0))
            # 处理2000年以后的简写
            return int(datetime.today().year/100)*100 + int(m.group(0))
        else:
            return int(m.group(0))
    else:
        return None


def parse_datetime(msg):
    """
    如果输入的msg符合日期规范就使用dateutil.parser的parse函数进行自动解析
    否则手动解析
    :param msg:待解析日期的字符串。
    在经过其他函数的处理之后，msg的格式不包括以下：
    1. 小于等于6位的纯数字:20180828 / 201822
    2. '日'或'号'之后以纯数字结尾：2018年8月18日123123
    :return: %Y-%m-%d %H:%M:%S   2018-8-18 18:28:55 形式的日期
    """
    if msg is None or len(msg) == 0:
        return None
    # 不直接使用dateutil.parser.parse的原因是如果msg是2018年08月18日下午三点
    # 这样的话就会被parse解析为2018-08-18 00:00:00，丢失了后面的时间
    # 这个正则表达式的match表示最多匹配到秒，之后出现的字符都被忽略
    m = re.match(r"([0-9零一二两三四五六七八九十]+年)?"
                 r"([0-9一二两三四五六七八九十]+月)?"
                 r"([0-9一二两三四五六七八九十]+[号日])?"
                 r"([上中下午晚早]+)?"
                 r"([0-9零一二两三四五六七八九十百]+[点:.时])?"
                 r"([0-9零一二三四五六七八九十百]+分?)?"
                 r"([0-9零一二三四五六七八九十百]+秒)?", msg)
    # 仅仅判断m是否为None【也就是判断是否成功匹配】是不行的
    # 因为上述的正则表达式全是'?'，也就是出现0次或1次
    # 即使msg不含日期，也会匹配成功，m也不会是none
    # 因此需要加一个条件：在匹配成功之后判断m.group(0)是否为空
    if m is not None and m.group(0) != '':
        # m.group(0)是匹配成功的整个字符串
        # m.group(x)是匹配成功的第x项，对应正则表达式的第x项
        res = {
            'year': m.group(1),
            'month': m.group(2),
            'day': m.group(3),
            'hour': m.group(5) if m.group(5) is not None else '00',
            'minute': m.group(6) if m.group(6) is not None else '00',
            'second': m.group(7) if m.group(7) is not None else '00',
        }
        params = {}
        for name in res:
            # 由于年月日是必需的，所以这三个key对应的value可能是None
            if res[name] is not None and len(res[name]) != 0:
                # 这里res字典中的value是未处理的日期时间段，如五十分、三月
                tmp = None
                if name == 'year':
                    # 由于value字符串的最后一个字符是时间单位，所以舍去
                    tmp = year2dig(res[name][:-1])
                else:
                    tmp = cn2dig(res[name][:-1])
                if tmp is not None:
                    params[name] = int(tmp)

        # 用上面提取出的时间来替换一个正规日期里的相关时间
        target_date = datetime.today().replace(**params)
        # 处理诸如'下午'、'晚上'之类的时间，转换成24小时制
        is_pm = m.group(4)
        if is_pm is not None:
            if is_pm == u'下午' or is_pm == u'晚上' or is_pm == '中午':
                hour = target_date.time().hour
                if hour < 12:
                    target_date = target_date.replace(hour=hour + 12)
        return target_date.strftime('%Y-%m-%d %H:%M:%S')

    # 如果正则匹配失败则解析失败，返回None
    else:
        return None


def check_time_valid(word):
    """
    1. 剔除小于等于6位的纯数字日期
    2. 将日之后以纯数字结尾的日期中的结尾数字去掉，并将号改成日
    :param word:
    :return:
    """
    m = re.match('\d+$', word)  # 匹配纯数字，如果word是纯数字则匹配成功
    if m and len(word) <= 6:  # 如果word是纯数字，且只有年月，则返回None
        return None
    # xx号123  =====>  xx日
    new_word = re.sub('[号日]\d+$', '日', word)
    
    if new_word != word:   # 如果输入的日期的尾部是'日'加数字，
        return check_time_valid(new_word)
    else:
        return new_word


def time_extract(text):
    time_res = []
    word = ''
    key_date = {'今天': 0, '明天': 1, '后天': 2}
    for k, v in psg.cut(text):
        # 从词性标注的结果中进行时间抽取
        if k in key_date:
            if word != '':  # 如果当前已经获得了一个日期时间段，如'2018年'
                time_res.append(word)  # 那就先保存下来，避免被其他时间段覆盖
            # 将'今明后'这些与时间相关的词转换成日期
            # timedelta可以与datetime直接运算
            word = (datetime.today()+timedelta(
                days=key_date.get(k, 0))).strftime('%Y年%m月%d日')
        elif word != '':
            if v in ['m', 't']:  # 因为在词性标注中m表示数词，t表示时间词
                word += k
            else:  # 遇到时间单位的话先保存已获得的日期时间段，再清空word的值
                time_res.append(word)
                word = ''
        elif v in ['m', 't']:
            word = k

    # 将循环结束后剩余的与时间相关的部分加入到列表中
    if word != '':
        time_res.append(word)

    result = list(filter(lambda x: x is not None,
                         [check_time_valid(w) for w in time_res]))
    final_res = [parse_datetime(w) for w in result]
    return [x for x in final_res if x is not None]


if __name__ == '__main__':
    text1 = '我要住到明天下午三点'
    print(text1, time_extract(text1), sep=': ')

    text2 = '预定28号的房间'
    print(text2, time_extract(text2), sep=': ')

    text3 = '我要从26号下午4点住到11月2号'
    print(text3, time_extract(text3), sep=': ')

    text4 = '我要预订今天到30的房间'
    print(text4, time_extract(text4), sep=': ')

    text5 = '今天30号呵呵'
    print(text5, time_extract(text5), sep=': ')
