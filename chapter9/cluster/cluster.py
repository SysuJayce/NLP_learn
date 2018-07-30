# -*- coding: utf-8 -*-
# @Time         : 2018-07-29 20:54
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : cluster.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce

import random
import pandas as pd
import numpy as np

from collections import Counter

from scipy.cluster.hierarchy import ward, dendrogram

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation  # AP聚类算法

from chapter9.cluster.normalization import normalize_corpus


def build_feature_matrix(documents, feature_type='frequency',
                         ngram_range=(1, 1), min_df=0.0, max_df=1.0):
    """
    提取document的特征
    :param documents: 待提取特征的文档集合
    :param feature_type: 需要提取的特征类型
    :param ngram_range: 生成词典时的词的范围，使用a个，a+1个，...a+b个
    :param min_df: 词频下界，低于就不计算
    :param max_df: 词频上界，高于就不计算
    :return: 向量化器，特征矩阵
    """
    feature_type = feature_type.lower().strip()
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, max_df=max_df,
                                     ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, max_df=max_df, min_df=min_df,
                                     ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise Exception("Wrong feature type entered. Possible values:"
                        "'binary', 'frequency', 'tfidf'")

    # 特征矩阵每一行为一个文档中各个词的tfidf值，这里的特征矩阵大小为2822x13670
    feature_matrix = vectorizer.fit_transform(documents).astype(float)

    return vectorizer, feature_matrix


def k_means(feature_matrix, num_clusters=10):
    """
    使用sklearn中的KMeans算法对特征矩阵进行聚类
    :param feature_matrix: 待聚类的特征矩阵
    :param num_clusters: 将要聚成的类的个数
    :return: km类对象，聚类结果
    """
    km = KMeans(n_clusters=num_clusters, max_iter=10000)
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters


def get_cluster_data(clustering_obj, book_data, feature_names,
                     num_clusters, top_n_features=10):
    """
    从Kmeans类对象中提取聚类的详细结果：类别、前n个关键特征、包含的书的书名
    :param clustering_obj: kmeans类对象
    :param book_data:
    :param feature_names:
    :param num_clusters:
    :param top_n_features:
    :return:
    """
    cluster_details = {}
    ordered_center_ids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        # 该类别中的top n个特征，列表
        key_features = [feature_names[index] for index in
                        ordered_center_ids[cluster_num, :top_n_features]]
        cluster_details[cluster_num]['key_features'] = key_features

        # 属于cluster_num类的书的书名列表
        books = book_data[book_data['Cluster'] == cluster_num]['title']\
            .values.tolist()
        cluster_details[cluster_num]['books'] = books

    return cluster_details


def print_cluster_data(cluster_data):
    """
    打印聚类结果
    :param cluster_data: 聚类结果(一个字典)
    key为类别(数字)，value为该类下的样本的具体内容：所属类别、特征、所含书名
    :return:
    """
    for cluster_num, cluster_details in cluster_data.items():
        print('\nCluster {} details:'.format(cluster_num))
        print('-' * 20)
        print('Key features:', cluster_details['key_features'])
        print('book in this cluster:')
        print(', '.join(cluster_details['books']))
        print('=' * 40)


def plot_clusters(num_clusters, feature_matrix, cluster_data,
                  book_data, plot_size=(16, 8)):
    """
    绘制聚类图
    :param num_clusters:
    :param feature_matrix:
    :param cluster_data:
    :param book_data:
    :param plot_size:
    :return:
    """
    def generate_random_color():
        color = '#%06x' % random.randint(0, 0xFFFFFF)
        return color

    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p',
               '*', 'h', 'H', 'D', 'd']
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=1)

    plot_positions = mds.fit_transform(cosine_distance)
    x_pos, y_pos = plot_positions[:, 0], plot_positions[:, 1]

    cluster_color_map = {}
    cluster_name_map = {}

    for cluster_num, cluster_details in cluster_data.items():
        cluster_color_map[cluster_num] = generate_random_color()
        cluster_name_map[cluster_num] = ', '.join(
            cluster_details['key_features'][: 5]).strip()

    cluster_plot_frame = pd.DataFrame(
        {'x': x_pos, 'y': y_pos,
         'label': book_data['Cluster'].values.tolist(),
         'title': book_data['title'].values.tolist()})
    grouped_plot_frame = cluster_plot_frame.groupby('label')

    fig, ax = plt.subplots(figsize=plot_size)
    ax.margins(0.05)

    for cluster_num, cluster_frame in grouped_plot_frame:
        marker = markers[cluster_num] if cluster_num < len(markers)\
            else np.random.choice(markers, size=1)[0]
        ax.plot(cluster_frame['x'], cluster_frame['y'],
                marker=marker, linestyle='', ms=12,
                label=cluster_name_map[cluster_num],
                color=cluster_color_map[cluster_num], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off',
                       labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off',
                       labelleft='off')

    fontP = FontProperties()
    fontP.set_size('small')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=True,
              shadow=True, ncol=5, numpoints=1, prop=fontP)

    for index in range(len(cluster_plot_frame)):
        ax.text(cluster_plot_frame.ix[index]['x'],
                cluster_plot_frame.ix[index]['y'],
                cluster_plot_frame.ix[index]['title'], size=8)
    plt.show()


def affinity_propagation(feature_matrix):
    sim = feature_matrix * feature_matrix.T
    sim = sim.todense()
    ap = AffinityPropagation()
    ap.fit(sim)
    clusters = ap.labels_
    return ap, clusters


def ward_hierarchical_clustering(feature_matrix):
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    linkage_matrix = ward(cosine_distance)
    return linkage_matrix


def plot_hierarchical_clusters(linkage_matrix, book_data, figure_size=(8, 12)):
    fig, ax = plt.subplots(figsize=figure_size)
    book_titles = book_data['title'].values.tolist()

    ax = dendrogram(linkage_matrix, orientation='left', labels=book_titles)
    plt.tick_params(axis='x', which='both', bottom='off', top='off',
                    labelbottom='off')
    plt.tight_layout()
    plt.savefig('./data/ward_hierachical_clusters.png', dpi=200)


def AP(feature_matrix, feature_names, book_data, num_clusters):
    """
    使用AP算法进行聚类
    :param feature_matrix:
    :param feature_names:
    :param book_data:
    :param num_clusters:
    :return:
    """
    ap_obj, clusters = affinity_propagation(feature_matrix)
    book_data['Cluster'] = clusters

    c = Counter(clusters)
    print(c.items())

    # 这里用AP算法聚类得到了1471个类，原因可能是因为没有对书名去重
    total_clusters = len(c)
    print("Total Clusters:", total_clusters)

    cluster_data = get_cluster_data(ap_obj, book_data, feature_names,
                                    num_clusters, 5)
    print_cluster_data(cluster_data)

    # plot_clusters(num_clusters, feature_matrix, cluster_data, book_data)

    linkage_matrix = ward_hierarchical_clustering(feature_matrix)
    plot_hierarchical_clusters(linkage_matrix, book_data, figure_size=(8, 10))


def KM(feature_matrix, feature_names, book_data, num_clusters):
    """
    使用KMeans算法进行聚类
    :param feature_matrix:
    :param feature_names:
    :param book_data:
    :param num_clusters:
    :return:
    """
    # 使用kmeans聚类，clusters是一个列表，保存聚类后特征矩阵中每一行对应的类别
    km_obj, clusters = k_means(feature_matrix, num_clusters)
    book_data['Cluster'] = clusters

    c = Counter(clusters)
    print(c.items())

    # 获取聚类后的具体结果并打印
    cluster_data = get_cluster_data(km_obj, book_data, feature_names,
                                    num_clusters, 5)
    print_cluster_data(cluster_data)

    # plot_clusters(num_clusters, feature_matrix, cluster_data, book_data)


def main():
    """
    使用KMeans和AP算法进行聚类后发现，如果没有对数据做去重处理的话，
    使用AP算法会得到非常多类，这时可能还是KMeans比较合适
    :return:
    """
    book_data = pd.read_csv('./data/data.csv')  # 读取数据集
    print(book_data.head())

    # 获取书名和书的内容
    book_titles = book_data['title'].tolist()
    book_content = book_data['content'].tolist()
    print("Title:", book_titles[0])
    print("Content:", book_content[0])

    # 将书的内容规范化，去除标点、空格，一本书的内容为一句不含标点空格的句子
    norm_book_content = normalize_corpus(book_content)

    # 从规范化后的书本内容中提取tfidf特征矩阵
    vectorizer, feature_matrix = build_feature_matrix(norm_book_content,
                                                      feature_type='tfidf',
                                                      min_df=0.2, max_df=0.9,
                                                      ngram_range=(1, 2))
    print(feature_matrix.shape)

    feature_names = vectorizer.get_feature_names()  # 书本内容的每个特征的名字
    print(feature_names[:10])

    num_clusters = 10  # 想要聚成多少个类

    KM(feature_matrix, feature_names, book_data, num_clusters)  # KMeans算法
    AP(feature_matrix, feature_names, book_data, num_clusters)  # AP算法


if __name__ == '__main__':
    main()
