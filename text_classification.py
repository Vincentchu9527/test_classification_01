# -*- coding:utf-8 -*-

import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. 加载停用词表
stop_words_path = './data/stop/stopword.txt'
with open(stop_words_path, 'r', encoding='utf-8') as sw_msg:
    stop_words = [line.strip() for line in sw_msg.readlines()]
    stop_words[0] = ","  # 修改'\ufeff,' 为 ‘,’
# print(stop_words)
# print(len(stop_words))

# 2. 加载数据并进行分词
# 2.1 训练集
train_contents = []
data_list1 = ['体育', '女性', '文学', '校园']       # 文件夹(标签)名字
for j in data_list1:
    errortimes = 0      # 初始化读取文档错误次数
    path_lst = [name for name in os.listdir('./data/train/' + j)
                if os.path.isfile(os.path.join('./data/train/' + j, name))]     # 获取对应文件名列表
    for path in path_lst:
        path = './data/train/' + j + '/' + path       # 补全为相对路径
        try:                # 使用jieba工具包进行分词
            with open(path, 'r') as train_msg0:
                train_msg1 = train_msg0.read()
                train_cut_msg0 = jieba.cut(train_msg1)
                train_cut_msg1 = " ".join(train_cut_msg0)
            train_contents.append(train_cut_msg1)
        except Exception:   # 读取文件错误时, 打印读取错误的文档名
            errortimes += 1
            print(path, ' 文件读取错误')
    print(j, "训练集中,文件读取错误数:", errortimes)
print("训练集长度:", len(train_contents))

# 2.2 测试集
test_contents = []
data_list2 = ['体育', '女性', '文学', '校园']
for j in data_list2:
    errortimes = 0
    path_lst = [name for name in os.listdir('./data/test/' + j)
                if os.path.isfile(os.path.join('./data/test/' + j, name))]
    for path in path_lst:
        path = './data/test/' + j + '/' + path
        try:
            with open(path, 'r') as test_msg0:
                test_msg1 = test_msg0.read()
                test_cut_msg0 = jieba.cut(test_msg1)
                test_cut_msg1 = " ".join(test_cut_msg0)
            test_contents.append(test_cut_msg1)
        except Exception:
            errortimes += 1
            print(path, '.txt 文件读取错误')
    print(j, "测试集中,文件读取错误数:", errortimes)
print("测试集长度:", len(test_contents))

# 3. 计算单词权重
tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
train_features = tf.fit_transform(train_contents)
print('训练集特征shape:', train_features.get_shape())
# 使用 TfidfVectorizer初始化向量空间模型  使用训练集词袋向量
tf_test = TfidfVectorizer(stop_words=stop_words, sublinear_tf=True, max_df=0.5, vocabulary=tf.vocabulary_)
test_features = tf_test.fit_transform(test_contents)
print('测试集特征shape:', test_features.get_shape())

# 4 生成分类器
# 构造与features对应的labels列表, 其中数字为测试集对应分类文件夹中文件个数
# 可使用len(#文件名列表#)获得, 在此为简化代码手动进入文件夹读取
train_labels = ['体育'] * 1337 + ['女性'] * 954 + ['文学'] * 766 + ['校园'] * 249
test_labels = ['体育'] * 115 + ['女性'] * 38 + ['文学'] * 31 + ['校园'] * 16
model = MultinomialNB(alpha=0.01)
model.fit(train_features, train_labels)
predict_labels = model.predict(test_features)
# print(predict_labels)
score = accuracy_score(test_labels, predict_labels)
print('准确率:', score)
