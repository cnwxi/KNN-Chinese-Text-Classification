from typing import Dict
import json
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from myknn import KNN
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV


def parse_line_to_sample(perLine: str) -> Dict:
    cols = perLine.split('_!_')
    return {
        'category': cols[2],
        'title': cols[3],
        'keyword': cols[4]
    }


corpus = []
print("读取文件")
print('-----------------------------')

with open('train.txt', 'r', encoding='utf-8') as f:
    file = f.readlines()
    for line in tqdm(file):
        sample = parse_line_to_sample(line)
        corpus.append(sample)

print("数据集大小")
print('-----------------------------')
print(len(corpus))
print("输出前三个样本")
print('-----------------------------')
print(json.dumps(corpus[:3], indent=2, ensure_ascii=False))
print('-----------------------------')
""" 采用的数据包括：标题、关键词"""
# 读取停用词
print("读取停用词")
print('-----------------------------')
stopwords = []
with open('stop_words.utf8', encoding="utf-8") as sf:
    readlines = sf.readlines()
    for line in readlines:
        stopwords.append(line.strip())


# stopwords = [line.strip() for line in open('stop_words.utf8', encoding="utf-8").readlines()]  # 加载停用词


def pre_process_sample(perSample: Dict) -> Dict:
    # constr = ''
    # for uchar in perSample['title']:
    #     if u'\u4e00' <= uchar <= u'\u9fa5':  # 是中文字符
    #         if uchar != ' ':  # 去除空格
    #             constr += uchar
    # for uchar in perSample['keyword']:
    #     if u'\u4e00' <= uchar <= u'\u9fa5':  # 是中文字符
    #         if uchar != ' ':  # 去除空格
    #             constr += uchar
    # perSample['tmp'] = list(jieba.cut(constr))  # 分词
    tmp = list(jieba.cut(perSample['title'] + perSample['keyword']))
    final = []
    # 去除停用词
    for word in tmp:
        if word not in stopwords:
            final.append(word)
    perSample['segmented_title'] = ' '.join(final)
    return perSample


print("数据处理，去除停用词标点符号，并使用jieba进行分词后用空格连接")
print('-----------------------------')
processed_corpus = []
for i in tqdm(corpus):
    processed_corpus.append(pre_process_sample(i))

# processed_corpus = [pre_process_sample(s) for s in corpus]
print("打印前三条数据")
print('-----------------------------')
print(json.dumps(processed_corpus[:3], indent=2, ensure_ascii=False))

x_data = [s['segmented_title'] for s in processed_corpus]
y_data = [s['category'] for s in processed_corpus]
#
"""划分数据集"""
# valid_x, test_x, valid_y, test_y = train_test_split(remain_x, remain_y, test_size=0.5, random_state=42)
#
# print(valid_x[:5])
# print(valid_y[:5])
print('-----------------------------')
print("划分数据集")
print('-----------------------------')
test_x, train_x, test_y, train_y = train_test_split(x_data, y_data, test_size=0.7, random_state=42)

print("训练集大小")
print('-----------------------------')
print(len(train_y))
print('-----------------------------')
print("输出前3条数据")
print('-----------------------------')
for i in range(3):
    print(train_x[i], train_y[i])
print("测试集大小")
print('-----------------------------')
print(len(test_y))
print('-----------------------------')
print("输出前3条数据")
for i in range(3):
    print(test_x[i], test_y[i])
print('-----------------------------')
"""向量化"""

print('向量化')
print('-----------------------------')
tfidf = TfidfVectorizer(norm='l2', ngram_range=(1, 2))
features = tfidf.fit_transform(train_x)
test = tfidf.transform(test_x)
print(features.shape)
print('-----------------------------')
for i in range(3):
    print(features[i])
'''测试'''
# tmp1 = features[0].toarray()[0]
# tmp2 = test[0].toarray()[0]
# print(tmp1)
# print(tmp2)
# dis = np.linalg.norm(tmp1 - tmp2) # np中计算欧式距离
# print(dis)

print('-----------------------------')
print('分类')
print('-----------------------------')

"""多项式贝叶斯"""
# print('多项式贝叶斯')
# print('-----------------------------')
# clf = MultinomialNB()

"""KNN"""
# print('KNN')
# print('-----------------------------')
# clf = KNeighborsClassifier(125)
# param_grid = [{'n_neighbors': list(range(100, 200))}]
# grid_search = GridSearchCV(clf, param_grid, cv=3,
#                            scoring='f1_macro')
# grid_search.fit(features, train_y)
# print("the best params：", grid_search.best_params_)

# print("训练")
# print('-----------------------------')
# clf.fit(features, train_y)
# print("预测")
# print('-----------------------------')
# y_pred = []
# index = 0
# for i in tqdm(test):
#     tmp_pred = clf.predict(i)
#     y_pred.append(tmp_pred)

'''我的KNN'''
print('MYKNN')
print('-----------------------------')
print("预测")
print('-----------------------------')
y_pred = []

for i in tqdm(test):
    pred = KNN(features.toarray(), train_y, i, 100)
    y_pred.append(pred)

"""性能指标"""
print('-----------------------------')
print("accuracy_score:", accuracy_score(test_y, y_pred))
print("precision_score:", precision_score(test_y, y_pred, average='micro'))
print("recall_score:", recall_score(test_y, y_pred, average='micro'))
print("f1_score:", f1_score(test_y, y_pred, average='micro'))
