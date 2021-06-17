from typing import Dict
import json
import jieba


def parse_line_to_sample(line: str) -> Dict:
    cols = line.split('_!_')
    return {
        'title': cols[3],
        'category': cols[2]
    }


corpus = []

with open('../final/train.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        sample = parse_line_to_sample(line)
        corpus.append(sample)

# 输出前三个样本
print(json.dumps(corpus[:3], indent=2, ensure_ascii=False))

def stopwordslist():
    stopwords = [line.strip() for line in open('../final/stop_words.utf8', encoding='utf-8').readlines()]
    return stopwords

def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    print("正在分词")
    sentence_depart = jieba.cut(sentence.strip())
    # 创建一个停用词列表
    stopwords = stopwordslist()
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t':
                outstr += word
            outstr += " "
    return outstr


def pre_process_sample(sample: Dict) -> Dict:
    """
    预处理数据
    :param sample: 原始样本
    :return: 预处理后的样本
    """
    print('分词')
    sentence_depart = jieba.cut()
    sample['segmented_title'] = list(jieba.cut(sample['title']))
    return sample


processed_corpus = [pre_process_sample(s) for s in corpus]
print(json.dumps(processed_corpus[:3], indent=2, ensure_ascii=False))

from sklearn.model_selection import train_test_split

x_data = [s['segmented_title'] for s in processed_corpus]
y_data = [s['category'] for s in processed_corpus]

from sklearn.preprocessing import LabelEncoder

# 把标签转数字,规格化处理
le = LabelEncoder()
y_data = le.fit_transform(y_data)

remain_x, train_x, remain_y, train_y = train_test_split(x_data, y_data, test_size=0.7, random_state=42)
valid_x, test_x, valid_y, test_y = train_test_split(remain_x, remain_y, test_size=0.5, random_state=42)
