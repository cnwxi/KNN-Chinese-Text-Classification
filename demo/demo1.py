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


def pre_process_sample(sample: Dict) -> Dict:
    """
    预处理数据
    :param sample: 原始样本
    :return: 预处理后的样本
    """
    sample['segmented_title'] = list(jieba.cut(sample['title']))
    return sample


processed_corpus = [pre_process_sample(s) for s in corpus]
print(json.dumps(processed_corpus[:3], indent=2, ensure_ascii=False))

from sklearn.model_selection import train_test_split

x_data = [s['segmented_title'] for s in processed_corpus]
y_data = [s['category'] for s in processed_corpus]

remain_x, train_x, remain_y, train_y = train_test_split(x_data, y_data, test_size=0.7, random_state=42)
valid_x, test_x, valid_y, test_y = train_test_split(remain_x, remain_y, test_size=0.5, random_state=42)

print(f"训练数据样本: {len(train_x)}")
print(f"评估数据样本: {len(valid_x)}")
print(f"测试数据样本: {len(test_x)}")

# labels = {"news_story", "news_culture", "news_entertainment",
#           "news_sports", "news_finance", "news_house",
#           "news_car", "news_edu", "news_tech",
#           "news_military", "news_travel", "news_world",
#           "stock", "news_agriculture", "news_game"}
# Base model 直接选择 BiLSTM_Model，经验值
from kashgari.tasks.classification import BiLSTM_Model

# 初始化模型
base_model = BiLSTM_Model()

# 使用训练和评估数据训练模型
# fit 方法将会返回一个 history 对象，里面有记录训练过程每一个 Epoch 的 Loss 和 Accuracy
# 现在存储下来，用于后续的可视化
base_model.fit(train_x,
                              train_y,
                              valid_x,
                              valid_y,
                              batch_size=128,
                              epochs=1)

# 使用测试数据集测试模型
# evaluate 方法输出详细的评估信息，同时以字典形式返回评估信息，存下来用于后续的比较
base_report = base_model.evaluate(test_x, test_y)
print(base_report)
base_report = base_model.predict(test_x)
print(base_report)
