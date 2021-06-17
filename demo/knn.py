import math


def sim(feature, di1, di2):
    b = 0.0
    c1 = 0.0
    c2 = 0.0
    for index, term in feature:  # feature保存所有特征字典
        x = di1.get(index)  # 获取对应特征的值，可能没有，如果没有x=0
        if not x: x = 0
        y = di2.get(index)
        if not y: y = 0
        b += x * y
        c1 += math.pow(x, 2)
        c2 += math.pow(y, 2)
    sim = b / math.sqrt(c1 * c2)  # cos值
    return sim


def classify(k, feature, trainli, di, cls):
    '''
    k：k的值
    feature：特征字典
    trainli:样本的列表
    di:分类文档的特征列表
    cls:分类文档的类别
    '''
    li = list()
    i = 0
    for cls, traindi in trainli:
        s = sim(feature, traindi, di)
        li.append((s, cls))
    li.sort(reverse=True)  # 排序
    tmpli = li[0:k]  # 取前k个
    print
    tmpli
    di = dict()
    for l in tmpli:
        s, cls = l
        times = di.get(cls)
        if not times:
            times = 0
        di[cls] = times + 1  # 统计属于类别的文档数
    sortli = sorted(di.iteritems(), None, lambda d: d[1], True)  # 排序取文档数最多的类
    print(cls, sortli)
    tocls = sortli[0][0]  # 分出来的类别
    return cls == tocls
