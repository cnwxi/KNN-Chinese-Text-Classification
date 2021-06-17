path1 = 'train.txt'  # 打开需要处理的txt文件
path2 = 'raw.txt'  # 储存处理后的数据
f = open(path1, 'r', encoding='utf-8')  # 将文本格式编码为utf-8，防止编码错误
fw = open(path2, 'w', encoding='utf-8')
for line in f:  # 逐行处理
    constr = ''  # 记录每行处理后的数据
    for uchar in line:
        if uchar >= u'\u4e00' and uchar <= u'\u9fa5':  # 是中文字符
            if uchar != ' ':  # 去除空格
                constr += uchar
    fw.write(constr + '\n')  # 写入处理后的数据，每行以空格隔开
