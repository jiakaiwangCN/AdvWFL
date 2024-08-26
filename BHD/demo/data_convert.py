import csv

x = [-0.8, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.75]
y = [1.9, 2.8, 3.7, 4.6, 6.4, 7.3, 8.2, 9.1]
for i in range(len(x)):
    for j in range(len(y)):
        # 读取原始文件内容
        with open('data/WiFiData{}.txt'.format(i * len(y) + j + 1), "r", encoding='utf-8') as f:
            content = f.read()

            # 将空格/tab替换为逗号
        content = content.replace("\t", ",")
        content = content.split('\n')

        # 定义CSV文件的表头
        header = ["Time", "Name", "IP", "Value", "X", "Y"]

        # 写入CSV文件
        with open('data/WiFiData{}.csv'.format(i * len(y) + j + 1), "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)  # 写入CSV文件的表头
            writer.writerows(k.split(',') + [x[i], y[j]] for k in content)  # 写入每行数据
