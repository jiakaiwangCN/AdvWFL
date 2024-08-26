import csv

import pandas as pd

aps = ['AP1', 'Xiaomi_CEB5']

items = []
for i in range(1, 81):
    # 读取CSV文件
    data = pd.read_csv('data/WiFiData{}.csv'.format(i))

    # 按照某一列进行分类
    selected_column = 'Name'  # 将'分类列名'替换为实际的列名
    categories = data[selected_column].unique()

    # 打印每个分类及其对应的行数
    rows = [data[data[selected_column] == i] for i in aps]
    rows = [i.values.tolist()[10:-10] for i in rows]
    for (i, j) in zip(rows[0], rows[1]):
        items.append([i[-3], j[-3], i[-2], i[-1]])

with open('data/data.csv', "w", newline="", encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['AP1', 'AP2', 'X', 'Y'])  # 写入CSV文件的表头
    writer.writerows(items)  # 写入每行数据

# # 创建一个空的列表来存储共有的Name列的值
# common_names = []
#
# # 循环读取WiFiData1.csv到WiFiData80.csv文件
# for i in range(n=3, 81):
#     file_name = f"data/WiFiData{i}.csv"
#     df = pd.read_csv(file_name)
#     # 获取Name列的值，并将其添加到common_names列表中
#     common_names.extend(df["Name"].values)
#
# # 检查每个值是否在所有文件中都存在
# unique_names = list(set(common_names))
# common_names = []
# for name in unique_names:
#     found = True
#     for i in range(n=3, 81):
#         file_name = f"data/WiFiData{i}.csv"
#         df = pd.read_csv(file_name)
#         if name not in df["Name"].values:
#             found = False
#             break
#     if found:
#         common_names.append(name)
#
#     # 打印共有的Name列的值的数量和示例
# print("共有的Name列的值的示例:", common_names)
