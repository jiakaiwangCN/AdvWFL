import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


normal = [0.683832894, 0.859583703, 0.658465957, 0.991945517, 0.317521765, 0.977441033, 0.701143607, 0.801831953, 1.498487628, 0.272148411, 0.609636268, 0.58279366, 1.060152326, 0.395697206, 1.21007684]
digital = [2.51780206, 2.358755886, 5.337566757, 2.476701291, 2.110204624, 2.370552487, 2.034781388, 2.179045743, 2.46097713, 1.418771509, 2.000158276, 1.931000648, 2.74703355, 1.576560178, 1.722876005]
physical = [1.105460101, 2.867146503, 5.682030601, 1.346039023, 1.014498403, 2.866440714, 1.46766077, 3.400532882, 1.990612554, 1.356746509, 1.311756785, 1.91472539, 2.994238327, 2.700162851, 4.081574489]
ler = []
se = []

for a, b in zip(normal, physical):
    ler.append((b - a) / a)
print(sum(element >= 2 for element in ler) / len(ler))
print(sum(element >= 1.5 for element in ler) / len(ler))
print(sum(element >= 1 for element in ler) / len(ler))
print(min(ler))

for a, b in zip(digital, physical):
    se.append(abs(b - a))
print(sorted(se, reverse=True))
print(sum(element >= 1.3 for element in se))


# positions = np.arange(len(normal))
#
# plt.bar(positions - 0.2, normal, width=0.2, label='Normal', color='#E1C855')
# plt.bar(positions, digital, width=0.2, label='Digital', color='#E07B54')
# plt.bar(positions + 0.2, physical, width=0.2, label='Physical', color='#51B1B7')
#
# # 设置图表标题和标签
# plt.xlabel('Position')
# plt.ylabel('LE(m)')
#
# # 设置位置标签
# plt.xticks(positions, [f'{i+1}' for i in range(len(normal))])
#
# # 显示图例
# plt.legend()
#
# plt.savefig('pic/phy1.pdf')
#
# # 显示图表
# plt.show()

bins = 30
alpha = 0.7
plt.hist(normal, bins=bins, alpha=alpha, label='Normal', color='#E1C855')
plt.hist(digital, bins=bins, alpha=alpha, label='Digital', color='#E07B54')
plt.hist(physical, bins=bins, alpha=alpha, label='Physical', color='#51B1B7')

# 设置图表标题和标签
plt.xlabel('LE(m)')
plt.ylabel('Frequency')

# 添加图例
plt.legend()

plt.savefig('pic/phy2.pdf')

# 显示图表
plt.show()

# sns.kdeplot(normal, color='#E1C855', label='Normal', multiple="stack")
# sns.kdeplot(digital, color='#E07B54', label='Digital', multiple="stack")
# sns.kdeplot(physical, color='#51B1B7', label='Physical', multiple="stack")
#
#
# # 设置图表标题和标签
# plt.xlabel('LE(m)')
# plt.ylabel('Frequency')
#
# # 添加图例
# plt.legend()
#
# plt.savefig('pic/phy3.pdf')
#
# # 显示图表
# plt.show()