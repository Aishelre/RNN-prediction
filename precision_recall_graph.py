from matplotlib import pyplot as plt
import numpy as np

def show_value(bar):
    for rect in bar:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.0 * h, '%.2f' % round(float(h), 2),
                ha='center', va='bottom')


path = "./CKPT/trained.csv"
path = "./CKPT/000660.csv"

with open(path, 'rt') as fp:
    pre = fp.readline().strip().split(",")
    rec = fp.readline().strip().split(",")

    print(pre)
    print(rec)

ax = plt.subplot(111)
code1 = ["000660","001020","002720","004540","010145","039740","049080"]
code2 = ["049180","060370","060560","068270","091990","092220","182690","192410","197210","20365","217270"]
code3 = ["9.26", "9.27", "9.28", "9.29"]

n = np.arange(len(code3))
bar1 = ax.bar(n-0.15, pre, width=0.3, color='b', align='center')
bar2 = ax.bar(n+0.15, rec, width=0.3, color='r', align='center')
ax.set_xticks(n)
ax.set_xticklabels(code3)
ax.set_ylabel("Precision & Recall")
ax.set_ylim(0, 1)
ax.legend((bar1[0], bar2[0]), ('Precision', 'Recall'))
ax.set_title("51 P CheckPoint")


show_value(bar1)
show_value(bar2)

plt.show()