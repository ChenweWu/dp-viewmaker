import pandas as pd
import numpy as np

# 使用pandas的read_csv函数读取CSV文件
df = pd.read_csv('classlabel.csv')
labels=[]
for label in df.values:
    labels.append(int(label[1]))
labels=np.array(labels)
#ratio0=sum(labels==0)/len(labels)
ratio1=sum(labels==1)/len(labels)
ratio2=sum(labels==2)/len(labels)
print(ratio1,ratio2)