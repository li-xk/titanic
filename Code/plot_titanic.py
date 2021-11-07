import numpy as np
import pandas as train_data
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
# 忽略错误
import warnings

warnings.filterwarnings('ignore')

# 解决图表中中文显示为方格的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决图表中负号显示为方格的问题
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
train_data = train_data.read_csv("train.csv")

# 数据可视化

# 生还者和未生还者数据可视化
train_data.info()
total_survived_sum = train_data['Survived'].sum()
total_nosurvived_sum =891 - train_data['Survived'].sum()

plt.pie([total_nosurvived_sum,total_survived_sum],labels=['no survived','survived'],autopct='%1.0f%%')
plt.title('Survival rate')
plt.show()


# 性别对生还率的影响
train_data_sex = train_data.Sex.value_counts()
train_data_sex

# 绘制不同性别的生还率
# 使用的是sns.barplot()显示的是某种分类变量分布的平均值
sns.barplot(x='Sex',y='Survived',data=train_data)
plt.title('不同性别的生还率')
plt.show()

# 按照年龄进行均匀分组，按10岁一组进行划分
bins=np.arange(0,90,10)
# 添加年龄区间列
train_data['Age_band']=pd.cut(train_data.Age,bins)
train_data['Age_band']
sns.barplot(x='Age_band',y='Survived',data=train_data)


# 绘制不同舱位的生还率
sns.barplot(x='Pclass', y='Survived', data=train_data)
plt.title('不同舱位的生还率')
plt.xlabel('舱位')
plt.ylabel('生还率')
plt.show()

# 绘制父母因素的生还率
parch_train_data=train_data[train_data['Parch']!=0]
no_parch_train_data=train_data[train_data['Parch']==0]
# 有父母
plt.pie([parch_train_data['Survived'][parch_train_data['Survived'] == 0].count(),parch_train_data['Survived'][parch_train_data['Survived'] == 1].count()],labels=['No Survived', 'Survived'],autopct='%1.0f%%')
plt.title('和父母一起登船的生还率')
plt.show()

#无父母
plt.pie([no_parch_train_data['Survived'][no_parch_train_data['Survived'] == 0].count(),no_parch_train_data['Survived'][no_parch_train_data['Survived'] ==1].count()],labels=['No Survived', 'Survived'],autopct='%1.0f%%')
plt.title('没有和父母一起登船的生还率')
plt.show()

# 绘制港口的生还率
sns.barplot(x='Embarked', y='Survived', data=train_data)
plt.title('不同港口的生还率')
plt.xlabel('登船港口')
plt.ylabel('生还率')
plt.show()


