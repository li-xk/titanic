#导入处理数据包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 忽略错误
import warnings

warnings.filterwarnings('ignore')

#导入数据
#训练数据集
train = pd.read_csv("./train.csv")
#测试数据集
test  = pd.read_csv("./test.csv")
print ('训练数据集:',train.shape,'测试数据集:',test.shape)
rowNum_train=train.shape[0]
rowNum_test=test.shape[0]
print('kaggle训练数据集有多少行数据：',rowNum_train,
     ',kaggle测试数据集有多少行数据：',rowNum_test,)
#合并数据集
full = train.append( test , ignore_index = True )#使用append进行纵向堆叠

# print ('合并后的数据集:',full.shape)
# print('处理前：')
print(full.isnull().sum())
#年龄(Age)
full['Age']=full['Age'].fillna( full['Age'].mean() )
#船票价格(Fare)
full['Fare'] = full['Fare'].fillna( full['Fare'].mean() )
# print('处理后：')
# print(full.isnull().sum())
full['Embarked'] = full['Embarked'].fillna( 'S' )
full['Cabin'] = full['Cabin'].fillna( 'U' )
sex_mapDict={'male':1,
            'female':0}
#map函数：对Series每个数据应用自定义的函数计算
full['Sex']=full['Sex'].map(sex_mapDict)
embarkedDf = pd.DataFrame()
embarkedDf = pd.get_dummies( full['Embarked'] , prefix='Embarked' )
full = pd.concat([full,embarkedDf],axis=1)
full.drop('Embarked',axis=1,inplace=True)
pclassDf = pd.DataFrame()

#使用get_dummies进行one-hot编码，列名前缀是Pclass
pclassDf = pd.get_dummies( full['Pclass'] , prefix='Pclass' )
full = pd.concat([full,pclassDf],axis=1)

#删掉客舱等级（Pclass）这一列
full.drop('Pclass',axis=1,inplace=True)
def getTitle(name):
    str1=name.split( ',' )[1] #Mr. Owen Harris
    str2=str1.split( '.' )[0]#Mr
    str3=str2.strip()
    return str3

titleDf = pd.DataFrame()
#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = full['Name'].map(getTitle)
title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }

#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = titleDf['Title'].map(title_mapDict)

titleDf = pd.get_dummies(titleDf['Title'])
#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,titleDf],axis=1)

#删掉姓名这一列
full.drop('Name',axis=1,inplace=True)
#存放客舱号信息
cabinDf = pd.DataFrame()

full[ 'Cabin' ] = full[ 'Cabin' ].map( lambda c : c[0] )#客舱号的首字母代表处于哪个，U代表不知道属于哪个船舱


cabinDf = pd.get_dummies( full['Cabin'] , prefix = 'Cabin' )
#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,cabinDf],axis=1)

#删掉客舱号这一列
full.drop('Cabin',axis=1,inplace=True)
familyDf = pd.DataFrame()

familyDf[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1
familyDf[ 'Family_Single' ] = familyDf[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
familyDf[ 'Family_Small' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
familyDf[ 'Family_Large' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )
full = pd.concat([full,familyDf],axis=1)
full.drop('FamilySize',axis=1,inplace=True)
corrDf = full.corr()
corrDf['Survived'].sort_values(ascending =False)

full_X = pd.concat( [titleDf,#头衔
                     pclassDf,#客舱等级
                     familyDf,#家庭大小
                     full['Fare'],#船票价格
                     full['Sex'],#性别
                     cabinDf,#船舱号
                     embarkedDf,#登船港口
                    ] , axis=1 )

sourceRow=891


#原始数据集：特征
source_X = full_X.loc[0:sourceRow-1,:]
#原始数据集：标签
source_y = full.loc[0:sourceRow-1,'Survived']

#预测数据集：特征
pred_X = full_X.loc[sourceRow:,:]

#建立模型用的训练数据集和测试数据集

size=np.arange(0.6,1,0.1)
scorelist=[[],[],[],[]]
from sklearn.model_selection import train_test_split
for i in range(0,4):
    train_X, test_X, train_y, test_y = train_test_split(source_X ,
                                                        source_y,
                                                      train_size=size[i],
                                                        random_state=5)
    #逻辑回归
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit( train_X , train_y )
    scorelist[0].append(model.score(test_X , test_y ))

    #随机森林Random Forests Model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit( train_X , train_y )
    scorelist[1].append(model.score(test_X , test_y ))

    #支持向量机Support Vector Machines
    from sklearn.svm import SVC
    model = SVC()
    model.fit( train_X , train_y )
    scorelist[2].append(model.score(test_X , test_y ))

    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors = 3)
    model.fit( train_X , train_y )
    scorelist[3].append(model.score(test_X , test_y ))

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
color_list = ('red', 'blue', 'lightgreen', 'cornflowerblue', 'turquoise', 'magenta')
for i in range(0,4):
    plt.plot(size,scorelist[i],color=color_list[i])
plt.legend(['逻辑回归', '随机森林','支持向量机', 'KNN最邻近算法'])

plt.xlabel('训练集占比')
plt.ylabel('准确率')
plt.title('不同的模型随着训练集占比变化曲线')
plt.show()

