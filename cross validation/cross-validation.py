import pandas as pd

df = pd.read_csv('datasets_4458_8204_winequality-red.csv')

# mapping quality from 0-5

quality_mapping = {
        3:0,
        4:1,
        5:2,
        6:3,
        7:4,
        8:5
        }

df['quality'] = df.quality.map(quality_mapping)

# shuffle the data
df = df.sample(frac = 1).reset_index(drop = True)

# splitting train and test
df_train = df.head(1000)
df_test = df.tail(599)


# decision tree
from sklearn import tree
from sklearn import metrics

clf = tree.DecisionTreeClassifier(max_depth = 7)

cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']


clf.fit(df_train[cols], df_train.quality)

# prediction on training and test

train_pred = clf.predict(df_train[cols])
test_pred = clf.predict(df_test[cols])

# accuracy on train and test

train_acc = metrics.accuracy_score(df_train.quality, train_pred)
test_acc = metrics.accuracy_score(df_test.quality, test_pred)

print('train acc: ',train_acc)
print('test acc: ', test_acc)

# plotting depth vs accuracy to see overfitting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# global label text size
matplotlib.rc('xtick',labelsize = 20)
matplotlib.rc('ytick',labelsize = 20)
train_accuracy = [0.5]
test_accuracy = [0.5]

for depth in range(1,25):
    clf = tree.DecisionTreeClassifier(max_depth = depth)
    
    clf.fit(df_train[cols],df_train.quality)
    
    train_pred = clf.predict(df_train[cols])
    test_pred = clf.predict(df_test[cols])

    # accuracy on train and test

    train_acc = metrics.accuracy_score(df_train.quality, train_pred)
    test_acc = metrics.accuracy_score(df_test.quality, test_pred)   
    
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)
    

# plots
    
plt.figure(figsize=(10,5))
sns.set_style('whitegrid')
plt.plot(train_accuracy,label = 'train')
plt.plot(test_accuracy,label = 'test')

plt.legend(loc = 'best',prop = {'size':15})
plt.xticks(range(0,26,5))
plt.xlabel('max-depth',size  =20)
plt.ylabel('accuracy',size = 20)



















