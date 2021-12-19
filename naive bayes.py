import pandas as pd
import sklearn
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt


Categories=['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']

train_df = pd.read_csv("data/sign_mnist_train.csv")
valid_df = pd.read_csv("data/sign_mnist_test.csv")

y_train = train_df['label']
y_valid = valid_df['label']
del train_df['label']
del valid_df['label']

x_train = train_df.values
x_valid = valid_df.values

x_train=x_train/255
x_valid=x_valid/255

gnb = GaussianNB().fit(x_train, y_train)
gnb_predictions = gnb.predict(x_valid)
accuracy = gnb.score(x_valid, y_valid)

print(accuracy)

cm = confusion_matrix(y_valid, gnb_predictions)


f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(confusion_matrix(y_valid,gnb_predictions), annot=True, linewidths=.1, fmt= '.0f',ax=ax)     # generating a heat map
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()






    
