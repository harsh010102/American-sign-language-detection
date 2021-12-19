import pandas as pd
import sklearn
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns            # visualization tool
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time

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

model = KNeighborsClassifier(n_neighbors=165)
start=time.time()
model.fit(x_train,y_train)
stop=time.time()

train_time=stop-start
print(train_time)
y_pred = model.predict(x_valid)
print("Accuracy:",metrics.accuracy_score(y_valid, y_pred))

img=imread('a.png')
plt.imshow(img)
plt.show()

img_resize=resize(img,(28,28,1))
l=[img_resize.flatten()]

start=time.time()
probability=model.predict_proba(l)
probability=model.predict_proba(l)
probability=model.predict_proba(l)
probability=model.predict_proba(l)
probability=model.predict_proba(l)
probability=model.predict_proba(l)
probability=model.predict_proba(l)
probability=model.predict_proba(l)
probability=model.predict_proba(l)
probability=model.predict_proba(l)


stop = time.time()
print((stop-start)/10, "seconds Classification time")





'''f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(confusion_matrix(y_valid,y_pred), annot=True, linewidths=.1, fmt= '.0f',ax=ax)     # generating a heat map
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()'''



