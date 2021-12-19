import pandas as pd
import sklearn
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns            # visualization tool
from sklearn.metrics import classification_report,confusion_matrix
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

from sklearn.svm import SVC
classifier = SVC(kernel='sigmoid', random_state = 1,probability=True)#kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
print('aaaaaaaaaaaaaaaa')

start = time.time()
classifier.fit(x_train,y_train)
stop = time.time()
print(stop-start, "seconds train time")


y_pred = classifier.predict(x_valid)
valid_df["Predictions"] = y_pred

cm = confusion_matrix(y_valid,y_pred)
accuracy = float(cm.diagonal().sum())/len(y_valid)
print("\nAccuracy Of SVM For The Given Dataset : ", accuracy)
#Accuracy Of SVM For The Given Dataset :  0.8418851087562744

img=imread('f.png')
plt.imshow(img)
plt.show()

img_resize=resize(img,(28,28,1))
l=[img_resize.flatten()]

start=time.time()
probability=classifier.predict_proba(l)
stop = time.time()
print(stop-start, "seconds Classification time")

for ind,val in enumerate(Categories):
    print(f'{val}={probability[0][ind]*100}%')
    print("The predicted image is : "+ Categories[classifier.predict(l)[0]])

f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(confusion_matrix(y_valid,y_pred), annot=True, linewidths=.1, fmt= '.0f',ax=ax)     # generating a heat map
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()

plt.scatter(x_train[:,0], x_train[:,1],s=0.2)
support_vector_indices = classifier.support_
plt.scatter(x_train[:,0], x_train[:,1], s=0.2)

support_vectors = classifier.support_vectors_
plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red',s=0.2)
plt.title('Linearly separable data with support vectors')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()







