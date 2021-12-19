import pandas as pd
import sklearn
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import seaborn as sns
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

accuracy_array=[]
depth=[]
train_time=[]
classify_time=[]
for i in range(1,50):
    print(i)
    DecisionTree_Class_Model = DecisionTreeClassifier(ccp_alpha=0.00, class_weight=None, criterion='entropy', max_depth=i, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort='deprecaed', random_state=None, splitter='best')
    start=time.time()
    DecisionTree_Class_Model.fit(x_train,y_train)
    stop=time.time()
    traintime=stop-start
    train_time.append(traintime)
    #DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort='deprecaed', random_state=None, splitter='')
    y_pred=DecisionTree_Class_Model.predict(x_valid)
    accuracy=metrics.accuracy_score(y_valid,y_pred)
    print('The accuracy of the model is: '+str(accuracy*100))
    accuracy_array.append(accuracy)
    img=imread('y.png')
    img_resize=resize(img,(28,28,1))
    l=[img_resize.flatten()]
    start=time.time()
    probability=DecisionTree_Class_Model.predict_proba(l)

    stop=time.time()
    classifytime=(stop-start)/40
    classify_time.append(classifytime)
    depth.append(i)
    
'''
DecisionTree_Class_Model = DecisionTreeClassifier(ccp_alpha=0.01, class_weight=None, criterion='entropy', max_depth=1000, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort='deprecaed', random_state=None, splitter='best')
DecisionTree_Class_Model.fit(x_train,y_train)
#DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort='deprecaed', random_state=None, splitter='')
y_pred=DecisionTree_Class_Model.predict(x_valid)
accuracy=metrics.accuracy_score(y_valid,y_pred)
print('The accuracy of the model is: '+str(accuracy*100))

img=imread('y.png')
plt.imshow(img)
img_resize=resize(img,(28,28,1))
l=[img_resize.flatten()]
probability=DecisionTree_Class_Model.predict_proba(l)

for ind,val in enumerate(Categories):
    print(f'{val}={probability[0][ind]*100}%')
    print("The predicted image is : "+ Categories[DecisionTree_Class_Model.predict(l)[0]])
plt.show()

f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(confusion_matrix(y_valid,y_pred), annot=True, linewidths=.1, fmt= '.0f',ax=ax)     # generating a heat map
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()
'''
'''
clf = DecisionTreeClassifier()
path = clf.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, impurities)
plt.xlabel("effective alpha")
plt.ylabel("total impurity of leaves")

clfs = []

for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(x_train, y_train)
    clfs.append(clf)

tree_depths = [clf.tree_.max_depth for clf in clfs]
plt.figure(figsize=(10,  6))
plt.plot(ccp_alphas[:-1], tree_depths[:-1])
plt.xlabel("effective alpha")
plt.ylabel("total depth")'''
