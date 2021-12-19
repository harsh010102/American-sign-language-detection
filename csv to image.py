import cv2
import pandas as pd
import numpy as np


test=pd.read_csv("data/sign_mnist_test.csv")
labels=test['label'].values

del test['label']

'''a=test.iloc[1].values
a = np.reshape(a, (28, 28))
a = a.astype(np.uint8)
a=255*a
a=cv2.resize(a,(224,224))
a = cv2.merge((a,a,a))'''

for i in range(0,test.shape[0]):
	a=test.iloc[i].values
	a = np.reshape(a, (28, 28))
	a = a.astype(np.uint8)
	a=255*a
	a=cv2.resize(a,(224,224))
	a = cv2.merge((a,a,a))
	cv2.imwrite('data/test/'+str(labels[i])+'/'+str(i)+".jpg",a)
	print(i)
	print(a)
