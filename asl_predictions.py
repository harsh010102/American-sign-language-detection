import tensorflow.keras as keras
import pandas as pd
from ann_visualizer.visualize import ann_viz;
import time

model = keras.models.load_model('asl_model')
print(model.summary())
import pandas as pd
import sklearn
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns            # visualization tool
from sklearn.metrics import classification_report,confusion_matrix

from tensorflow.keras.preprocessing import image as image_utils
import cv2

cap=cv2.VideoCapture(0)
while(True):
    _,imagee=cap.read()
    image=cv2.cvtColor(imagee,cv2.COLOR_BGR2GRAY)
    image = image_utils.img_to_array(image)
    image = image.reshape(1,28,28,1)
    import numpy as np
    prediction = model.predict(image)
    predicted_letter = dictionary[np.argmax(prediction)]
    print(predicted_letter)
    cv2.imshow(imagee,'image')
    cv2.waitKey(1)

cv2.destroyAllWindows
cap.release()



