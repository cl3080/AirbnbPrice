
# coding: utf-8

# In[ ]:


# use NIMA model to score images
import tensorflow as tf
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

from utils import mean_score, std_score

NIMA_score = []
NIMA_dic = {}
image_name = os.path.join(Photo_path, str(photo_id)+str('.jpg'))

with tf.device('/CPU:0'):
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('weights/mobilenet_weights.h5')
        
    for i in range(len(df_image)):     
        photo_id = df_image['id'][i]
        image_name = os.path.join(Photo_path, str(photo_id)+str('.jpg'))

        img = load_img(image_name)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        x = preprocess_input(x)

        scores = model.predict(x, batch_size=1, verbose=1)[0]

        mean = mean_score(scores)
        std = std_score(scores)
        
       # with open("output.bin", "wb") as output:
       #     pickle.dump(NIMA_score, output)   
       # 
        print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
        
        NIMA_dic[photo_id] = mean

