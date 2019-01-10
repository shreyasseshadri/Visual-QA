from sklearn.feature_extraction.text import TfidfVectorizer
import tensornets as nets
import keras
from keras.models import Model
import numpy as np
import os
def text_feature_extraction(corpus):
    '''
    Corpus : A list of Questions in the dataset
    '''
    if os.path.exists("../data/text_features.npy"):
        ans=input("Model already exists,do you want to re run it?(y/n)\n")
        if ans=="n":
            return
        else:
            pass
    vectorizer=TfidfVectorizer()
    text_features=np.array(vectorizer.fit_transform(corpus).todense())
    np.save(text_features,"../data/text_features.npy")

def image_feature_extraction(images):
    '''
    images: A numpy 4D array of shape (no. of examples,299,299,3)
    '''
    if os.path.exists("../data/image_features.npy"):
        ans=input("Model already exists,do you want to re run it?(y/n)\n")
        if ans=="n":
            return
        else:
            pass
    model=keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', pooling='avg')
    model.pop()
    model = Model(model.input, model.layers[-1].output)
    image_features=model.predict(images)
    np.save(image_features,"../data/image_features.npy")        
