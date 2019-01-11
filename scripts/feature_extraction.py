from sklearn.feature_extraction.text import TfidfVectorizer
import tensornets as nets
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import numpy as np
import os
import pandas as pd

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

def image_feature_extraction(df):
    '''
    images: A numpy 4D array of shape (no. of examples,299,299,3)
    '''
    if os.path.exists("../data/image_features.npy"):
        ans=input("Model already exists,do you want to re run it?(y/n)\n")
        if ans=="n":
            return
        else:
            pass
    file_list = os.listdir('../data/train_images')
    print(len(file_list))
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df,
        directory="../data/train_images/",
        x_col="image",
        y_col=None,
        batch_size=32,
        class_mode=None,
        target_size=(229,229))
    filenames = test_generator.filenames
    nb_samples = len(filenames)
    print(nb_samples)
    model=keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', pooling='avg')
    model.layers.pop()
    model = Model(model.input, model.layers[-1].output)
    image_features = model.predict_generator(test_generator,steps = np.ceil(nb_samples/64))
    print(image_features.shape)
    #np.save(image_features,"../data/image_features.npy")        

data=pd.read_csv("../data/train.csv")
image_feature_extraction(data)
