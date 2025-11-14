from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tqdm.notebook import tqdm
import os
import numpy as np
import pandas as pd

TRAIN_DIR = "images/train"
TEST_DIR = "images/test" 

def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)

test = pd.DataFrame()
test['image'], test['label'] = createdataframe(TEST_DIR)

def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode= 'grayscale')
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48,48,1)
    return features

train_features = extract_features(train['image'])

test_features = extract_features(test['image'])

x_train = train_features/255.0
x_test = test_features/255.0

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train['label'])

y_train = le.transform(train['label'])
y_test = le.transform(test['label'])

y_train = to_categorical(y_train, num_classes = 7)
y_test = to_categorical(y_test, num_classes = 7)

model = Sequential()

model.add(Conv2D(128, kernel_size=(3,3), activation= 'relu', input_shape= (48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512, activation= 'relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation= 'relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation= 'softmax'))
          
model.compile(optimizer = 'adam', loss= 'categorical_crossentropy', metrics=['accuracy'])

model.fit(x= x_train, y= y_train, batch_size = 128, epochs = 25, validation_data= (x_test,y_test))

model.save("expressiondetector_modern.keras") # Saving the model in the modern .keras format

label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def single_extract(image):
    img = load_img(image,color_mode= 'grayscale')
    feature = np.array(img)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

import matplotlib.pyplot as plt

image = "images/train/happy/28.jpg"
print("Original image is of happy")
img = single_extract(image)
Predict = model.predict(img)
Predict_label = label[Predict.argmax()]
print("Model prediction is", Predict_label)
plt.imshow(img.reshape(48,48), cmap='gray')