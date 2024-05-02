
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,concatenate,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose, BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras.datasets import fashion_mnist
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from preprocess import ArtDataPreprocessor
from keras.optimizers import legacy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

classes = ['Abstract_Expressionism', 'Art_Nouveau_Modern','Baroque', 'Cubism', 'Expressionism', 'Impressionism', 
'Minimalism', 'Mannerism_Late_Renaissance', 'Naive_Art_Primitivism', 'Northern_Renaissance', 'Pop_Art', 'Post_Impressionism', 
'Realism', 'Rococo', 'Romanticism', 'Symbolism']

batch_size = 64
epochs = 2
inChannel = 3
x, y = 224, 224
num_classes = len(classes)

def encoder(input_img):
    #encoder
    #input = 224 x 224 x 3 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #224 x 224 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #112 x 112 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #112 x 112 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #56 x 56 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #56 x 56 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #28 x 28 x 64
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3) #28 x 28 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    drp = Dropout(0.2)(den)
    den = Dense(64, activation='relu')(flat)
    drp = Dropout(0.2)(den)
    den = Dense(64, activation='relu')(flat)
    drp = Dropout(0.2)(den)
    out = Dense(num_classes, activation='softmax')(drp)
    return out



input_img = Input(shape = (x, y, inChannel))
encode = encoder(input_img)
full_model = Model(input_img,fc(encode))
full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=legacy.Adam(),metrics=['accuracy'])

test_data = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')

encoder_layer = Model(inputs=full_model.input, outputs=full_model.get_layer('batch_normalization_7').output)
encoded_images = encoder_layer.predict(test_data)
print(f"Images encoded! Encoded images shape is {encoded_images.shape}\n")

num_img = encoded_images.shape[0]
flattened_encodings = np.reshape(encoded_images, (num_img, -1))
pca = PCA(n_components=2)
pca_reduced_encodings = pca.fit_transform(flattened_encodings)
print(f"PCA encodings shape is {pca_reduced_encodings.shape}\n")
plt.scatter(pca_reduced_encodings[:, 0], pca_reduced_encodings[:, 1], c=test_labels, s=50, cmap='viridis')
plt.show()

# Plotting TSNE components
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(flattened_encodings)
print(f"tsne encodings shape is {X_tsne.shape}")
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=test_labels, s=50, cmap='viridis')
plt.show()