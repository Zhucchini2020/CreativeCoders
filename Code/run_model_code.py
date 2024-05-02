import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #model will be trained on GPU 0

import keras
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,concatenate,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose, BatchNormalization
from keras.models import Model,Sequential
from keras.losses import MSE, categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from keras.optimizers.legacy import Adam
from create_train_test_code import train_test_code
from hyperparams import *
from sklearn.model_selection import train_test_split
from run_autoencoder_code import autoencoder_code

# Create dictionary of target classes
label_dict = train_test_code(CREATE_NEW_DATA)

train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')
train_labels = np.load('train_labels.npy')
test_labels = np.load('test_labels.npy')

# Shapes of training set
print(f"Training set (images) shape: {train_data.shape}")

# Shapes of test set
print(f"Test set (images) shape: {test_data.shape}")

# Types of train and test data
print(f"Train type {train_data.dtype}")
print(f"Test type {test_data.dtype}")

train_X,valid_X,train_ground,valid_ground = train_test_split(train_data,
                                                             train_data,
                                                             test_size=0.2,
                                                             random_state=13)


epochs = 2
inChannel = 3
x, y = IMG_X, IMG_Y
input_img = Input(shape = (x, y, inChannel))
num_classes = len(CLASSES)

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

def decoder(conv4):
    #decoder
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #28 x 28 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    up3 = UpSampling2D((2,2))(conv5) #56 x 56 x 64
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up3) #56 x 56 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #112 x 112 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 112 x 112 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 224 x 224 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 224 x 224 x 1
    return decoded

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

def class_cce(y_true, y_pred):
    class_true = y_true[1]
    class_pred = y_pred[1]
    return categorical_crossentropy(class_true, class_pred)

def auto_mse(y_true, y_pred):
    auto_true = y_true[0]
    auto_pred = y_pred[0]
    return MSE(auto_true, auto_pred)

def joint_loss():
    def my_loss(y_true, y_pred):       
        return AUTOENCODER_WEIGHT * auto_mse(y_true, y_pred) + CLASSIFIER_WEIGHT * class_cce(y_true, y_pred)
    return my_loss

# Start by training the autoencoder
autoencoder = autoencoder_code(CREATE_NEW_AUTOENCODER, input_img, decoder, encoder, train_X, train_ground, valid_X, valid_ground)

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_labels)
test_Y_one_hot = to_categorical(test_labels)

# Display the change for category label using one-hot encoding
print('Original label:', train_labels[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

# Testing shapes of different areas
train_X,valid_X,train_label,valid_label = train_test_split(train_data,train_Y_one_hot,test_size=0.2,random_state=13)
print(f'train_X shape: {train_X.shape}, valid_X shape: {valid_X.shape}, train_label shape: {train_label.shape}, valid_label shape: {valid_label.shape}')

# Then transfer the weights to the full model, train jointly on both losses
full_model = Model(input_img, [decoder(encoder(input_img)),fc(encoder(input_img))])
full_model.compile(loss = joint_loss, optimizer = Adam())

# Transfer weights from pre-trained autoencoder to new model
for l1,l2 in zip(full_model.layers[:20],autoencoder.layers[0:20]):
    l1.set_weights(l2.get_weights())

# Start by training only the feed forward layer
for layer in full_model.layers[0:20]:
    layer.trainable = False
full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam(),metrics=['accuracy', auto_mse, class_cce])
full_model.summary()
joint_train_label = [train_ground, train_label]
joint_valid_label = [valid_ground, valid_label]
classify_train = full_model.fit(train_X, joint_train_label, batch_size=BATCH_SIZE,epochs=FULL_MODEL_FIRST_EPOCHS,verbose=1,validation_data=(valid_X, joint_valid_label))
full_model.save_weights('autoencoder_classification.h5')

# Then train both the encoder and the feed forward layer
for layer in full_model.layers[0:20]:
    layer.trainable = True
full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam(),metrics=['accuracy'])
classify_train = full_model.fit(train_X, joint_train_label, batch_size=BATCH_SIZE,epochs=FULL_MODEL_SECOND_EPOCHS,verbose=1,validation_data=(valid_X, joint_valid_label))
full_model.save_weights('classification_complete.h5')
# full_model.load_weights('classification_complete.h5')
accuracy = classify_train.history['accuracy']
val_accuracy = classify_train.history['val_accuracy']
loss = classify_train.history['loss']
val_loss = classify_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# Test visualizations, etc.
test_eval = full_model.evaluate(test_data, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
predicted_classes = full_model.predict(test_data)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
predicted_classes.shape, test_labels.shape
correct = np.where(predicted_classes==test_labels)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_data[correct].reshape((IMG_X,IMG_Y,3)), interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_labels[correct]))
    plt.tight_layout()
incorrect = np.where(predicted_classes!=test_labels)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_data[incorrect].reshape((IMG_X,IMG_Y,3)), interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_labels[incorrect]))
    plt.tight_layout()


from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_labels, predicted_classes, target_names=target_names))
