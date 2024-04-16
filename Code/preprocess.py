import os
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

class ArtDataPreprocessor:
    """ Class for preprocessing the WikiArt dataset for art classification. """

    def __init__(self, data_dir, classes):
        self.data_dir = data_dir
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.train_data, self.test_data = self._load_and_split_data()

    def _load_and_split_data(self, test_size=0.2, random_state=42):
        """ Load and split data into train and test sets. """
        train_data = []
        test_data = []

        for cls in self.classes:
            class_dir = os.path.join(self.data_dir, cls)

            if not os.path.isdir(class_dir):
                continue
            
            images = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
            images = [os.path.join(class_dir, img) for img in images]\

            # split into train and test
            train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=random_state)

            train_data.extend([(img, self.class_to_idx[cls]) for img in train_imgs])
            test_data.extend([(img, self.class_to_idx[cls]) for img in test_imgs])

        return train_data, test_data

    def preprocess_image(self, image_path, target_size=(224, 224)):
        """ Preprocess image for deep learning model input. """
        img = Image.open(image_path)
        img = img.convert('RGB')  # ensure 3 channels if image is GS
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # normalize pixel values to [0, 1]
        return img_array

    def get_train_data(self):
        """ Get preprocessed training data. """
        random.shuffle(self.train_data)
        images, labels = zip(*self.train_data)
        image_arrays = [self.preprocess_image(img) for img in images]
        return np.array(image_arrays), np.array(labels)

    def get_test_data(self):
        """ Get preprocessed test data. """
        random.shuffle(self.test_data)
        images, labels = zip(*self.test_data)
        image_arrays = [self.preprocess_image(img) for img in images]
        return np.array(image_arrays), np.array(labels)


if __name__ == '__main__':
    data_dir = 'wikiart'  # path to wikiart dataset
    classes = ['Cubism', 'Mannerism_Late_Renaissance']  # insert classes

    # initialize preprocessor
    preprocessor = ArtDataPreprocessor(data_dir, classes)

    # get preprocessed train and test data
    X_train, y_train = preprocessor.get_train_data()
    X_test, y_test = preprocessor.get_test_data()

    print(f"Train data shape: {X_train.shape}, labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, labels shape: {y_test.shape}")
    print(X_train[0].shape,X_train[2806].shape)
    print(y_train[0],y_train[2806])

    

