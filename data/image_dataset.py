import pandas as pd
import tensorflow as tf
from utils.tf_utils import read_tf_image
from utils.augmentation_utils import dermoscopic_augment


class ImageDataset:
    def __init__(self, image_folder, data_df,
                 image_col, label_col, image_size, split_col='split', image_extension='.jpg'):
        self.image_folder = image_folder
        self.data_df = data_df
        self.label_set = set(self.data_df[label_col])
        self.image_col = image_col
        self.label_col = label_col
        self.split_col = split_col
        self.image_size = image_size
        self.image_extension = image_extension

    def process_sample(self, name, label, aug_func):
        img = read_tf_image(self.image_folder + name + self.image_extension, self.image_size)
        img = aug_func(img)
        return img, label

    def one_hot(self, label):
        one_hot = [0] * len(self.label_set)
        label_idx = list(self.label_set).index(label)
        one_hot[label_idx] = 1
        return one_hot

    def get_ds(self, aug_func, batch_size, split=None):
        if split:
            data_split = self.data_df[self.data_df[self.split_col] == split]
        else:
            data_split = self.data_df
        image_names = list(data_split[self.image_col])
        labels = list(data_split[self.label_col])
        labels = [self.one_hot(label) for label in labels]
        img_ds = tf.data.Dataset.from_tensor_slices(image_names)
        labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        ds = tf.data.Dataset.zip((img_ds, labels_ds)).map(lambda img, label: self.process_sample(img, label, aug_func))
        return ds.shuffle(1024).batch(batch_size)

    def get_train_test_validation(self, aug_func, batch_size):
        return self.get_ds(aug_func, batch_size, 'train'), \
            self.get_ds(aug_func, batch_size, 'test'), \
            self.get_ds(aug_func, batch_size, 'validation')


if __name__ == '__main__':
    image_folder, label_file_path = '', ''
    image_col, label_col, image_size = 'image_name', 'target', 128
    image_ds = ImageDataset(image_folder, label_file_path,
                            image_col, label_col, image_size)
    aug_func = lambda img: dermoscopic_augment(img, image_size)
    batch_size = 512
    train_ds = image_ds.get_ds(aug_func, batch_size, 'train')
    test_ds = image_ds.get_ds(aug_func, batch_size, 'test')
