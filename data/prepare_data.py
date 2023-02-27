import pandas as pd
from sklearn.model_selection import train_test_split
import zipfile
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd
import shutil
import os


def sample_data(df, sample_size):
    return df.sample(sample_size)


def identify_train_test_validation(x, test_data, validation):
    if x in test_data:
        return 'test'
    if x in validation:
        return 'validation'
    return 'train'


def split_data(df, test_ratio, validation_ratio, x_col, label_col):
    train, test = train_test_split(df, test_size=test_ratio, random_state=0, stratify=df[[label_col]], shuffle=True)
    if validation_ratio > 0:
        train, validation = train_test_split(train, test_size=validation_ratio, random_state=0,
                                             stratify=train[[label_col]])
        validation_names = list(validation[x_col])
    else:
        validation_names = []
    test_names = list(test[x_col])
    df['split'] = [identify_train_test_validation(x, test_names, validation_names) for x in df[x_col]]
    print(
        f'test: {len(test_names)}, validation: {len(validation_names)}, train: {len(df) - len(test_names) - len(validation_names)}')
    return df


def copy_sample_data_from_zip(sample_df, image_col, image_zip_path, dest_path,
                              zip_subfolder, image_extension):
    zip_file = zipfile.ZipFile(image_zip_path)
    for image_name in sample_df[image_col]:
        zip_file.extract(f'{zip_subfolder}/{image_name}.{image_extension}',
                         dest_path)


def copy_isic_sample_to_drive():
    drive_path = ''
    zip_subfolder = ''
    zip_folder_path = ''
    sample_df = pd.read_csv('sample-1000.csv')
    dest_path = drive_path + 'miccai/isic2020/sample-1000/'
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    copy_sample_data_from_zip(sample_df, 'image_name', zip_folder_path, dest_path, zip_subfolder, '.jpg')


def copy_sample_data(sample_df, image_col, image_folder, dest_folder, image_extension='.jpg'):
    for img in sample_df[image_col]:
        img += image_extension
        shutil.copy(image_folder + img, dest_folder + img)


def balance_data(train_df, x_col, y_col, Sampler):
    x_train, y_train = np.array(train_df[x_col]), np.array(train_df[y_col])
    sampler = Sampler(random_state=42)

    X_res, y_res = sampler.fit_resample(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
    print(f"Training target statistics: {Counter(y_res)}")

    balanced_df = pd.DataFrame({x_col: list(X_res[:, 0]), y_col: list(y_res)})
    return balanced_df


def split_isic_data():
    data_folder = '~/data/master-thesis/'
    data = pd.read_csv(f'{data_folder}/labels.csv')
    splited = split_data(data, 0.3, 0.1, 'image_name', 'target')
    splited.to_csv(f'{data_folder}/splitted-data.csv')


def under_sample_splitted_isic():
    data_folder = '~/data/master-thesis/'
    df = pd.read_csv(data_folder + 'splitted.csv')
    train = df[df['split'] == 'train']

    balanced_train = balance_data(train, 'image_name', 'target', RandomUnderSampler)
    print(f'len original train: {len(train)}, len undersample train: {len(balanced_train)}')
    to_drop = train[~train['image_name'].isin(balanced_train['image_name'])]
    balanced_df = df[~df['image_name'].isin(to_drop['image_name'])]
    print(f'df len: {len(df)}, under sample df len: {len(balanced_df)}')
    balanced_df.to_csv(data_folder + 'splitted-under-sample.csv')


def under_sample_isic():
    data_folder = '~/data/master-thesis/'
    df = pd.read_csv(data_folder + 'labels.csv')

    balanced = balance_data(df, 'image_name', 'target', RandomUnderSampler)
    balanced.to_csv(data_folder + 'under-sample.csv')


def split_under_sample():
    data_folder = '~/data/master-thesis/'
    data = pd.read_csv(f'{data_folder}/under-sample.csv')
    splited = split_data(data, 0.2, 0.25, 'image_name', 'target')
    splited.to_csv(f'{data_folder}/under-sample-splitted.csv')


def sample_train_data():
    data_folder = 'F:/data/isic2020/'
    data_path = data_folder + 'under-sample-splitted.csv'
    data = pd.read_csv(data_path)
    train = data[data['split'] == 'train']
    sample_train, _ = train_test_split(train, test_size = 0.9, random_state=0, stratify=train[['target']],
                                       shuffle=True)
    non_train = data[data['split'] != 'train']
    keep_names = set(sample_train['image_name']).union(set(non_train['image_name']))
    sampled_data = data[data['image_name'].isin(list(keep_names))]
    sampled_data.to_csv(data_folder + f'sample-train-10percent.csv')


if __name__ == '__main__':
    sample_train_data()
    # under_sample_isic()
    # split_under_sample()
