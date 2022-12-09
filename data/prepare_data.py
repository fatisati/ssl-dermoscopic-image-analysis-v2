import pandas as pd
from sklearn.model_selection import train_test_split
import zipfile


def sample_data(df, sample_size):
    return df.sample(sample_size)


def identify_train_test_validation(x, test_data, validation):
    if x in test_data:
        return 'test'
    if x in validation:
        return 'validation'
    return 'train'


def split_data(df, test_ratio, validation_ratio, x_col, label_col):
    train, test = train_test_split(df, test_size=test_ratio, random_state=0, stratify=df[[label_col]])
    validation_ratio = validation_ratio / test_ratio
    test, validation = train_test_split(test, test_size=validation_ratio, random_state=0, stratify=test[[label_col]])
    df['split'] = [identify_train_test_validation(x, list(test[x_col]), list(validation[x_col])) for x in df[x_col]]
    return df


def copy_sample_data(sample_df, image_col, image_zip_path, dest_path,
                     zip_subfolder, image_extension):
    zip_file = zipfile.ZipFile(image_zip_path)
    for image_name in sample_df[image_col]:
        zip_file.extract(f'{zip_subfolder}/{image_name}.{image_extension}',
                         dest_path)


if __name__ == '__main__':
    data_folder = '~/data/master-thesis/'
    data = pd.read_csv(f'{data_folder}/labels.csv')
    # sample_size = 200
    # sample = sample_data(data, sample_size)
    splited = split_data(data, 0.3, 0.1, 'image_name', 'target')
    splited.to_csv(f'{data_folder}/splitted-data.csv')
