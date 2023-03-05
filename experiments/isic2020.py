import os.path

from data.isic2020 import ISIC2020
from utils import augmentation_utils
from models import models
from utils import tf_utils
from train import train_model
from experiments.experiment_params import ExperimentParams
import tensorflow as tf
import numpy as np


# experiment params:
# dataset
# data params: image size, augmentation, batch-size
# model architecture params (batchnorm, dropout, attention, hidden_dim)
# compile params (loss, optimizer, metrics)
# train params (epochs)

# load data
# build model
# compile model
# train
# evaluate
# plots

class Isic2020Experiment:
    def __init__(self, pretrained_path, image_folder, label_file_path, model_dir):
        self.pretrained_path, self.image_folder, self.label_file_path, self.model_dir = pretrained_path, image_folder, label_file_path, model_dir
        self.batch_size, self.image_size = 64, 128
        self.batchnorm, self.dropout, self.attention, self.hidden_dim = True, False, False, 2048
        self.label_dim = 2

        # compile params
        self.loss, self.optimizer, self.metrics = 'binary_crossentropy', 'adam', tf_utils.get_metrics()

    def pretrained_semi_supervised(self, epochs, name=''):
        return self.pretrained(self.pretrained_path, epochs, True, name)

    def pretrain_linear(self, epochs, name=''):
        return self.pretrained(self.pretrained_path, epochs, False, name)

    def pretrained(self, pretrained_path, epochs, trainable, name=''):
        def get_pretrained_model():
            pretrained = tf.keras.models.load_model(pretrained_path)
            model = models.get_classifier(pretrained, self.image_size, self.label_dim, use_attention=False,
                                          trainable_backbone=trainable)
            return model

        experiment_id = 'pretrained'
        if trainable:
            experiment_id += '-semi-supervised'
        else:
            experiment_id += '-linear'
        if len(name) > 0:
            experiment_id = f'{experiment_id}-{name}'
        self.standard_experiment(epochs, experiment_id, get_pretrained_model)

    def linear(self, epochs, name=''):
        def get_linear_model():
            return models.get_linear_classifier(self.image_size, self.label_dim, hidden_dim=self.hidden_dim,
                                                use_batchnorm=self.batchnorm, use_dropout=self.dropout)

        experiment_id = 'linear'
        if len(name) > 0:
            experiment_id = f'{experiment_id}-{name}'
        self.standard_experiment(epochs, experiment_id, get_linear_model)

    def resnet_fully_supervised(self, epochs, postfix=None):
        def get_fully_supervised_model():
            return models.get_resnet_fully_supervised_classifier(self.image_size, self.label_dim,
                                                                 hidden_dim=self.hidden_dim,
                                                                 use_batchnorm=self.batchnorm, use_dropout=self.dropout)

        name = 'resnet-fully-supervised'
        if postfix:
            name += f'-{postfix}'
        self.standard_experiment(epochs, name,
                                 get_fully_supervised_model)

    def standard_experiment(self, epochs, model_name, get_model):
        result_dir = f'{self.model_dir}/isic2020/{model_name}/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        # data params
        print('loading data...')
        aug_func = lambda img: augmentation_utils.dermoscopic_augment(img, self.image_size)
        aug_name = 'dermoscopic-augment'
        data = ISIC2020(self.image_folder, self.label_file_path, self.image_size)
        train, test, validation = data.get_train_test_validation(aug_func, self.batch_size)
        print('---done---')

        # model params
        print('define and compiling model...')
        model = get_model()

        # compile params
        model.compile(self.optimizer, self.loss, self.metrics)
        print('---done---')

        print('training...')
        experiment_params = ExperimentParams(data, self.batch_size, self.image_size, aug_name,
                                             self.batchnorm, self.dropout, self.attention, self.hidden_dim,
                                             model_name, self.loss, self.optimizer, epochs)
        experiment_params.save_params(result_dir)
        history = train_model(model, train, test, validation, result_dir, epochs)
        print('---done---')
        return history

    def compare_models_predictions(self, models_path_list, model_names):
        models_list = [models.load_resent_classifier(model_path, self.image_size, 2) for model_path in models_path_list]
        aug_func = lambda img: augmentation_utils.dermoscopic_augment(img, self.image_size)
        isic2020 = ISIC2020(self.image_folder, self.label_file_path, self.image_size)
        data = isic2020.get_ds(aug_func, self.batch_size)

        predictions = [model.predict(data) for model in models_list]

        def clean_prediction(prediction):
            return [np.argmax(prediction_row) for prediction_row in prediction]

        predictions = [clean_prediction(prediction) for prediction in predictions]
        data_df = isic2020.data_df

        def add_model_prediction_correctness(model_name, prediction):
            data_df[model_name] = prediction == data_df[isic2020.label_col]
            return data_df

        agreement = np.array([True] * len(data_df))
        for model_name, prediction in zip(model_names, predictions):
            data_df = add_model_prediction_correctness(model_name, prediction)
            agreement = np.logical_and(agreement, data_df[model_name])
        data_df['agreement'] = agreement
        return data_df
