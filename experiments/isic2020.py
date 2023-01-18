import os.path

from data.isic2020 import ISIC2020
from utils import augmentation_utils
from models import models
from utils import tf_utils
from train import train_model
from experiments.experiment_params import ExperimentParams


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

def linear(image_folder, label_file_path, model_dir, epochs, name):
    def get_linear_model(image_size, label_dim, batchnorm, dropout, attention, hidden_dim):
        return models.get_linear_classifier(image_size, label_dim, hidden_dim=hidden_dim,
                                            use_batchnorm=batchnorm, use_dropout=dropout)

    experiment_id = 'linear'
    if len(name) > 0:
        experiment_id = f'{experiment_id}-{name}'
    standard_experiment(image_folder, label_file_path, model_dir, epochs, experiment_id, get_linear_model)


def resnet_fully_supervised(image_folder, label_file_path, model_dir, epochs):
    def get_fully_supervised_model(image_size, label_dim, batchnorm, dropout, attention, hidden_dim):
        return models.get_resnet_fully_supervised_classifier(image_size, label_dim, hidden_dim=hidden_dim,
                                                             use_batchnorm=batchnorm, use_dropout=dropout)

    standard_experiment(image_folder, label_file_path, model_dir, epochs, 'resnet-fully-supervised',
                        get_fully_supervised_model)


def standard_experiment(image_folder, label_file_path, model_dir, epochs, model_name, get_model):
    result_dir = f'{model_dir}/isic2020/{model_name}/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # data params
    print('loading data...')
    batch_size, image_size = 64, 128
    aug_func = lambda img: augmentation_utils.dermoscopic_augment(img, image_size)
    aug_name = 'dermoscopic-augment'
    data = ISIC2020(image_folder, label_file_path, image_size)
    train, test, validation = data.get_train_test_validation(aug_func, batch_size)
    print('---done---')

    # model params
    print('define and compiling model...')
    batchnorm, dropout, attention, hidden_dim = True, False, False, 2048
    model = get_model(image_size, len(data.label_set), batchnorm, dropout, attention, hidden_dim)

    # compile params
    loss, optimizer, metrics = 'binary_crossentropy', 'adam', tf_utils.get_metrics()
    model.compile(optimizer, loss, metrics)
    print('---done---')

    print('training...')
    experminet_params = ExperimentParams(data, batch_size, image_size, aug_name,
                                         batchnorm, dropout, attention, hidden_dim,
                                         model_name, loss, optimizer, epochs)
    experminet_params.save_params(result_dir)
    history = train_model(model, train, test, validation, result_dir, epochs)
    print('---done---')
    return history
