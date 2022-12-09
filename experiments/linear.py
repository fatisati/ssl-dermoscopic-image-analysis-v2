from data.isic2020 import ISIC2020
from utils import augmentation_utils
from models import models
from utils import tf_utils
from train import train_model
from experiment_params import ExperimentParams


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

def isic2020():
    image_folder, label_file_path = '', ''
    image_col, label_col, = 'image_name', 'target'

    # data params
    batch_size, image_size = 512, 128
    aug_func = lambda img: augmentation_utils.dermoscopic_augment(img, image_size)
    aug_name = 'dermoscopic-augment'
    data = ISIC2020(image_folder, label_file_path,
                    image_col, label_col, image_size)
    train, test, validation = data.get_train_test_validation(aug_func, batch_size)

    # model params
    batchnorm, dropout, attention, hidden_dim = True, False, False, 2048
    model = models.get_linear_classifier(image_size, len(data.label_set), hidden_dim=hidden_dim,
                                         use_batchnorm=batchnorm, use_dropout=dropout)
    model_name = 'linear'

    # compile params
    loss, optimizer, metrics = 'binary_crossentropy', 'adam', tf_utils.get_metrics()
    model.compile(optimizer, loss, metrics)

    name = 'isic2020-linear'
    path = '../models/'
    epochs = 500

    experminet_params = ExperimentParams(data, batch_size, image_size, aug_name,
                                         batchnorm, dropout, attention, hidden_dim,
                                         model_name, loss, optimizer, epochs)
    experminet_params.save_params(path + name + '/')
    train_model(model, train, test, validation, name, path, epochs)
