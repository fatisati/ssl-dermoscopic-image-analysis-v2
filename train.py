import time
import os
import tensorflow as tf
import numpy as np
from keras.callbacks import CSVLogger


def print_time(st):
    print(f'done took {time.time() - st}')


def load_model(path):
    print(f'loading model in path: {path} ...')
    st = time.time()
    model = tf.keras.models.load_model(path)
    print_time(st)
    return model


def find_latest_model(path, name):
    if name not in os.listdir(path):
        return -1, -1
    files = os.listdir(path + name)
    model_files = []
    for file in files:
        if ('.' not in file) and (file[0] == 'e'):
            model_files.append(int(file[1:]))
    if len(model_files) == 0:
        return -1, -1
    model_files = sorted(model_files)
    print(model_files)
    model = load_model(f'{path}{name}/e{model_files[-1]}')
    print(f'best founded model {model_files[-1]}')
    return model, int(model_files[-1])


def check_folder(path, folder):
    print('generating model folder if not exist...')
    if not folder in os.listdir(path):
        os.mkdir(path + folder)
    print('done')


# https://www.tensorflow.org/tutorials/keras/save_and_load
def train_model(model, train, test, validation, name, path, epochs):
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = f"{path}/{name}" + "-{epoch:04d}.ckpt"

    # Create a callback that saves the model's weights every 10 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=10)

    csv_logger = CSVLogger(path + name + '/log.csv', append=True, separator=',')

    # Train the model with the new callback
    history = model.fit(train,
                        test,
                        epochs=epochs,
                        callbacks=[cp_callback, csv_logger],
                        validation_data=validation,
                        verbose=1)
    return history