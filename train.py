import tensorflow as tf
from keras.callbacks import CSVLogger


def save_metrics(result_folder, test_metrics, validation_metrics, metric_names):
    test_metrics = [str(metric) for metric in test_metrics]
    validation_metrics = [str(metric) for metric in validation_metrics]
    eval_file = open(f'{result_folder}/metrics.csv', 'w')
    eval_file.write(f'data, {",".join(metric_names)}\n')
    eval_file.write(f'validation, {",".join(validation_metrics)}\n')
    eval_file.write(f'test, {",".join(test_metrics)}\n')
    eval_file.close()


# https://www.tensorflow.org/tutorials/keras/save_and_load
def train_model(model, train, test, validation, result_dir, epochs):
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = f"{result_dir}/" + "best.ckpt"

    latest = tf.train.latest_checkpoint(result_dir)
    if latest:
        print(f'loading model weights from checkpoint_path {latest}')
        model.load_weights(latest)
    else:
        print('no checkpoint found')

    # Create a callback that saves the model's weights every 10 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=10)

    csv_logger = CSVLogger(f'{result_dir}/log.csv', append=True, separator=',')

    # Train the model with the new callback
    history = model.fit(train,
                        epochs=epochs,
                        callbacks=[cp_callback, csv_logger],
                        validation_data=validation,
                        verbose=1)
    test_metrics = model.evaluate(test)
    validation_metrics = model.evaluate(validation)
    metric_names = model.metrics_names
    save_metrics(result_dir, test_metrics, validation_metrics, metric_names)
    return history
