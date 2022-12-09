import tensorflow as tf


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # img = tf.image.per_image_standardization(img)
    # img = tf.divide(img, 255)
    return img


def read_tf_image(path, size = None):
    img = tf.io.read_file(path)
    img = decode_img(img)
    if size:
        img = tf.image.resize(img, (size, size))
    return img


def tf_ds_from_arr(arr):
    return tf.data.Dataset.from_tensor_slices(arr)

def get_metrics():
    METRICS = [
        # keras.metrics.TruePositives(name='tp'),
        # keras.metrics.FalsePositives(name='fp'),
        # keras.metrics.TrueNegatives(name='tn'),
        # keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
        tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ]
    return METRICS