# https://keras.io/examples/vision/grad_cam/

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from models import models

from data.isic2020 import ISIC2020
from utils import augmentation_utils
import os
import random


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    plt.imshow(superimposed_img)

    # # Save the superimposed image
    # superimposed_img.save(cam_path)

    # # Display Grad CAM
    # display(Image(cam_path))


def find_last_conv_name(layers):
    i = 1
    while 'conv' not in layers[-i]:
        i += 1
    return layers[-i]


def model_images_gradcam(encoder, image_names, image_folder):
    encoder_layers = [layer.name for layer in encoder.layers]
    last_conv_name = find_last_conv_name(encoder_layers)

    image_path_arr = [image_folder + name for name in image_names]

    all_imgs = []
    heatmaps = []
    for image_path in image_path_arr:
        img = get_img_array(image_path, (128, 128))
        heatmap = make_gradcam_heatmap(img, encoder, last_conv_name, pred_index=None)

        all_imgs.append(img)
        heatmaps.append(heatmap)

    fig = plt.figure(figsize=(10, 7))

    i = 1
    for img_path, heatmap in zip(image_path_arr, heatmaps):
        fig.add_subplot(3, 5, i)
        i += 1
        save_and_display_gradcam(img_path, heatmap, alpha=0.4)
        plt.axis('off')

    for img_name in image_names:
        fig.add_subplot(3, 5, i)
        i += 1

        img = keras.preprocessing.image.load_img(image_folder + img_name)
        img = keras.preprocessing.image.img_to_array(img)
        img = keras.preprocessing.image.array_to_img(img)
        plt.axis('off')
        plt.imshow(img)

    for map in heatmaps:
        fig.add_subplot(3, 5, i)
        i += 1
        plt.axis('off')
        plt.imshow(map)

    plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)


if __name__ == '__main__':
    # load model
    drive_path = '/content/drive/MyDrive/'
    model_dir = drive_path + 'miccai/models/isic2020/'
    model_path = model_dir + 'linear/best.ckpt'
    image_size = 128
    out_shape = 2
    model = models.load_resent_classifier(model_path, image_size, out_shape)
    encoder = models.get_resnet_model_encoder(model)

    # choose samples
    image_folder = '/content/isic2020-under-sample-images/'
    all_img_names = list(os.listdir(image_folder))
    sample_img_name = random.sample(all_img_names
                                    , 5)

    # show gradcam
    model_images_gradcam(encoder, sample_img_name, image_folder)
