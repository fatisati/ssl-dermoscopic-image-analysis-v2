import resnet20
import tensorflow as tf


def get_resnet_encoder(resnet_backbone):
    embed_idx = -8
    backbone = tf.keras.Model(
        resnet_backbone.input, resnet_backbone.layers[embed_idx].output
    )
    return backbone


# from tf.keras.layers import BatchNormalization
def get_classifier(barlow_encoder, in_shape, out_shape,
                   use_attention=False, trainable_backbone=False):
    # Extract the backbone ResNet20.
    backbone = get_resnet_encoder(barlow_encoder)

    # We then create our linear classifier and train it.
    if not trainable_backbone:
        backbone.trainable = False

    inputs = tf.keras.layers.Input((in_shape, in_shape, 3))
    x = backbone(inputs, training=trainable_backbone)

    if use_attention:
        attention_weights = tf.keras.layers.Dense(x.shape[-1], activation="softmax")(x)
        x = tf.multiply(x, attention_weights)
        # x, _ = BahdanauAttention(10)(x)

    # batch_out = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(out_shape, activation="softmax")(x)
    classifier = tf.keras.Model(inputs, outputs, name="classifier")
    return classifier


def get_linear_classifier(in_shape, out_shape, hidden_dim=2048,
                          use_batchnorm=True, use_dropout=False, dropout_rate=None):
    resnet = resnet20.ResNet(use_batchnorm, use_dropout, dropout_rate)
    backbone = resnet.get_network(in_shape, hidden_dim=hidden_dim)
    return get_classifier(backbone, in_shape, out_shape, False, False)
