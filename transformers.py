#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import metrics, backend, layers, optimizers, callbacks
import numpy as np

def transformer(timeseriesLength=11, numClasses=2, n_bands=1, num_layers=4, d_model=128, num_heads=8, dff=512, maximum_position_encoding=24, layer_norm=True, activation='linear', loss='categorical_crossentropy', optimiser='Nadam', lr=1e-1):

    inputs = layers.Input((timeseriesLength, n_bands))

    if numClasses==1:
        accuracy = metrics.BinaryAccuracy(name='accuracy', dtype=tf.float32)
        #miou = metrics.MeanIoU(num_classes=2, dtype=tf.float32, name='Mean_IoU')
        miou = metrics.BinaryIoU(target_class_ids=[0, 1], dtype=tf.float32, name='Mean_IoU')
    else:
        #miou = MeanIoU(num_classes=numClasses, dtype=tf.float32, name='Mean_IoU')
        miou = metrics.OneHotMeanIoU(num_classes=numClasses, dtype=tf.float32, name='Mean_IoU')
        accuracy = metrics.CategoricalAccuracy(name='accuracy', dtype=tf.float32)
    optimiserFunction = optimizers.get(optimiser)
    optimiserFunction.learning_rate = lr

    encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, maximum_position_encoding=maximum_position_encoding, layer_norm=layer_norm)
    dense = tf.keras.layers.Dense(units=numClasses,  activation=activation)
    model = tf.keras.Sequential([encoder, dense, tf.keras.layers.MaxPool1D(pool_size=timeseriesLength),
            tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, axis=-2), name='squeeze'),
            tf.keras.layers.Softmax()])
    model.compile(optimizer=optimiserFunction, loss=loss, metrics=[accuracy, miou])
    return model

# This code is taken from the TF tutorial on transformers
# https://www.tensorflow.org/tutorials/text/transformer
def scaled_dot_product_attention(q, k, v, mask=None):
    """ Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class eoMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(eoMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def positional_encoding(positions, d_model, T=10000):

    if isinstance(positions, int):
        positions = np.arange(positions)
    else:
        positions = np.array(positions)

    def _get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(T, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    depths = np.arange(d_model)

    angle_rads = _get_angles(positions[:, np.newaxis],
                            depths[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = eoMultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=None, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1, layer_norm=False, T=10000):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.lnorm_in = tf.keras.layers.LayerNormalization() if layer_norm else None
        self.lnorm_conv = tf.keras.layers.LayerNormalization() if layer_norm else None

        # replace embedding with 1d convolution
        self.conv_in = layers.Conv1D(d_model, 1)
        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model, T=T)

        encoder_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                          for _ in range(num_layers)]
        self.encoder = tf.keras.Sequential(encoder_layers)

        self.dropout = tf.keras.layers.Dropout(rate)


    def call(self, x, training=None, mask=None):
        seq_len = tf.shape(x)[1]

        if self.lnorm_in:
            x = self.lnorm_in(x)

        # adding embedding and position encoding.
        x = self.conv_in(x, training=training)  # (batch_size, input_seq_len, d_model)
        if self.lnorm_conv:
            x = self.lnorm_conv(x)

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        x = self.encoder(x, training=training, mask=mask)

        return x  # (batch_size, input_seq_len, d_model)