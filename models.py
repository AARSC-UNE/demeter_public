
import tensorflow as tf
import numpy as np
import tensorflow_decision_forests as tfdf
from sklearn import svm

"""Note these model are not compatible with tensorflow version 2.16.0 and above"""

def transformer_regression(timeseriesLength=11, numClasses=2, num_layers=4, d_model=128, num_heads=8, dff=512, 
                           maximum_position_encoding=24, layer_norm=True, activation='linear', rate=0.1, classActivation='linear'):
    """
    Creates a transformer model. For a regression problem use the linear activation function in the final class activation layer. 
    For a classification problem use the softmax activation function in the final class activation layer.

    Args:
        timeseriesLength (int): The length of the input timeseries.
        numClasses (int): The number of output classes.
        num_layers (int): The number of transformer layers.
        d_model (int): The dimensionality of the transformer model.
        num_heads (int): The number of attention heads in the transformer model.
        dff (int): The dimensionality of the feed-forward network in the transformer model.
        maximum_position_encoding (int): The maximum position encoding value.
        layer_norm (bool): Whether to apply layer normalization in the transformer model.
        activation (str): The activation function to use in the dense layer.
        rate (float): The dropout rate in the transformer model.
        classActivation (str): The activation function to use in the final output layer.

    Returns:
        tf.keras.Model: The transformer regression model.
    """

    encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, 
                      maximum_position_encoding=maximum_position_encoding, layer_norm=layer_norm, rate=rate)
    dense = tf.keras.layers.Dense(units=numClasses,  activation=activation)
    model = tf.keras.Sequential([encoder, dense, tf.keras.layers.MaxPool1D(pool_size=timeseriesLength),
            tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, axis=-2), name='squeeze'), 
            tf.keras.layers.Activation(classActivation)])
    
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
    """
    Custom multi-head attention layer.

    Args:
        d_model (int): The dimensionality of the model.
        num_heads (int): The number of attention heads.

    Attributes:
        num_heads (int): The number of attention heads.
        d_model (int): The dimensionality of the model.
        depth (int): The depth of each attention head.
        wq (tf.keras.layers.Dense): The dense layer for the query projection.
        wk (tf.keras.layers.Dense): The dense layer for the key projection.
        wv (tf.keras.layers.Dense): The dense layer for the value projection.
        dense (tf.keras.layers.Dense): The dense layer for the output projection.

    Methods:
        split_heads(x, batch_size): Splits the last dimension of the input tensor into (num_heads, depth).
        call(v, k, q, mask=None): Performs the forward pass of the multi-head attention layer.

    """

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
        """
        Performs the forward pass of the multi-head attention layer.

        Args:
            v (tf.Tensor): The value tensor of shape (batch_size, seq_len_v, d_model).
            k (tf.Tensor): The key tensor of shape (batch_size, seq_len_k, d_model).
            q (tf.Tensor): The query tensor of shape (batch_size, seq_len_q, d_model).
            mask (tf.Tensor, optional): The mask tensor of shape (batch_size, seq_len_q, seq_len_k), default is None.

        Returns:
            output (tf.Tensor): The output tensor of shape (batch_size, seq_len_q, d_model).
            attention_weights (tf.Tensor): The attention weights tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k).

        """
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
    """
    Creates a point-wise feed-forward network.

    Args:
        d_model (int): The dimensionality of the output space.
        dff (int): The number of units in the feed-forward network.

    Returns:
        tf.keras.Sequential: A sequential model consisting of two dense layers.

    """
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def positional_encoding(positions, d_model, T=10000):
    """
    Generates positional encodings for a given sequence length and model dimension.

    Args:
        positions (int or array-like): The sequence length or an array of positions.
        d_model (int): The model dimension.
        T (int, optional): The scaling factor for the positional encodings. Defaults to 10000.

    Returns:
        tf.Tensor: The positional encodings as a tensor of shape (1, positions, d_model).
    """

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
    """
    EncoderLayer class represents a single layer in the encoder of a transformer model.

    Args:
        d_model (int): The dimensionality of the model.
        num_heads (int): The number of attention heads.
        dff (int): The dimensionality of the feed-forward network.
        rate (float, optional): The dropout rate. Defaults to 0.1.

    Attributes:
        mha (eoMultiHeadAttention): The multi-head attention layer.
        ffn (point_wise_feed_forward_network): The feed-forward network layer.
        layernorm1 (tf.keras.layers.LayerNormalization): The first layer normalization layer.
        layernorm2 (tf.keras.layers.LayerNormalization): The second layer normalization layer.
        dropout1 (tf.keras.layers.Dropout): The first dropout layer.
        dropout2 (tf.keras.layers.Dropout): The second dropout layer.
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = eoMultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=None, mask=None):
        """
        Forward pass of the EncoderLayer.

        Args:
            x (tf.Tensor): The input tensor.
            training (bool, optional): Whether the model is in training mode. Defaults to None.
            mask (tf.Tensor, optional): The mask tensor. Defaults to None.

        Returns:
            tf.Tensor: The output tensor.
        """
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class Encoder(tf.keras.layers.Layer):
    """
    The Encoder class represents the encoder component of a transformer model.

    Args:
        num_layers (int): The number of encoder layers.
        d_model (int): The dimensionality of the model.
        num_heads (int): The number of attention heads.
        dff (int): The dimensionality of the feed-forward network.
        maximum_position_encoding (int): The maximum position encoding value.
        rate (float, optional): The dropout rate. Defaults to 0.1.
        layer_norm (bool, optional): Whether to apply layer normalization. Defaults to False.
        T (int, optional): The value used in the positional encoding calculation. Defaults to 10000.

    Attributes:
        d_model (int): The dimensionality of the model.
        num_layers (int): The number of encoder layers.
        lnorm_in (tf.keras.layers.LayerNormalization): Layer normalization for the input.
        lnorm_conv (tf.keras.layers.LayerNormalization): Layer normalization for the convolutional layer.
        conv_in (tf.keras.layers.Conv1D): 1D convolutional layer.
        pos_encoding (tf.Tensor): Positional encoding tensor.
        encoder (tf.keras.Sequential): Sequential encoder layers.
        dropout (tf.keras.layers.Dropout): Dropout layer.

    Methods:
        call(x, training=None, mask=None): Performs the forward pass of the encoder.
    """

    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1, layer_norm=False, T=10000):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.lnorm_in = tf.keras.layers.LayerNormalization() if layer_norm else None
        self.lnorm_conv = tf.keras.layers.LayerNormalization() if layer_norm else None

        # replace embedding with 1d convolution
        self.conv_in = tf.keras.layers.Conv1D(d_model, 1)
        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model, T=T)

        encoder_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                          for _ in range(num_layers)]
        self.encoder = tf.keras.Sequential(encoder_layers)

        self.dropout = tf.keras.layers.Dropout(rate)


    def call(self, x, training=None, mask=None):
        """
        Performs the forward pass of the encoder.

        Args:
            x (tf.Tensor): The input tensor.
            training (bool, optional): Whether the model is in training mode. Defaults to None.
            mask (tf.Tensor, optional): The mask tensor. Defaults to None.

        Returns:
            tf.Tensor: The output tensor.
        """
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

    # def get_config(self):
    #     config = super().get_config().copy()
    #     config.update({
    #         'num_layers': self.num_layers,
    #         'd_model': self.d_model,
    #         'num_heads': self.num_heads,
    #         'dff': self.dff,
    #         'maximum_position_encoding': self.maximum_position_encoding,
    #         'rate': self.rate,
    #         'layer_norm': self.layer_norm,
    #         'T': self.T,
    #     })
    #     return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

class GRUModel(tf.keras.Model):
    def __init__(self, tsLength, nBands, nLayers=3, nUnits=128, dropout=0.2, activation='relu', batch_norm=False, 
                 kernel_initializer='he_normal', kernel_regularizer=0.001, classActivation='linear'):
        """
        Initialize the Gate Recurrent Unit (GRU) model.

        Args:
            tsLength (int): The length of the time series.
            nBands (int): The number of bands in the time series.
            nLayers (int, optional): The number of GRU layers. Defaults to 3.
            nUnits (int, optional): The number of units in each GRU layer. Defaults to 128.
            dropout (float, optional): The dropout rate. Defaults to 0.2.
            activation (str, optional): The activation function to use. Defaults to 'relu'.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            kernel_initializer (str, optional): The initializer for the kernel weights. Defaults to 'he_normal'.
            kernel_regularizer (float, optional): The regularization strength for the kernel weights. Defaults to 0.001.
            classActivation (str, optional): The activation function for the output layer. Defaults to 'linear'.
        """
        super(GRUModel, self).__init__()

        self.tsLength = tsLength
        self.nBands = nBands
        self.nLayers = nLayers
        self.nUnits = nUnits
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.classActivation = classActivation
        self.input_layer = tf.keras.layers.Input((tsLength, nBands))

        self.gru_layers = []
        for i in range(nLayers):
            self.gru_layers.append(tf.keras.layers.GRU(units=nUnits, return_sequences=True, kernel_initializer=kernel_initializer, 
                                                        kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer)))
            if batch_norm:
                self.gru_layers.append(tf.keras.layers.BatchNormalization())
            
            self.gru_layers.append(tf.keras.layers.Activation(activation))
            self.gru_layers.append(tf.keras.layers.Dropout(dropout))

        self.gru_layers.append(tf.keras.layers.Flatten())

        for i in range(nLayers):
            self.gru_layers.append(tf.keras.layers.Dense(units=nUnits, kernel_initializer=kernel_initializer, 
                                                        kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer)))
            if batch_norm:
                self.gru_layers.append(tf.keras.layers.BatchNormalization())
            
            self.gru_layers.append(tf.keras.layers.Activation(activation))
            self.gru_layers.append(tf.keras.layers.Dropout(dropout))
        
        self.gru_layers.append(tf.keras.layers.Dense(units=1, kernel_initializer=kernel_initializer, 
                                                    kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer)))

        self.gru_layers.append(tf.keras.layers.Activation(classActivation))

        net = self.input_layer
        for layer in self.gru_layers:
            net = layer(net)

        self.model = tf.keras.Model(inputs=self.input_layer, outputs=net)

    def call(self, inputs):
        x = self.model(inputs)
        return x

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'tsLength': self.tsLength,
    #         'nBands': self.nBands,
    #         'nLayers': self.nLayers,
    #         'nUnits': self.nUnits,
    #         'dropout': self.dropout,
    #         'activation': self.activation,
    #         'batch_norm': self.batch_norm,
    #         'kernel_initializer': self.kernel_initializer,
    #         'kernel_regularizer': self.kernel_regularizer,
    #         'classActivation': self.classActivation,
    #     })
    #     return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

class LSTMModel(tf.keras.Model):
    """
    A custom Long Short-Term Memory (LSTM) model.

    Args:
        tsLength (int): The length of the time series.
        nBands (int): The number of bands in the time series.
        nLayers (int, optional): The number of LSTM layers. Defaults to 3.
        nUnits (int, optional): The number of units in each LSTM layer. Defaults to 128.
        dropout (float, optional): The dropout rate. Defaults to 0.2.
        activation (str, optional): The activation function to use. Defaults to 'relu'.
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
        kernel_initializer (str, optional): The initializer for the kernel weights. Defaults to 'he_normal'.
        kernel_regularizer (float, optional): The regularization strength for the kernel weights. Defaults to 0.001.
        classActivation (str, optional): The activation function for the output layer. Defaults to 'linear'.
    """

    def __init__(self, tsLength, nBands, nLayers=3, nUnits=128, dropout=0.2, activation='relu', batch_norm=False, 
                 kernel_initializer='he_normal', kernel_regularizer=0.001, classActivation='linear'):
        super(LSTMModel, self).__init__()

        self.tsLength = tsLength
        self.nBands = nBands
        self.nLayers = nLayers
        self.nUnits = nUnits
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.classActivation = classActivation
        self.input_layer = tf.keras.layers.Input((tsLength, nBands))

        self.lstm_layers = []
        for i in range(nLayers):
            self.lstm_layers.append(tf.keras.layers.LSTM(units=nUnits, return_sequences=True, kernel_initializer=kernel_initializer, 
                                                        kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer)))
            if batch_norm:
                self.lstm_layers.append(tf.keras.layers.BatchNormalization())
            
            self.lstm_layers.append(tf.keras.layers.Activation(activation))
            self.lstm_layers.append(tf.keras.layers.Dropout(dropout))

        self.lstm_layers.append(tf.keras.layers.Flatten())

        for i in range(nLayers):
            self.lstm_layers.append(tf.keras.layers.Dense(units=nUnits, kernel_initializer=kernel_initializer, 
                                                        kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer)))
            if batch_norm:
                self.lstm_layers.append(tf.keras.layers.BatchNormalization())
            
            self.lstm_layers.append(tf.keras.layers.Activation(activation))
            self.lstm_layers.append(tf.keras.layers.Dropout(dropout))
        
        self.lstm_layers.append(tf.keras.layers.Dense(units=1, kernel_initializer=kernel_initializer, 
                                                    kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer)))

        self.lstm_layers.append(tf.keras.layers.Activation(classActivation))

        net = self.input_layer
        for layer in self.lstm_layers:
            net = layer(net)

        self.model = tf.keras.Model(inputs=self.input_layer, outputs=net)

    def call(self, inputs):
        x = self.model(inputs)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'tsLength': self.tsLength,
            'nBands': self.nBands,
            'nLayers': self.nLayers,
            'nUnits': self.nUnits,
            'dropout': self.dropout,
            'activation': self.activation,
            'batch_norm': self.batch_norm,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'classActivation': self.classActivation,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def DistributedGradientBoostedTreesModel(num_trees=100, max_depth=None, task='regression', num_threads=12):
    """
    Creates a distributed gradient boosted trees model.

    Args:
        num_trees (int): The number of trees in the model. Default is 100.
        max_depth (int): The maximum depth of each tree. Default is None.
        task (str): The task type of the model, either 'regression' or 'classification'. Default is 'regression'.
        num_threads (int): The number of threads to use for training. Default is 12.

    Returns:
        model: The created gradient boosted trees model.

    """
    if task == 'regression':
        model = tfdf.keras.GradientBoostedTreesModel(
            task=tfdf.keras.Task.REGRESSION,
            num_trees=num_trees,
            max_depth=max_depth,
            num_threads=num_threads
        )
    elif task == 'classification':
        model = tfdf.keras.GradientBoostedTreesModel(
            task=tfdf.keras.Task.CLASSIFICATION,
            num_trees=num_trees,
            max_depth=max_depth,
            num_threads=num_threads
        )
    return model

def RandomForestModel(num_trees=100, max_depth=None, task='regression', num_threads=12):
    """
    Creates a random forest model using TensorFlow Decision Forests.

    Args:
        num_trees (int): The number of trees in the random forest. Default is 100.
        max_depth (int): The maximum depth of each tree. Default is None.
        task (str): The task type of the model. Can be 'regression' or 'classification'. Default is 'regression'.
        num_threads (int): The number of threads to use for training the model. Default is 12.

    Returns:
        tfdf.keras.RandomForestModel: The random forest model.

    """
    if task == 'regression':
        model = tfdf.keras.RandomForestModel(
            task=tfdf.keras.Task.REGRESSION,
            num_trees=num_trees,
            max_depth=max_depth,
            num_threads=num_threads
        )
    elif task == 'classification':
        model = tfdf.keras.RandomForestModel(
            task=tfdf.keras.Task.CLASSIFICATION,
            num_trees=num_trees,
            max_depth=max_depth,
            num_threads=num_threads
        )
    return model

class biRNNModel(tf.keras.Model):
    def __init__(self, tsLength, nBands, nLayers=3, nUnits=128, dropout=0.2, activation='relu', batch_norm=False, 
                 kernel_initializer='he_normal', kernel_regularizer=0.001, classActivation='linear'):
        """
        Initialize the bidirectional RNN model.

        Args:
            tsLength (int): The length of the time series.
            nBands (int): The number of bands in the time series.
            nLayers (int, optional): The number of RNN layers. Defaults to 3.
            nUnits (int, optional): The number of units in each RNN layer. Defaults to 128.
            dropout (float, optional): The dropout rate. Defaults to 0.2.
            activation (str, optional): The activation function to use. Defaults to 'relu'.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            kernel_initializer (str, optional): The initializer for the kernel weights. Defaults to 'he_normal'.
            kernel_regularizer (float, optional): The regularization strength for the kernel weights. Defaults to 0.001.
            classActivation (str, optional): The activation function for the output layer. Defaults to 'linear'.
        """
        super(biRNNModel, self).__init__()

        self.tsLength = tsLength
        self.nBands = nBands
        self.nLayers = nLayers
        self.nUnits = nUnits
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.classActivation = classActivation
        self.input_layer = tf.keras.layers.Input((tsLength, nBands))

        self.rnn_layers = []
        for i in range(nLayers):
            self.rnn_layers.append(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=nUnits, return_sequences=True, kernel_initializer=kernel_initializer, 
                                                        kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer))))
            if batch_norm:
                self.rnn_layers.append(tf.keras.layers.BatchNormalization())
            
            self.rnn_layers.append(tf.keras.layers.Activation(activation))
            self.rnn_layers.append(tf.keras.layers.Dropout(dropout))

        self.rnn_layers.append(tf.keras.layers.Flatten())

        for i in range(nLayers):
            self.rnn_layers.append(tf.keras.layers.Dense(units=nUnits, kernel_initializer=kernel_initializer, 
                                                        kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer)))
            if batch_norm:
                self.rnn_layers.append(tf.keras.layers.BatchNormalization())
            
            self.rnn_layers.append(tf.keras.layers.Activation(activation))
            self.rnn_layers.append(tf.keras.layers.Dropout(dropout))
        
        self.rnn_layers.append(tf.keras.layers.Dense(units=1, kernel_initializer=kernel_initializer, 
                                                    kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer)))

        self.rnn_layers.append(tf.keras.layers.Activation(classActivation))

        net = self.input_layer
        for layer in self.rnn_layers:
            net = layer(net)

        self.model = tf.keras.Model(inputs=self.input_layer, outputs=net)

    def call(self, inputs):
        x = self.model(inputs)
        return x

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'tsLength': self.tsLength,
    #         'nBands': self.nBands,
    #         'nLayers': self.nLayers,
    #         'nUnits': self.nUnits,
    #         'dropout': self.dropout,
    #         'activation': self.activation,
    #         'batch_norm': self.batch_norm,
    #         'kernel_initializer': self.kernel_initializer,
    #         'kernel_regularizer': self.kernel_regularizer,
    #         'classActivation': self.classActivation,
    #     })
    #     return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

class TempCNNModel(tf.keras.Model):
    def __init__(self, tsLength, nBands, nLayers=3, nFilters=128, kernelSize=2, dropout=0.2, activation='relu', batch_norm=False, 
                 kernel_initializer='he_normal', kernel_regularizer=0.001, classActivation='linear'):
        """
        Initialize the Temporal Convolutional Neural Network (CNN) model.

        Args:
            tsLength (int): The length of the time series.
            nBands (int): The number of bands in the time series.
            nLayers (int, optional): The number of convolutional layers. Defaults to 3.
            nFilters (int, optional): The number of filters in each convolutional layer. Defaults to 128.
            kernelSize (int, optional): The size of the convolutional kernel. Defaults to 2.
            dropout (float, optional): The dropout rate. Defaults to 0.2.
            activation (str, optional): The activation function. Defaults to 'relu'.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            kernel_initializer (str, optional): The initializer for the convolutional kernel. Defaults to 'he_normal'.
            kernel_regularizer (float, optional): The regularization parameter for the convolutional kernel. Defaults to 0.001.
            classActivation (str, optional): The activation function for the output layer. Defaults to 'linear'.
        """
        super(TempCNNModel, self).__init__()

        self.tsLength = tsLength
        self.nBands = nBands
        self.nLayers = nLayers
        self.nFilters = nFilters
        self.kernelSize = kernelSize
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.classActivation = classActivation

        self.input_layer = tf.keras.layers.Input((tsLength, nBands))

        self.conv_layers = []
        for i in range(nLayers):
            self.conv_layers.append(tf.keras.layers.Conv1D(filters=nFilters, kernel_size=kernelSize, padding='same', activation=activation, dilation_rate=2**i,
                                                        kernel_initializer=kernel_initializer, kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer)))
            if batch_norm:
                self.conv_layers.append(tf.keras.layers.BatchNormalization())

            self.conv_layers.append(tf.keras.layers.Activation(activation))
            self.conv_layers.append(tf.keras.layers.Dropout(dropout))

        self.conv_layers.append(tf.keras.layers.Flatten())

        for i in range(nLayers):
            self.conv_layers.append(tf.keras.layers.Dense(units=nFilters, kernel_initializer=kernel_initializer, 
                                                        kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer)))
            if batch_norm:
                self.conv_layers.append(tf.keras.layers.BatchNormalization())
            
            self.conv_layers.append(tf.keras.layers.Activation(activation))
            self.conv_layers.append(tf.keras.layers.Dropout(dropout))
        
        self.conv_layers.append(tf.keras.layers.Dense(units=1, kernel_initializer=kernel_initializer, 
                                                    kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer)))

        self.conv_layers.append(tf.keras.layers.Activation(classActivation))

        net = self.input_layer
        for layer in self.conv_layers:
            net = layer(net)

        self.model = tf.keras.Model(inputs=self.input_layer, outputs=net)

    def call(self, inputs):
        x = self.model(inputs)
        return x

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'tsLength': self.tsLength,
    #         'nBands': self.nBands,
    #         'nLayers': self.nLayers,
    #         'nFilters': self.nFilters,
    #         'kernelSize': self.kernelSize,
    #         'dropout': self.dropout,
    #         'activation': self.activation,
    #         'batch_norm': self.batch_norm,
    #         'kernel_initializer': self.kernel_initializer,
    #         'kernel_regularizer': self.kernel_regularizer,
    #         'classActivation': self.classActivation,
    #     })
    #     return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

class ResidualBlock(tf.keras.layers.Layer):
    """     
    A residual block layer.

    Code taken from keras-tcn implementation on available on
    https://github.com/philipperemy/keras-tcn/blob/master/tcn/tcn.py

    This class represents a residual block layer, which is commonly used in deep learning models.
    It consists of a series of convolutional layers, activation functions, and dropout layers.

    Args:
        dilation_rate (int): The dilation rate for the convolutional layers.
        nb_filters (int): The number of filters for the convolutional layers.
        kernel_size (int): The kernel size for the convolutional layers.
        padding (str): The padding mode for the convolutional layers.
        activation (str, optional): The activation function to use. Defaults to 'relu'.
        dropout_rate (float, optional): The dropout rate for the dropout layers. Defaults to 0.
        kernel_initializer (str, optional): The initializer for the kernel weights. Defaults to 'he_normal'.
        use_batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
        use_layer_norm (bool, optional): Whether to use layer normalization. Defaults to False.
        last_block (bool, optional): Whether this is the last residual block in the model. Defaults to True.

    Attributes:
        residual_layers (list): A list of the layers in the residual block.
        shape_match_conv (tf.keras.layers.Conv1D or tf.keras.layers.Lambda): The convolutional layer used for shape matching.
        res_output_shape (tuple): The output shape of the residual block.
        final_activation (tf.keras.layers.Activation): The final activation layer.

    """
    def __init__(self,
                 dilation_rate,
                 nb_filters,
                 kernel_size,
                 padding,
                 activation='relu',
                 dropout_rate=0,
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 last_block=True,
                 **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.kernel_initializer = kernel_initializer
        self.last_block = last_block
        self.residual_layers = list()
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

    def _add_and_activate_layer(self, layer):
        """Helper function for building layer.

        Args:
            layer (tf.keras.layers.Layer): The layer to be added and activated.

        """
        self.residual_layers.append(layer)
        self.residual_layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.residual_layers[-1].compute_output_shape(self.res_output_shape)

    def build(self, input_shape):
        """Builds the residual block.

        Args:
            input_shape (tuple): The shape of the input tensor.

        """
        with tf.keras.backend.name_scope(self.name):
            self.res_output_shape = input_shape

            for k in range(2):
                name = f'conv1D_{k}'
                with tf.keras.backend.name_scope(name):
                    self._add_and_activate_layer(tf.keras.layers.Conv1D(filters=self.nb_filters,
                                                                        kernel_size=self.kernel_size,
                                                                        dilation_rate=self.dilation_rate,
                                                                        padding=self.padding,
                                                                        name=name,
                                                                        kernel_initializer=self.kernel_initializer))

                if self.use_batch_norm:
                    self._add_and_activate_layer(tf.keras.layers.BatchNormalization())
                elif self.use_layer_norm:
                    self._add_and_activate_layer(tf.keras.layers.LayerNormalization())

                self._add_and_activate_layer(tf.keras.layers.Activation('relu'))
                self._add_and_activate_layer(tf.keras.layers.SpatialDropout1D(rate=self.dropout_rate))

            if not self.last_block:
                name = f'conv1D_{k+1}'
                with tf.keras.backend.name_scope(name):
                    self.shape_match_conv = tf.keras.layers.Conv1D(filters=self.nb_filters,
                                                                   kernel_size=1,
                                                                   padding='same',
                                                                   name=name,
                                                                   kernel_initializer=self.kernel_initializer)

            else:
                self.shape_match_conv = tf.keras.layers.Lambda(lambda x: x, name='identity')

            self.shape_match_conv.build(input_shape)
            self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self.final_activation = tf.keras.layers.Activation(self.activation)
            self.final_activation.build(self.res_output_shape)

            for layer in self.residual_layers:
                self.__setattr__(layer.name, layer)

            super(ResidualBlock, self).build(input_shape)

    def call(self, inputs, training=None):
        """Performs the forward pass of the residual block.

        Args:
            inputs (tf.Tensor): The input tensor.
            training (bool, optional): Whether the model is in training mode or not. Defaults to None.

        Returns:
            list: A list containing the residual model tensor and the skip connection tensor.

        """
        x = inputs
        for layer in self.residual_layers:
            if isinstance(layer, tf.keras.layers.SpatialDropout1D):
                x = layer(x, training=training)
            else:
                x = layer(x)

        x2 = self.shape_match_conv(inputs)
        res_x = tf.keras.layers.add([x2, x])
        return [self.final_activation(res_x), x]

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the residual block.

        Args:
            input_shape (tuple): The shape of the input tensor.

        Returns:
            list: A list containing the output shape of the residual block.

        """
        return [self.res_output_shape, self.res_output_shape]

class TCNModel(tf.keras.Model):
    def __init__(self, tsLength, nBands, stacks=1, nb_filters=64, kernel_size=2, dilations=[1, 2, 4, 8, 16, 32], padding='causal', activation='linear', kernel_initializer='he_normal', 
                use_batch_norm=False, use_layer_norm=False, use_skip_connections=True, return_sequences=False, dropout_rate=0.2, classActivation='linear'):
        """
        Initialize the Temporal Convolutional Neural Network (TCN) model.

        Args:
            tsLength (int): Length of the time series.
            nBands (int): Number of bands in the time series.
            stacks (int, optional): Number of TCN stacks. Defaults to 1.
            nb_filters (int, optional): Number of filters in the TCN layers. Defaults to 64.
            kernel_size (int, optional): Size of the kernel in the TCN layers. Defaults to 2.
            dilations (list, optional): List of dilation rates for the TCN layers. Defaults to [1, 2, 4, 8, 16, 32].
            padding (str, optional): Padding type for the TCN layers. Defaults to 'causal'.
            activation (str, optional): Activation function for the TCN layers. Defaults to 'linear'.
            kernel_initializer (str, optional): Kernel initializer for the TCN layers. Defaults to 'he_normal'.
            use_batch_norm (bool, optional): Whether to use batch normalization in the TCN layers. Defaults to False.
            use_layer_norm (bool, optional): Whether to use layer normalization in the TCN layers. Defaults to False.
            use_skip_connections (bool, optional): Whether to use skip connections in the TCN layers. Defaults to True.
            return_sequences (bool, optional): Whether to return sequences from the TCN layers. Defaults to False.
            dropout_rate (float, optional): Dropout rate for the TCN layers. Defaults to 0.2.
            classActivation (str, optional): Activation function for the output layer. Defaults to 'linear'.
        """
        super(TCNModel, self).__init__()

        # Add 'self.' before each attribute
        self.tsLength = tsLength
        self.nBands = nBands
        self.stacks = stacks
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_skip_connections = use_skip_connections
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.classActivation = classActivation

        inputs = tf.keras.layers.Input((tsLength, nBands))
        net = inputs
        
        blocks = []
        skip_connections = []
        total_num_blocks = stacks * len(dilations)
        if not use_skip_connections:
            total_num_blocks += 1

        net = tf.keras.layers.Conv1D(filters=nb_filters,
                                    kernel_size=1,
                                    padding=padding,
                                    kernel_initializer=kernel_initializer)(net)

        # Create the TCN layers for the network.
        for s in range(stacks):
            for d, dilation_rate in enumerate(dilations):
                is_last_block = s == stacks - 1 and d == len(dilations) - 1
                block_params = {
                    'dilation_rate': dilation_rate,
                    'nb_filters': nb_filters,
                    'kernel_size': kernel_size,
                    'padding': padding,
                    'activation': activation,
                    'dropout_rate': 0,
                    'kernel_initializer': kernel_initializer,
                    'use_batch_norm': use_batch_norm,
                    'use_layer_norm': use_layer_norm,
                    'last_block': is_last_block,
                    'name': f'stack_{s}_block_{d}'
                }
                net, skip = ResidualBlock(**block_params)(net)          
                blocks.append(net)
                skip_connections.append(skip)

        output_slice_index = int(net.shape.as_list()[1] / 2) \
            if padding.lower() == 'same' else -1
        lambda_layer = tf.keras.layers.Lambda(lambda tt: tt[:, output_slice_index, :])

        if use_skip_connections:
            net = tf.keras.layers.add(skip_connections)

        if not return_sequences:
            net = lambda_layer(net)

        net = tf.keras.layers.Dense(1)(net)

        net = tf.keras.layers.Activation(classActivation)(net)

        self.model = tf.keras.Model(inputs=inputs, outputs=net)

    def call(self, inputs):
        return self.model(inputs)

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'tsLength': self.tsLength,
    #         'nBands': self.nBands,
    #         'stacks': self.stacks,
    #         'nb_filters': self.nb_filters,
    #         'kernel_size': self.kernel_size,
    #         'dilations': self.dilations,
    #         'padding': self.padding,
    #         'activation': self.activation,
    #         'kernel_initializer': self.kernel_initializer,
    #         'use_batch_norm': self.use_batch_norm,
    #         'use_layer_norm': self.use_layer_norm,
    #         'use_skip_connections': self.use_skip_connections,
    #         'return_sequences': self.return_sequences,
    #         'dropout_rate': self.dropout_rate,
    #         'classActivation': self.classActivation,
    #     })
    #     return config

    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

# def SupportVectorMachineModel(kernel='rbf', C=1.0, gamma='scale', degree=3, coef0=0.0, tol=1e-3, max_iter=-1):
#     model = svm.SVR(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0, tol=tol, max_iter=max_iter)
#     return model

# class SupportVectorMachineModel(tf.keras.Model):
#     def __init__(self, kernel='rbf', C=1.0, gamma='scale', degree=3, coef0=0.0, tol=1e-3, max_iter=-1):
#         super(SupportVectorMachineModel, self).__init__()
#         #self.svm_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(max_tokens=None, output_mode="binary", sparse=False, pad_to_max_tokens=True)
#         self.svm = svm.SVR(kernel=kernel, C=C, gamma=gamma, degree=degree, coef0=coef0, tol=tol, max_iter=max_iter)

#     def call(self, inputs):
#         x = self.svm_layer(inputs)
#         return self.svm(x)


