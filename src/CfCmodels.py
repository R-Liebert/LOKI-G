#!/usr/bin/env python3

""" 
Script is adapted from Mathias Lechner's ncps example (https://github.com/mlech26l/ncps/blob/master/examples/atari_tf.py) under apache license v2.0.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense, TimeDistributed
from tensorflow.keras.activations import softmax
from ncps.tf import CfC
from ncps.wirings import AutoNCP


class ConvNormLayer(tf.keras.layers.Layer):
    """
    A custom convolutional layer used in the Impala model for deep reinforcement learning.

    Args:
        filters (int): The dimensionality of the output space.
        kernel_size (int or tuple): The size of the convolutional kernel.
        strides (int or tuple): The stride length of the convolution.
        padding (str, optional): The padding algorithm to use. Defaults to 'valid'.
        use_bias (bool, optional): Whether or not to include a bias term. Defaults to False.

    Attributes:
        conv (tf.keras.layers.Conv2D): The convolutional layer.
        bn (tf.keras.layers.BatchNormalization): The batch normalization layer.
        relu (tf.keras.layers.ReLU): The ReLU activation layer.
    """
    def __init__(self, filters, kernel_size, strides, name, padding='valid', use_bias=False):
        super(ConvNormLayer, self).__init__()
        # Define the convolutional layer with given parameters.
        self.conv = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            name=name,
            # Use a variance scaling initializer for improved convergence.
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='truncated_normal')
        )
        # Define the batch normalization layer with momentum and epsilon parameters.
        self.bn = BatchNormalization(momentum=0.99, epsilon=0.001)
        # Define the ReLU activation layer.
        self.relu = ReLU()

    @tf.function
    def call(self, inputs):
        """
        Perform a forward pass through the ImpalaConvLayer.

        Args:
            inputs (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The output tensor.
        """
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class impalaConvLayer(tf.keras.layers.Layer):
    """
    A custom convolutional layer used in the Impala model for deep reinforcement learning.

    Args:
        filters (int): The dimensionality of the output space.
        kernel_size (int or tuple): The size of the convolutional kernel.
        strides (int or tuple): The stride length of the convolution.
        padding (str, optional): The padding algorithm to use. Defaults to 'valid'.
        use_bias (bool, optional): Whether or not to include a bias term. Defaults to False.

    Attributes:
        conv (tf.keras.layers.Conv2D): The convolutional layer.
        bn (tf.keras.layers.BatchNormalization): The batch normalization layer.
        relu (tf.keras.layers.ReLU): The ReLU activation layer.
    """
    def __init__(self, filters, kernel_size, strides, padding, **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv2D(filters, kernel_size, strides, padding=padding, name=self.name + '_conv')
        self.bn = BatchNormalization()
        self.relu = ReLU()

        self.conv1x1 = Conv2D(filters, 1, strides, padding=padding, name=self.name + '_conv1x1')  # Added 1x1 convolution

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)

        inputs_res = self.conv1x1(inputs)  # Added 1x1 convolution before residual connection

        return x + inputs_res


    
class ImpalaConvBlock(tf.keras.models.Sequential):
    """
    A custom convolutional block used in the Impala model for deep reinforcement learning.
    For the small model, filters=(16, 32, 32) and kernel_size=(8, 4, 3) are used.
    For the large model, filters=(32, 64, 128) and kernel_size=(8, 4, 3) are used.

    Attributes:
        layers (list): A list of layers comprising the convolutional block.
    """

    def __init__(self):
        super(ImpalaConvBlock, self).__init__(layers=[
            tf.keras.Input((240, 320, 3)),
            tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0),
            Conv2D(16, 8, 4, padding='same', name='conv1'),
            impalaConvLayer(filters=32, kernel_size=4, strides=2, padding='same', name='conv2'),
            impalaConvLayer(filters=32, kernel_size=3, strides=2, padding='same', name='conv3'),  # Changed strides to 2
            Flatten(),
            Dense(units=256, activation='relu')
        ])


class ConvNormBlock(tf.keras.models.Sequential):
    """
    A custom convolutional block used in the Impala model for deep reinforcement learning.
    For the small model, filters=(16, 32, 32) and kernel_size=(8, 4, 3) are used.
    For the large model, filters=(32, 64, 128) and kernel_size=(8, 4, 3) are used.

    Attributes:
        layers (list): A list of layers comprising the convolutional block.
    """
    def __init__(self):
        super(ConvNormBlock, self).__init__(layers=[
            tf.keras.Input((240, 320, 3)),
            tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0), 
            ConvNormLayer(filters=32, kernel_size=8, strides=4, padding='same', name='conv1'),
            ConvNormLayer(filters=64, kernel_size=4, strides=2, padding='same', name='conv2'),
            ConvNormLayer(filters=128, kernel_size=3, strides=1, padding='same', name='conv3'),
            Flatten(),
            Dense(units=256, activation='relu')
        ])



class ConvCfC(tf.keras.Model):
    """
    A custom model combining a convolutional neural network with a CFC (Continual Flow of Convolutionals) module
    for deep reinforcement learning.

    Attributes:
        n_actions (int): The number of possible actions in the environment.
        impala (ImpalaConvBlock): A convolutional neural network for feature extraction.
        td_impala (TimeDistributed): Time-distributed version of the ImpalaConvBlock for processing sequences of inputs.
        rnn (CfC): The CFC module.
        logits (Dense): A fully connected layer with softmax activation to produce the action probabilities.
        value_fn (Dense): A fully connected layer with linear activation to produce the state values.
        value (tf.Tensor): The state value tensor.
    """
    def __init__(self, n_actions, CfC_size=32, CNN_type='norm'):
        super().__init__()
        self.n_actions = n_actions
        self.cnn = ImpalaConvBlock() if CNN_type == 'impala' else ConvNormBlock()
        self.td_cnn = TimeDistributed(self.cnn)
        wirings = AutoNCP(CfC_size, self.n_actions)
        self.rnn = CfC(wirings, return_sequences=True, return_state=True)
        self.logits = Dense(self.n_actions, activation=softmax, name='logits')
        self.value_fn = Dense(1)
        

    def get_initial_states(self, batch_size=1):
        return self.rnn.cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

    def get_value(self):
        return self.value

    
    def call(self, x, training=True, **kwargs):
        has_hx = isinstance(x, tuple)
        initial_state = None
        if has_hx:
            # additional inputs are passed as a tuple
            x, initial_state = x
        
        if isinstance(x, tuple):
            x = x[0]
   
        x = self.td_cnn(x, training=training)
        
        x, next_state = self.rnn(x, initial_state=initial_state)
        if initial_state is not None:
            has_hx = True

        v = tf.keras.layers.Reshape((6, 1))(x)
        self.value = self.value_fn(v)
        x = self.logits(x)
        if has_hx:
            return (x, next_state)
        return x
    
