import math
import tensorflow as tf

from . import cnn_layers
from utils import cascade_layers

class RetinaNetBBPredictor(tf.keras.Model):
    def __init__(self, width: int, depth: int, num_anchors: int = 9, prefix: str = ''):
        super(RetinaNetBBPredictor, self).__init__()
        self.num_anchors = num_anchors

        self.feature_extractors = [cnn_layers.ConvBlock(width, kernel_size=3, activation='swish', padding='same', separable_conv=True, pointwise_initializer=tf.initializers.VarianceScaling(), depthwise_initializer=tf.initializers.VarianceScaling(), layer_prefix=prefix + f'conv_block_{i}/')
            for i in range(depth)]

        self.bb_regressor = tf.keras.layers.SeparableConv2D(num_anchors * 4, depth_multiplier=1, pointwise_initializer=tf.initializers.VarianceScaling(), depthwise_initializer=tf.initializers.VarianceScaling(), kernel_size=3, padding='same', name=prefix + 'regress_conv')

    def call(self, features: tf.Tensor, training: bool = True) -> tf.Tensor:
        batch_size = tf.shape(features)[0]

        x = cascade_layers.cascade_layers(self.feature_extractors, features, training=training)
        return tf.reshape(self.bb_regressor(x), [batch_size, -1, 4])


class RetinaNetClassifier(tf.keras.Model):
    def __init__(self, width: int, depth: int, num_classes: int, num_anchors: int = 9, prefix: str = ''):
        super(RetinaNetClassifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.feature_extractors = [
            cnn_layers.ConvBlock(width, kernel_size=3, activation='swish', padding='same', layer_prefix=prefix + f'conv_block_{i}/', separable_conv=True, pointwise_initializer=tf.initializers.VarianceScaling(), depthwise_initializer=tf.initializers.VarianceScaling(), bias_initializer='zeros')
            for i in range(depth)]
        
        prob = 0.01
        w_init = tf.constant_initializer(-math.log((1 - prob) / prob))
        self.cls_score = tf.keras.layers.SeparableConv2D(num_anchors * num_classes, kernel_size=3, depth_multiplier=1, pointwise_initializer=tf.initializers.VarianceScaling(), depthwise_initializer=tf.initializers.VarianceScaling(), activation='sigmoid', padding='same', bias_initializer=w_init, name=prefix + 'clf_conv')

    def call(self, features: tf.Tensor, training: bool = True) -> tf.Tensor:
        batch_size = tf.shape(features)[0]

        x = cascade_layers.cascade_layers(self.feature_extractors, features, training=training)
        return tf.reshape(self.cls_score(x), [batch_size, -1, self.num_classes])
