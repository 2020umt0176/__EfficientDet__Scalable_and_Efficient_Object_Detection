import abc
from typing import Callable

import tensorflow as tf
import tensorflow_addons as tfa

def focal_loss(y_true: tf.Tensor, y_pred: tf.Tensor, gamma: float = 1.5, alpha: float = 0.25, from_logits: bool = False, reduction: str = 'sum') -> tf.Tensor:
    # Apply sigmoid function if 'from_logits' is True
    if from_logits:
        y_pred = tf.sigmoid(y_pred)
    
    epsilon = 1e-6
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_true = tf.cast(y_true, tf.float32)

    # Calculate alpha values for balancing class weights
    alpha = tf.ones_like(y_true) * alpha 
    alpha = tf.where(tf.equal(y_true, 1.), alpha, 1 - alpha)
    
    # Calculate the focal loss
    pt = tf.where(tf.equal(y_true, 1.), y_pred, 1 - y_pred)
    loss = -alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt)
    
    # Apply specified reduction method
    if reduction == 'mean':
        return tf.reduce_mean(loss)
    elif reduction == 'sum':
        return tf.reduce_sum(loss)

    return loss

def huber_loss(y_true: tf.Tensor, y_pred: tf.Tensor, clip_delta: float = 1.0, reduction: str = 'sum') -> tf.Tensor:
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    error = y_true - y_pred
    cond  = tf.abs(error) < clip_delta

    squared_loss = 0.5 * tf.square(error)
    linear_loss  = clip_delta * (tf.abs(error) - 0.5 * clip_delta)

    loss = tf.where(cond, squared_loss, linear_loss)
    loss = tf.reduce_mean(loss, axis=-1)

    # Apply specified reduction method
    if reduction == 'mean':
        return tf.reduce_mean(loss)
    elif reduction == 'sum':
        return tf.reduce_sum(loss)

    return loss

# Custom loss class for EfficientDet with Focal Loss
class EfficientDetFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha: float = 0.25, gamma: float = 1.5) -> None:
        super(EfficientDetFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # Use the Sigmoid Focal Cross-Entropy loss function from TensorFlow Addons
        self.loss_fn = tfa.losses.SigmoidFocalCrossEntropy(alpha=self.alpha, gamma=self.gamma, reduction=tf.losses.Reduction.SUM)
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Extract anchor states and labels
        anchors_states = y_true[:, :, -1]
        labels = y_true[:, :, :-1]

        not_ignore_idx = tf.where(tf.not_equal(anchors_states, -1.))
        true_idx = tf.where(tf.equal(anchors_states, 1.))

        # Calculate the normalizer for the loss
        normalizer = tf.shape(true_idx)[0]
        normalizer = tf.cast(normalizer, tf.float32)
        normalizer = tf.maximum(tf.constant(1., dtype=tf.float32), normalizer)

        y_true = tf.gather_nd(labels, not_ignore_idx)
        y_pred = tf.gather_nd(y_pred, not_ignore_idx)

        # Divide the loss by the normalizer
        return tf.divide(self.loss_fn(y_true, y_pred), normalizer)    
        
# Custom loss class for EfficientDet with Huber Loss
class EfficientDetHuberLoss(tf.keras.losses.Loss):
    def __init__(self, delta: float = 1.) -> None:
        super(EfficientDetHuberLoss, self).__init()
        self.delta = delta
        # Use the Huber loss function from TensorFlow
        self.loss_fn = tf.losses.Huber(reduction=tf.losses.Reduction.SUM, delta=self.delta)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Extract anchor states and labels
        anchors_states = y_true[:, :, -1]
        labels = y_true[:, :, :-1]

        true_idx = tf.where(tf.equal(anchors_states, 1.)

        # Calculate the normalizer for the loss
        normalizer = tf.shape(true_idx)[0]
        normalizer = tf.cast(normalizer, tf.float32)
        normalizer = tf.maximum(tf.constant(1., dtype=tf.float32), normalizer)
        normalizer = tf.multiply(normalizer, tf.constant(4., dtype=tf.float32))

        y_true = tf.gather_nd(labels, true_idx)
        y_pred = tf.gather_nd(y_pred, true_idx)

        # Multiply the loss by 50 and divide by the normalizer
        return 50. * tf.divide(self.loss_fn(y_true, y_pred), normalizer)