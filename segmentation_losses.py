"""This module was originally imported from the reference implementation in the Week 3 of Course 1 project in the
AI for Medicine Specialization course on Coursera.

The binary segmentation losses and metric were imported from shruti-jadon's 'Semantic-Segmentation-Loss-Functions'
 GitHub repository for binary segmentation training mode:
 https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
"""

import tensorflow.keras.backend as K
import tensorflow as tf


def dice_coefficient(y_true, y_pred, axis=(0, 1, 2, 3)):
    """
    Compute mean dice coefficient over all classes.

    Args:
        y_true: Ground truth values for all classes.
                 shape: (batch_size, x_dim, y_dim, z_dim, num_classes)
        y_pred: Predictions for all classes.
                 shape: (batch_size, x_dim, y_dim, z_dim, num_classes)
        axis (tuple): Spatial axes to sum over when computing numerator and
                      denominator in formula for dice coefficient.
    Returns:
        dice_coefficient (float): Mean value of dice coefficient over all classes.
    """
    dice_numerator = 2 * K.sum(y_pred * y_true, axis=axis) + K.epsilon()
    dice_denominator = K.sum(y_pred ** 2, axis=axis) + K.sum(y_true ** 2, axis=axis) + K.epsilon()

    dice_coefficient = K.mean(dice_numerator / dice_denominator)

    return dice_coefficient


def dice_coefficient_binary(y_true, y_pred):
    """
    Compute dice coefficient (binary segmentation).

    Args:
        y_true: Ground truth values for all classes.
                 shape: (batch_size, x_dim, y_dim, z_dim)
        y_pred: Predictions for all classes.
                 shape: (batch_size, x_dim, y_dim, z_dim)

    Returns:
        dice_coefficient (float): Value of dice coefficient.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + K.epsilon()) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


def dice_loss(y_true, y_pred, axis=(0, 1, 2, 3)):
    """
    Compute mean dice loss over all classes.

    Args:
        y_true: Ground truth values for all classes.
                 shape: (batch_size, x_dim, y_dim, z_dim, num_classes)
        y_pred: Predictions for all classes.
                 shape: (batch_size, x_dim, y_dim, z_dim, num_classes)
        axis (tuple): Spatial axes to sum over when computing numerator and
                      denominator in formula for dice loss.
    Returns:
        dice_loss (float): Mean value of dice loss over all classes.
    """
    dice_loss = 1 - dice_coefficient(
        y_true=y_true,
        y_pred=y_pred,
        axis=axis
    )

    return dice_loss


def dice_loss_binary(y_true, y_pred):
    """
    Compute dice loss (binary segmentation).

    Args:
        y_true: Ground truth values for all classes.
                 shape: (batch_size, x_dim, y_dim, z_dim)
        y_pred: Predictions for all classes.
                 shape: (batch_size, x_dim, y_dim, z_dim)

    Returns:
        dice_loss (float): Value of dice loss.
    """
    loss = 1 - dice_coefficient_binary(y_true, y_pred)

    return loss


def log_cosh_dice_loss(y_true, y_pred, axis=(0, 1, 2, 3)):
    """
    Compute mean log cosh dice loss over all classes.

    Args:
        y_true: Ground truth values for all classes.
                 shape: (batch_size, x_dim, y_dim, z_dim, num_classes)
        y_pred: Predictions for all classes.
                 shape: (batch_size, x_dim, y_dim, z_dim, num_classes)
        axis (tuple): Spatial axes to sum over when computing numerator and
                      denominator in formula for dice loss.
    Returns:
        dice_loss (float): Mean value of dice loss over all classes.
    """
    x = 1 - dice_coefficient(
        y_true=y_true,
        y_pred=y_pred,
        axis=axis
    )

    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)


def log_cosh_dice_loss_binary(y_true, y_pred):
    """
    Compute log cosh dice loss (binary segmentation).

    Args:
        y_true: Ground truth values for all classes.
                 shape: (batch_size, x_dim, y_dim, z_dim)
        y_pred: Predictions for all classes.
                 shape: (batch_size, x_dim, y_dim, z_dim)

    Returns:
        log_cosh_dice_loss (float): Value of log cosh dice loss.
    """
    x = dice_loss_binary(y_true, y_pred)

    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)


def iou(y_true, y_pred, smooth=1., axis=(0, 1, 2, 3)):
    """Compute mean intersection over union (iou) over all classes.

    Args:
        y_true: Ground truth values for all classes.
                 shape: (batch_size, x_dim, y_dim, z_dim, num_classes)
        y_pred: Predictions for all classes.
                 shape: (batch_size, x_dim, y_dim, z_dim, num_classes)
        axis (tuple): Spatial axes to sum over when computing numerator and
                      denominator in formula for iou.
    Returns:
        iou (float): Mean value of iou over all classes.
    """
    intersection = K.mean(K.sum(y_true * y_pred, axis=axis))

    return (intersection + smooth) / (K.mean(K.sum(y_true, axis=axis)) + K.mean(K.sum(y_pred, axis=axis)) - intersection + smooth)


def iou_binary(y_true, y_pred, smooth=1.):
    """
    Compute intersection over union metric (binary segmentation).
    Added from: https://github.com/karolzak/keras-unet/blob/master/keras_unet/metrics.py

    Args:
        y_true: Ground truth values for all classes.
                 shape: (batch_size, x_dim, y_dim, z_dim)
        y_pred: Predictions for all classes.
                 shape: (batch_size, x_dim, y_dim, z_dim)
        smooth: Smoothing parameters. (default: 1.)

    Returns:
        iou_binary (float): Value of iou metric.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)
