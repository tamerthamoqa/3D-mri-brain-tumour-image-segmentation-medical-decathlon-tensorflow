# This U-Net implementation is originally imported from zhixuhao's 'unet' GitHub repository and modified for
#  3D convolutions instead of 3D convolutions.
#   https://github.com/zhixuhao/unet/blob/master/model.py


from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv3D,
    MaxPooling3D,
    UpSampling3D,
    Dropout,
    Conv3DTranspose,
    BatchNormalization,
    concatenate
)


def unet_3d_upsampling_dropout(input_size=(240, 240, 144, 4), unet_resize_factor=2, unet_dropout_rate=0.3, num_classes=4,
                               binary_model=False):
    """Constructs a U-Net 3D segmentation model with Dropout layers and UpSampling3D -> Conv3D layers.

    Args:
        input_size: (tuple) Keras model input shape is  (batch_size, height, width, length, channels) with
                    'channels_last', (default: (240, 240, 144, 4)). Note: depth must be a multiple of 16.
                    Source: 'data_format' parameter documentation: https://keras.io/api/layers/convolution_layers/convolution3d/
        unet_resize_factor: (int) Resize factor of the number of filters (channels) per Convolutional layer in the U-Net
                             model (must be >= 1, such that 1 means retaining the original number of filters (channels)
                             per Convolutional layer in the U-Net model) (default: 2 (half-size))
        unet_dropout_rate: (float) Dropout rate for the Dropout layers in the U-Net model (default: 0.3).
        num_classes: (int) Number of classes in the training dataset (default: 4).
        binary_model: (boolean) If True, make the last layer have one filter with 'sigmoid' activation for a 3D binary
                        segmentation model.
    """
    inputs = Input(shape=input_size)

    # Contractive path
    conv1 = Conv3D(filters=64 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv3D(filters=64 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(filters=128 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv3D(filters=128 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(filters=256 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv3D(filters=256 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(filters=512 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv3D(filters=512 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(rate=unet_dropout_rate)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(filters=1024 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv3D(filters=1024 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(rate=unet_dropout_rate)(conv5)

    # Expansive path
    up6 = Conv3D(filters=512 // unet_resize_factor, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=4)
    conv6 = Conv3D(filters=512 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv3D(filters=512 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv3D(filters=256 // unet_resize_factor, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=4)
    conv7 = Conv3D(filters=256 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv3D(filters=256 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv3D(filters=128 // unet_resize_factor, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=4)
    conv8 = Conv3D(filters=128 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv3D(filters=128 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv3D(filters=64 // unet_resize_factor, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=4)
    conv9 = Conv3D(filters=64 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv3D(filters=64 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv3D(filters=2, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    # Final layer
    if binary_model:
        conv10 = Conv3D(filters=1, kernel_size=1, activation="sigmoid")(conv9)
    else:
        conv10 = Conv3D(filters=num_classes, kernel_size=1, activation="softmax")(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model


def unet_3d_conv3dtranspose_dropout(input_size=(240, 240, 144, 4), unet_resize_factor=2, unet_dropout_rate=0.3, num_classes=4,
                                    binary_model=False):
    """Constructs a U-Net 3D segmentation model with Dropout layers and Conv3DTranspose layers instead of
     UpSampling3D -> Conv3D layers.

    Args:
        input_size: (tuple) Keras model input shape is  (batch_size, height, width, length, channels) with
                    'channels_last', (default: (240, 240, 144, 4)). Note: depth must be a multiple of 16.
                    Source: 'data_format' parameter documentation: https://keras.io/api/layers/convolution_layers/convolution3d/
        unet_resize_factor: (int) Resize factor of the number of filters (channels) per Convolutional layer in the U-Net
                             model (must be >= 1, such that 1 means retaining the original number of filters (channels)
                             per Convolutional layer in the U-Net model) (default: 2 (half-size))
        unet_dropout_rate: (float) Dropout rate for the Dropout layers in the U-Net model (default: 0.3).
        num_classes: (int) Number of classes in the training dataset (default: 4).
        binary_model: (boolean) If True, make the last layer have one filter with 'sigmoid' activation for a 3D binary
                segmentation model.
    """
    inputs = Input(shape=input_size)

    # Contractive path
    conv1 = Conv3D(filters=64 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv3D(filters=64 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(filters=128 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv3D(filters=128 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(filters=256 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv3D(filters=256 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(filters=512 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv3D(filters=512 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(rate=unet_dropout_rate)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(filters=1024 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv3D(filters=1024 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(rate=unet_dropout_rate)(conv5)

    # Expansive path
    up6 = Conv3DTranspose(filters=512 // unet_resize_factor, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same", kernel_initializer='he_normal')(drop5)
    merge6 = concatenate([drop4, up6], axis=4)
    conv6 = Conv3D(filters=512 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv3D(filters=512 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv3DTranspose(filters=128 // unet_resize_factor, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same", kernel_initializer='he_normal')(conv6)
    merge7 = concatenate([conv3, up7], axis=4)
    conv7 = Conv3D(filters=256 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv3D(filters=256 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv3DTranspose(filters=64 // unet_resize_factor, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same", kernel_initializer='he_normal')(conv7)
    merge8 = concatenate([conv2, up8], axis=4)
    conv8 = Conv3D(filters=128 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv3D(filters=128 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv3DTranspose(filters=32 // unet_resize_factor, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same", kernel_initializer='he_normal')(conv8)
    merge9 = concatenate([conv1, up9], axis=4)
    conv9 = Conv3D(filters=64 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv3D(filters=64 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv3D(filters=2, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    # Final layer
    if binary_model:
        conv10 = Conv3D(filters=1, kernel_size=1, activation="sigmoid")(conv9)
    else:
        conv10 = Conv3D(filters=num_classes, kernel_size=1, activation="softmax")(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model


def unet_3d_upsampling_batchnormalization(input_size=(240, 240, 144, 4), unet_resize_factor=2, num_classes=4, binary_model=False):
    """Constructs a U-Net 3D segmentation model with BatchNormalization layers after each Conv3D layer instead of
     using Dropout layers in the expansive path and with using UpSampling3D -> Conv3D layers.

    Args:
        input_size: (tuple) Keras model input shape is  (batch_size, height, width, length, channels) with
                    'channels_last', (default: (240, 240, 144, 4)). Note: depth must be a multiple of 16.
                    Source: 'data_format' parameter documentation: https://keras.io/api/layers/convolution_layers/convolution3d/
        unet_resize_factor: (int) Resize factor of the number of filters (channels) per Convolutional layer in the U-Net
                             model (must be >= 1, such that 1 means retaining the original number of filters (channels)
                             per Convolutional layer in the U-Net model) (default: 2 (half-size)).
        num_classes: (int) Number of classes in the training dataset (default: 4).
        binary_model: (boolean) If True, make the last layer have one filter with 'sigmoid' activation for a 3D binary
                segmentation model.
    """
    inputs = Input(shape=input_size)

    # Contractive path
    conv1 = Conv3D(filters=64 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    bn1 = BatchNormalization()(conv1)
    conv1 = Conv3D(filters=64 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(bn1)

    conv2 = Conv3D(filters=128 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    bn2 = BatchNormalization()(conv2)
    conv2 = Conv3D(filters=128 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(bn2)

    conv3 = Conv3D(filters=256 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    bn3 = BatchNormalization()(conv3)
    conv3 = Conv3D(filters=256 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn3)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(bn3)

    conv4 = Conv3D(filters=512 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    bn4 = BatchNormalization()(conv4)
    conv4 = Conv3D(filters=512 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn4)
    bn4 = BatchNormalization()(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(bn4)

    conv5 = Conv3D(filters=1024 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    bn5 = BatchNormalization()(conv5)
    conv5 = Conv3D(filters=1024 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn5)
    bn5 = BatchNormalization()(conv5)

    # Expansive path
    up6 = Conv3D(filters=512 // unet_resize_factor, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(bn5))
    bn6 = BatchNormalization()(up6)
    merge6 = concatenate([bn4, bn6], axis=4)
    conv6 = Conv3D(filters=512 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    bn6 = BatchNormalization()(conv6)
    conv6 = Conv3D(filters=512 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn6)
    bn6 = BatchNormalization()(conv6)

    up7 = Conv3D(filters=256 // unet_resize_factor, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(bn6))
    bn7 = BatchNormalization()(up7)
    merge7 = concatenate([conv3, bn7], axis=4)
    conv7 = Conv3D(filters=256 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    bn7 = BatchNormalization()(conv7)
    conv7 = Conv3D(filters=256 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn7)
    bn7 = BatchNormalization()(conv7)

    up8 = Conv3D(filters=128 // unet_resize_factor, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(bn7))
    bn8 = BatchNormalization()(up8)
    merge8 = concatenate([conv2, bn8], axis=4)
    conv8 = Conv3D(filters=128 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    bn8 = BatchNormalization()(conv8)
    conv8 = Conv3D(filters=128 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn8)
    bn8 = BatchNormalization()(conv8)

    up9 = Conv3D(filters=64 // unet_resize_factor, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(2, 2, 2))(bn8))
    bn9 = BatchNormalization()(up9)
    merge9 = concatenate([conv1, bn9], axis=4)
    conv9 = Conv3D(filters=64 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    bn9 = BatchNormalization()(conv9)
    conv9 = Conv3D(filters=64 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn9)
    bn9 = BatchNormalization()(conv9)
    conv9 = Conv3D(filters=2, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn9)
    bn9 = BatchNormalization()(conv9)

    # Final layer
    if binary_model:
        conv10 = Conv3D(filters=1, kernel_size=1, activation="sigmoid")(bn9)
    else:
        conv10 = Conv3D(filters=num_classes, kernel_size=1, activation="softmax")(bn9)

    model = Model(inputs=inputs, outputs=conv10)

    return model


def unet_3d_conv3dtranspose_batchnormalization(input_size=(240, 240, 144, 4), unet_resize_factor=2, num_classes=4, binary_model=False):
    """Constructs a U-Net 3D segmentation model with BatchNormalization layers after each Conv3D layer instead of
     using Dropout layers in the expansive path and with using Conv3DTranspose layers instead of UpSampling3D -> Conv3D
     layers.

    Args:
        input_size: (tuple) Keras model input shape is  (batch_size, height, width, length, channels) with
                    'channels_last', (default: (240, 240, 144, 4)). Note: depth must be a multiple of 16.
                    Source: 'data_format' parameter documentation: https://keras.io/api/layers/convolution_layers/convolution3d/
        unet_resize_factor: (int) Resize factor of the number of filters (channels) per Convolutional layer in the U-Net
                             model (must be >= 1, such that 1 means retaining the original number of filters (channels)
                             per Convolutional layer in the U-Net model) (default: 2 (half-size)).
        num_classes: (int) Number of classes in the training dataset (default: 4).
        binary_model: (boolean) If True, make the last layer have one filter with 'sigmoid' activation for a 3D binary
                segmentation model.
    """
    inputs = Input(shape=input_size)

    # Contractive path
    conv1 = Conv3D(filters=64 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    bn1 = BatchNormalization()(conv1)
    conv1 = Conv3D(filters=64 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn1)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(bn1)

    conv2 = Conv3D(filters=128 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    bn2 = BatchNormalization()(conv2)
    conv2 = Conv3D(filters=128 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn2)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(bn2)

    conv3 = Conv3D(filters=256 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    bn3 = BatchNormalization()(conv3)
    conv3 = Conv3D(filters=256 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn3)
    bn3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(bn3)

    conv4 = Conv3D(filters=512 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    bn4 = BatchNormalization()(conv4)
    conv4 = Conv3D(filters=512 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn4)
    bn4 = BatchNormalization()(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(bn4)

    conv5 = Conv3D(filters=1024 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    bn5 = BatchNormalization()(conv5)
    conv5 = Conv3D(filters=1024 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn5)
    bn5 = BatchNormalization()(conv5)

    # Expansive path
    up6 = Conv3DTranspose(filters=512 // unet_resize_factor, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same", kernel_initializer='he_normal')(bn5)
    bn6 = BatchNormalization()(up6)
    merge6 = concatenate([bn4, bn6], axis=4)
    conv6 = Conv3D(filters=512 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    bn6 = BatchNormalization()(conv6)
    conv6 = Conv3D(filters=512 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn6)
    bn6 = BatchNormalization()(conv6)

    up7 = Conv3DTranspose(filters=128 // unet_resize_factor, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same", kernel_initializer='he_normal')(bn6)
    bn7 = BatchNormalization()(up7)
    merge7 = concatenate([conv3, bn7], axis=4)
    conv7 = Conv3D(filters=256 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    bn7 = BatchNormalization()(conv7)
    conv7 = Conv3D(filters=256 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn7)
    bn7 = BatchNormalization()(conv7)

    up8 = Conv3DTranspose(filters=64 // unet_resize_factor, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same", kernel_initializer='he_normal')(bn7)
    bn8 = BatchNormalization()(up8)
    merge8 = concatenate([conv2, bn8], axis=4)
    conv8 = Conv3D(filters=128 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    bn8 = BatchNormalization()(conv8)
    conv8 = Conv3D(filters=128 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn8)
    bn8 = BatchNormalization()(conv8)

    up9 = Conv3DTranspose(filters=32 // unet_resize_factor, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="same", kernel_initializer='he_normal')(bn8)
    bn9 = BatchNormalization()(up9)
    merge9 = concatenate([conv1, bn9], axis=4)
    conv9 = Conv3D(filters=64 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    bn9 = BatchNormalization()(conv9)
    conv9 = Conv3D(filters=64 // unet_resize_factor, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn9)
    bn9 = BatchNormalization()(conv9)
    conv9 = Conv3D(filters=2, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(bn9)
    bn9 = BatchNormalization()(conv9)

    # Final layer
    if binary_model:
        conv10 = Conv3D(filters=1, kernel_size=1, activation="sigmoid")(bn9)
    else:
        conv10 = Conv3D(filters=num_classes, kernel_size=1, activation="softmax")(bn9)

    model = Model(inputs=inputs, outputs=conv10)

    return model
