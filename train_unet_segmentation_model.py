import argparse
import os
import json
import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from datagenerator import DataGenerator
from utils import NumpyEncoder, plot_metrics
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from model import (
    unet_3d_upsampling_dropout,
    unet_3d_conv3dtranspose_dropout,
    unet_3d_upsampling_batchnormalization,
    unet_3d_conv3dtranspose_batchnormalization
)
from segmentation_losses import (
    iou,
    iou_binary,
    dice_coefficient,
    dice_coefficient_binary,
    dice_loss,
    dice_loss_binary,
    log_cosh_dice_loss,
    log_cosh_dice_loss_binary
)

parser = argparse.ArgumentParser(description="Training U-Net 3D image segmentation model.")
parser.add_argument('--train_data_dir', type=str, required=True,
                    help="(Required) Path to the train dataset folder"
                    )
parser.add_argument('--val_data_dir', type=str, required=True,
                    help="(Required) Path to the val dataset folder"
                    )
parser.add_argument('--model_architecture', default="conv3dtranspose_batchnormalization", type=str, choices=["upsampling_dropout", "conv3dtranspose_dropout", "upsampling_batchnormalization", "conv3dtranspose_batchnormalization"],
                    help="Which model architecture to build the binary 3D U-Net segmentation with: ('upsampling_dropout','conv3dtranspose_dropout','upsampling_batchnormalization','conv3dtranspose_batchnormalization'), default: 'conv3dtranspose_batchnormalization'"
                    )
parser.add_argument('--unet_resize_factor', default=2, type=int,
                    help="(integer value) Resize factor of the number of filters (channels) per Convolutional layer in the U-Net model (must be >= 1, such that 1 means retaining the original number of filters (channels) per Convolutional layer in the U-Net model) (default: 2 (half the original size))"
                    )
parser.add_argument('--unet_dropout_rate', default=0.3, type=float,
                    help="Dropout rate for the Dropout layers in the U-Net model, must be < 1 and > 0 (default: 0.3)"
                    )
parser.add_argument('--num_classes', type=int, required=True,
                    help="(Required) Number of classes in dataset: (Task01_BrainTumour: 4, Task02_Heart: 2, Task03_Liver: 3, Task04_Hippocampus: 3, Task05_Prostate: 3, Task06_Lung: 2, Task07_Pancreas: 3, Task08_HepaticVessel: 3, Task09_Spleen: 2, Task10_Colon: 2)"
                    )
parser.add_argument('--num_channels', type=int, required=True,
                    help="(Required) Number of channels in image mri file in dataset (modality): (Task01_BrainTumour: 4, Task02_Heart: 1, Task03_Liver: 1, Task04_Hippocampus: 1, Task05_Prostate: 2, Task06_Lung: 1, Task07_Pancreas: 1, Task08_HepaticVessel: 1, Task09_Spleen: 1, Task10_Colon: 1)"
                    )
parser.add_argument('--weighted_classes', default=True, type=bool,
                    help="If set to True, train model with sample weighting; the sample weights per class would be calculated from the training set by the Data Generator (default: True)"
                    )
parser.add_argument('--train_multi_gpu', default=False, type=bool,
                    help="If set to True, train model with multiple GPUs. (default: False)"
                    )
parser.add_argument('--num_gpus', default=1, type=int,
                    help="Set number of available GPUs for multi-gpu training, '--train_multi_gpu' must be also set to True  (default: 1)"
                    )
parser.add_argument('--training_epochs', default=200, type=int,
                    help="Required training epochs (default: 200)"
                    )
parser.add_argument('--model_path', default="unet_3d_segmentation_model.h5",  type=str,
                    help='Path to model checkpoint (default: "unet_3d_segmentation_model.h5")'
                    )
parser.add_argument('--resume_train', default=False, type=bool,
                    help="If set to True, resume model training from model_path (default: False)"
                    )
parser.add_argument('--loss', type=str, default="log_dice", choices=["dice", "log_dice"],
                    help="Required segmentation loss function for training the multiclass segmentation model: ('dice','log_dice'), (default: 'log_dice')"
                    )
parser.add_argument('--optimizer', type=str, default="adam", choices=["sgd", "adam", "nadam"],
                    help="Required optimizer for training the model: ('sgd','adam','nadam'), (default: 'adam')"
                    )
parser.add_argument('--lr', default=0.0001, type=float,
                    help="Learning rate for the optimizer (default: 0.0001)"
                    )
parser.add_argument('--use_nesterov_sgd', default=False, type=bool,
                    help="Use Nesterov momentum with SGD optimizer: ('True', 'False') (default: False)"
                    )
parser.add_argument('--use_amsgrad_adam', default=False, type=bool,
                    help="Use AMSGrad with adam optimizer: ('True', 'False') (default: False)"
                    )
parser.add_argument('--train_batch_size', default=1, type=int,
                    help="Batch size for train dataset datagenerator, if --train_multi_gpu then the minimum value must be the number of GPUs (default: 1)"
                    )
parser.add_argument('--val_batch_size', default=1, type=int,
                    help="Batch size for validation dataset datagenerator, if --train_multi_gpu then the minimum value must be the number of GPUs  (default: 1)"
                    )
parser.add_argument('--mri_width', default=240, type=int,
                    help="Input mri slice width (default: 240)"
                    )
parser.add_argument('--mri_height', default=240, type=int,
                    help="Input mri slice height (default: 240)"
                    )
parser.add_argument('--mri_depth', default=160, type=int,
                    help="Input mri depth, must be a multiple of 16 for the unet model (default: 160)"
                    )
parser.add_argument('--num_workers', default=4, type=int,
                    help="Number of workers for fit_generator (default: 4)"
                    )
args = parser.parse_args()


def set_tensorflow_mirrored_strategy_gpu_devices_list(num_gpus):
    gpu_devices = [""] * num_gpus

    for i in range(num_gpus):
        gpu_devices[i] = f"/gpu:{i}"  # e.g: devices=["/gpu:0", "/gpu:1"]

    return gpu_devices


def build_unet_model_architecture(model_architecture, input_size, unet_resize_factor, unet_dropout_rate, num_classes,
                                  binary_model):
    if model_architecture == "upsampling_dropout":
        model = unet_3d_upsampling_dropout(
            input_size=input_size,
            unet_resize_factor=unet_resize_factor,
            unet_dropout_rate=unet_dropout_rate,
            num_classes=num_classes,
            binary_model=binary_model
        )
    if model_architecture == "conv3dtranspose_dropout":
        model = unet_3d_conv3dtranspose_dropout(
            input_size=input_size,
            unet_resize_factor=unet_resize_factor,
            unet_dropout_rate=unet_dropout_rate,
            num_classes=num_classes,
            binary_model=binary_model
        )
    if model_architecture == "upsampling_batchnormalization":
        model = unet_3d_upsampling_batchnormalization(
            input_size=input_size,
            unet_resize_factor=unet_resize_factor,
            num_classes=num_classes,
            binary_model=binary_model
        )
    if model_architecture == "conv3dtranspose_batchnormalization":
        model = unet_3d_conv3dtranspose_batchnormalization(
            input_size=input_size,
            unet_resize_factor=unet_resize_factor,
            num_classes=num_classes,
            binary_model=binary_model
        )

    return model


def set_segmentation_loss_function(loss, binary_training):
    if loss == "dice":
        if binary_training:
            return dice_loss_binary
        else:
            return dice_loss

    if loss == "log_dice":
        if binary_training:
            return log_cosh_dice_loss_binary
        else:
            return log_cosh_dice_loss


def set_metrics(binary_training):
    if binary_training:
        return dice_coefficient_binary, iou_binary
    else:
        return dice_coefficient, iou


def set_optimizer(optimizer, learning_rate, use_nesterov_sgd, use_amsgrad_adam):
    if optimizer == "sgd":
        optimizer = optimizers.SGD(
            lr=learning_rate,
            momentum=0.9,
            nesterov=use_nesterov_sgd,
            clipvalue=50  # For weighted class training yielding exploding gradients
        )

    elif optimizer == "adam":
        # According to the notes here: https://keras.io/api/optimizers/adam/
        #  the default value of 1e-7 for epsilon might not be a good default in general
        #  for example, when training an inception network on ImageNet a current good
        #  choice is 1.0 or 0.1
        #  Since the Inception-ResNet-V2 model is being used for experiments, I decided
        #   to change epsilon from 1e-7 to 0.1
        optimizer = optimizers.Adam(
            lr=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=0.1,
            amsgrad=use_amsgrad_adam,
            clipvalue=50  # For weighted class training yielding exploding gradients
        )

    elif optimizer == "nadam":
        # According to the notes here: https://keras.io/api/optimizers/adam/
        #  the default value of 1e-7 for epsilon might not be a good default in general
        #  for example, when training an inception network on ImageNet a current good
        #  choice is 1.0 or 0.1
        #  Since the Inception-ResNet-V2 model is being used for experiments, I decided
        #   to change epsilon from 1e-7 to 0.1, I am assuming the Nesterov momentum won't
        #   be affected by the increased epsilon value
        optimizer = optimizers.Nadam(
            lr=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=0.1,
            clipvalue=50  # For weighted class training yielding exploding gradients
        )

    return optimizer


def main():
    train_data_dir = args.train_data_dir
    val_data_dir = args.val_data_dir
    model_architecture = args.model_architecture
    unet_resize_factor = args.unet_resize_factor
    unet_dropout_rate = args.unet_dropout_rate
    num_classes = args.num_classes
    num_channels = args.num_channels
    weighted_classes = args.weighted_classes
    train_multi_gpu = args.train_multi_gpu
    num_gpus = args.num_gpus
    training_epochs = args.training_epochs
    model_path = args.model_path
    resume_train = args.resume_train
    loss = args.loss
    optimizer = args.optimizer
    learning_rate = args.lr
    use_nesterov_sgd = args.use_nesterov_sgd
    use_amsgrad_adam = args.use_amsgrad_adam
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    mri_width = args.mri_width
    mri_height = args.mri_height
    mri_depth = args.mri_depth
    num_workers = args.num_workers

    # Set to binary segmentation training if num_classes=2
    binary_training = False
    if num_classes == 2:
        binary_training = True

    train_mri_paths = [os.path.join(os.path.join(train_data_dir, "images"), x) for x in os.listdir(os.path.join(train_data_dir,"images"))]
    train_mask_paths = [os.path.join(os.path.join(train_data_dir, "masks"), x) for x in os.listdir(os.path.join(train_data_dir, "masks"))]

    val_mri_paths = [os.path.join(os.path.join(val_data_dir, "images"), x) for x in os.listdir(os.path.join(val_data_dir, "images"))]
    val_mask_paths = [os.path.join(os.path.join(val_data_dir, "masks"), x) for x in os.listdir(os.path.join(val_data_dir, "masks"))]

    train_datagenerator = DataGenerator(
        mri_paths=train_mri_paths,
        mask_paths=train_mask_paths,
        mri_width=mri_width,
        mri_height=mri_height,
        mri_depth=mri_depth,
        batch_size=train_batch_size,
        shuffle=True,
        num_channels=num_channels,
        augment=True,
        standardization=True,
        num_classes=num_classes,
        weighted_classes=weighted_classes
    )

    # Set sample_weights parameter as train_datagenerator's sample_weights for validation set datagenerator
    if weighted_classes:
        val_sample_weights = train_datagenerator.sample_weights
    else:
        val_sample_weights = None

    val_datagenerator = DataGenerator(
        mri_paths=val_mri_paths,
        mask_paths=val_mask_paths,
        mri_width=mri_width,
        mri_height=mri_height,
        mri_depth=mri_depth,
        batch_size=val_batch_size,
        shuffle=False,
        num_channels=num_channels,
        augment=False,
        standardization=True,
        num_classes=num_classes,
        weighted_classes=weighted_classes,
        sample_weights=val_sample_weights
    )

    # Set GPU devices list for Tensorflow MirroredStrategy() 'devices' parameter for Multi-GPU training:
    gpu_devices = set_tensorflow_mirrored_strategy_gpu_devices_list(num_gpus=num_gpus)

    # Set segmentation loss and metrics
    loss = set_segmentation_loss_function(loss=loss, binary_training=binary_training)
    dice_coefficient, iou = set_metrics(binary_training=binary_training)

    optimizer = set_optimizer(
        optimizer=optimizer,
        learning_rate=learning_rate,
        use_nesterov_sgd=use_nesterov_sgd,
        use_amsgrad_adam=use_amsgrad_adam
    )

    # Path 1: Resume training from model heckpoint
    if resume_train:
        # Multi GPU training
        if train_multi_gpu:
            strategy = tf.distribute.MirroredStrategy(devices=gpu_devices)
            with strategy.scope():
                if loss == "dice":
                    if binary_training:
                        model = load_model(
                            model_path,
                            custom_objects={
                                "dice_loss_binary": loss,
                                "dice_coefficient_binary": dice_coefficient,
                                "iou_binary": iou
                            }
                        )
                    else:
                        model = load_model(
                            model_path,
                            custom_objects={
                                "dice_loss": loss,
                                "dice_coefficient": dice_coefficient,
                                "iou": iou
                            }
                        )

                if loss == "log_dice":
                    if binary_training:
                        model = load_model(
                            model_path,
                            custom_objects={
                                "log_cosh_dice_loss_binary": loss,
                                "dice_coefficient_binary": dice_coefficient,
                                "iou_binary": iou
                            }
                        )
                    else:
                        model = load_model(
                            model_path,
                            custom_objects={
                                "log_cosh_dice_loss": loss,
                                "dice_coefficient": dice_coefficient,
                                "iou": iou
                            }
                        )
                # https://github.com/tensorflow/tensorflow/issues/45903#issuecomment-804973541
                model.compile(
                    loss=loss,
                    optimizer=model.optimizer,
                    metrics=[dice_coefficient, iou, "accuracy"]
                )

        # Single-GPU training
        else:
            if loss == "dice":
                if binary_training:
                    model = load_model(
                        model_path,
                        custom_objects={
                            "dice_loss_binary": loss,
                            "dice_coefficient_binary": dice_coefficient,
                            "iou_binary": iou
                        }
                    )
                else:
                    model = load_model(
                        model_path,
                        custom_objects={
                            "dice_loss": loss,
                            "dice_coefficient": dice_coefficient,
                            "iou": iou
                        }
                    )

            if loss == "log_dice":
                if binary_training:
                    model = load_model(
                        model_path,
                        custom_objects={
                            "log_cosh_dice_loss_binary": loss,
                            "dice_coefficient_binary": dice_coefficient,
                            "iou_binary": iou
                        }
                    )
                else:
                    model = load_model(
                        model_path,
                        custom_objects={
                            "log_cosh_dice_loss": loss,
                            "dice_coefficient": dice_coefficient,
                            "iou": iou
                        }
                    )
            # https://github.com/tensorflow/tensorflow/issues/45903#issuecomment-804973541
            model.compile(
                loss=loss,
                optimizer=model.optimizer,
                metrics=[dice_coefficient, iou, "accuracy"]
            )

        # Change Learning Rate
        keras.backend.set_value(model.optimizer.lr, learning_rate)

    # Path 2: Train from scratch
    else:
        # Multi GPU training
        if train_multi_gpu:
            strategy = tf.distribute.MirroredStrategy(devices=gpu_devices)
            with strategy.scope():
                # Keras 3D CNN model input shape is  (batch_size, height, width, depth, channels) with 'channels_last'
                #  Source: 'data_format' parameter documentation:
                #  https://keras.io/api/layers/convolution_layers/convolution3d/
                model = build_unet_model_architecture(
                    model_architecture=model_architecture,
                    input_size=(mri_height, mri_width, mri_depth, num_channels),
                    unet_resize_factor=unet_resize_factor,
                    unet_dropout_rate=unet_dropout_rate,
                    num_classes=num_classes,
                    binary_model=binary_training
                )
                model.compile(
                    loss=loss,
                    optimizer=optimizer,
                    metrics=[dice_coefficient, iou, "accuracy"]
                )
        # Single GPU training
        else:
            # Keras 3D CNN model input shape is  (batch_size, height, width, depth, channels) with 'channels_last'
            #  Source: 'data_format' parameter documentation:
            #  https://keras.io/api/layers/convolution_layers/convolution3d/
            model = build_unet_model_architecture(
                model_architecture=model_architecture,
                input_size=(mri_height, mri_width, mri_depth, num_channels),
                unet_resize_factor=unet_resize_factor,
                unet_dropout_rate=unet_dropout_rate,
                num_classes=num_classes,
                binary_model=binary_training
            )
            model.compile(
                loss=loss,
                optimizer=optimizer,
                metrics=[dice_coefficient, iou, "accuracy"]
            )

    print(model.summary())

    if binary_training:
        print("\nTraining binary 3D U-Net {} segmentation model!".format(model_architecture))
    else:
        print("\nTraining multiclass 3D U-Net {} segmentation model!".format(model_architecture))

    if train_multi_gpu:
        print("Training on Multi-GPU mode!\n")
    else:
        print("Training on Single-GPU mode!\n")

    if binary_training:
        reducelronplateau = ReduceLROnPlateau(
            monitor="val_dice_coefficient_binary",
            factor=0.1,
            patience=20,
            verbose=1,
            mode="max",
            min_lr=1e-6
        )

        checkpoint = ModelCheckpoint(
            filepath=model_path,
            monitor='val_dice_coefficient_binary',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='max'
        )

    else:
        reducelronplateau = ReduceLROnPlateau(
            monitor="val_dice_coefficient",
            factor=0.1,
            patience=20,
            verbose=1,
            mode="max",
            min_lr=1e-6
        )

        checkpoint = ModelCheckpoint(
            filepath=model_path,
            monitor='val_dice_coefficient',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='max'
        )

    fit = model.fit(
        x=train_datagenerator,
        epochs=training_epochs,
        validation_data=val_datagenerator,
        verbose=1,
        callbacks=[reducelronplateau, checkpoint],
        workers=num_workers
    )

    # Modified to fix the 'np.float32 is not JSON serializable issue'
    dumped = json.dumps(fit.history, cls=NumpyEncoder)
    with open('model_history.txt', 'w') as f:
        json.dump(dumped, f)

    # Plot train losses and validation losses
    plot_metrics(fit.history, stop=training_epochs)


if __name__ == '__main__':
    main()
