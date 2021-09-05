import argparse
import os
import numpy as np
import nibabel as nib
import albumentations as A
from tqdm import tqdm
from pprint import pprint
from collections import OrderedDict
from tensorflow.keras.models import load_model
from segmentation_losses import (
    dice_coefficient,
    dice_loss,
    log_cosh_dice_loss,
    iou,
    dice_coefficient_binary,
    dice_loss_binary,
    log_cosh_dice_loss_binary,
    iou_binary
)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True,
                    help="(Required) Path to trained model (.h5 file)"
                    )
parser.add_argument('--model_loss', type=str, default="log_dice", choices=["dice", "log_dice"],
                    help="Segmentation loss function the segmentation model was trained with: ('dice','log_dice'), (default: 'log_dice')"
                    )
parser.add_argument('--input_directory', type=str, required=True,
                    help="(Required) Path to image folder containing the test images to generate the predictions from ('ImagesTs' folder in the original datasets')"
                    )
parser.add_argument('--output_directory', type=str, required=True,
                    help="(Required) Path to output dataset folder that will contain the model predictions"
                    )
parser.add_argument('--image_height_size', type=int, default=240,
                    help='Required image height size for model input. Default: (240)'
                    )
parser.add_argument('--image_width_size', type=int, default=240,
                    help='Required image width size for model input. Default: (240)'
                    )
parser.add_argument('--file_layer_size', type=int, default=160,
                    help='Required file layer size for model input. Default: (160)'
                    )
parser.add_argument('--num_classes', type=int, required=True,
                    help="(Required) Number of classes in dataset: (Task01_BrainTumour: 4, Task02_Heart: 2, Task03_Liver: 3, Task04_Hippocampus: 3, Task05_Prostate: 3, Task06_Lung: 2, Task07_Pancreas: 3, Task08_HepaticVessel: 3, Task09_Spleen: 2, Task10_Colon: 2)"
                    )

args = parser.parse_args()
model_path = args.model_path
model_loss = args.model_loss
input_directory = args.input_directory
output_directory = args.output_directory
image_height_size = args.image_height_size
image_width_size = args.image_width_size
file_layer_size = args.file_layer_size
num_classes = args.num_classes


def standardize(mri):
    """
    Standardize mean and standard deviation of each channel and z_dimension slice to mean 0 and standard
     deviation 1.

    Note: setting the type of the input mri to np.float16 beforehand causes issues, set it afterwards.

    Args:
        mri (np.array): input mri, shape (dim_x, dim_y, dim_z, num_channels)

    Returns:
        standardized_mri (np.array): standardized version of input mri
    """

    standardized_mri = np.zeros(mri.shape)

    # Iterate over channels
    for c in range(mri.shape[3]):
        # Iterate over the `z` depth dimension
        for z in range(mri.shape[2]):
            # Get a slice of the mri at channel c and z-th dimension
            mri_slice = mri[:, :, z, c]

            # Subtract the mean from mri_slice
            centered = mri_slice - np.mean(mri_slice)

            # Divide by the standard deviation (only if it is different from zero)
            if np.std(centered) != 0:
                centered_scaled = centered / np.std(centered)

                # Update the slice of standardized mri with the centered and scaled mri
                standardized_mri[:, :, z, c] = centered_scaled

    return standardized_mri


def generate_model_prediction(model, mri, binary_mode):
    mri_standardized = standardize(mri)
    mri_standardized = np.expand_dims(mri_standardized, axis=0)  # Keras models require an additional dimension of 'batch_size'

    prediction = model.predict(mri_standardized)
    prediction = np.squeeze(prediction, axis=0)  # Remove 'batch_size' dimension

    if binary_mode:  # binary segmentation -> sigmoid
        prediction = (prediction >= 0.5).astype(np.uint8)  # Apply "Sigmoid" function on predicted probabilities per voxel
        prediction = prediction.reshape(prediction.shape[0], prediction.shape[1], prediction.shape[2])  # Remove extra channel dimension
    else:  # multiclass segmentation -> softmax
        prediction = np.argmax(prediction, axis=3).astype(np.uint8)

    return prediction


def calculate_shape_count_dict(directory):
    file_paths = [os.path.join(directory, x) for x in os.listdir(directory)]
    shape_count_dict = {}

    for path in tqdm(file_paths):
        mri = nib.load(path).get_fdata()
        shape = mri.shape

        if shape not in shape_count_dict.keys():
            shape_count_dict[shape] = 1
        else:
            shape_count_dict[shape] += 1

    shape_count_dict = OrderedDict(sorted(shape_count_dict.items(), key=lambda item: item[1], reverse=True))

    return shape_count_dict


def generate_prediction_files_for_challenge_submission(model_path, model_loss, input_directory, output_directory,
                                                       image_height_size, image_width_size, file_layer_size,
                                                       num_classes):
    """Loops through all files in the specified input_directory, resizes the files to the specified image height and
    image width for model input, generates model predictions, resizes them to original height and width if they are
    different than the model input shape and saves them as type (np.uint8) Nifti files in the output_directory with
    the original filename.

     If the file layer size is bigger than the file layer size for model input, the file
    would be split to different chunks and the predictions would be concatenated to one file. If the file layer
    size is less than the layer size for model input the rest of the required layers would be padded with zeros and
    the slices of the original file would be extracted from the full prediction array.
    """

    transform = A.Compose(
        p=1.0,
        transforms=[
            A.Resize(
                height=image_height_size,
                width=image_width_size,
                interpolation=1,
                p=1
            )
        ]
    )

    # Load binary or multiclass model
    binary_mode = False
    if num_classes == 2:
        binary_mode = True

    if binary_mode:
        print(f"\nLoading binary segmentation model: {model_path}")
        if model_loss == "dice":
            model = load_model(
                model_path,
                custom_objects={
                    "dice_loss_binary": dice_loss_binary,
                    "dice_coefficient_binary": dice_coefficient_binary,
                    "iou_binary": iou_binary
                }
            )
        if model_loss == "log_dice":
            model = load_model(
                model_path,
                custom_objects={
                    "log_cosh_dice_loss_binary": log_cosh_dice_loss_binary,
                    "dice_coefficient_binary": dice_coefficient_binary,
                    "iou_binary": iou_binary
                }
            )
    else:
        print(f"\nLoading multiclass segmentation model: {model_path}")
        if model_loss == "dice":
            model = load_model(
                model_path,
                custom_objects={
                    "dice_loss": dice_loss,
                    "dice_coefficient": dice_coefficient,
                    "iou": iou
                }
            )
        if model_loss == "log_dice":
            model = load_model(
                model_path,
                custom_objects={
                    "log_cosh_dice_loss": log_cosh_dice_loss,
                    "dice_coefficient": dice_coefficient,
                    "iou": iou
                }
            )
    print("Model loaded.")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for (root, dirs, files) in os.walk(input_directory, topdown=True):
        if len(files) != 0:
            for filename in tqdm(files):
                filename_path = os.path.join(root, filename)

                # Set as type np.float32
                mri = nib.load(filename_path).get_fdata().astype(np.float32)

                original_mri_image_height_size = mri.shape[0]
                original_mri_image_width_size = mri.shape[1]
                original_file_layer_size = mri.shape[2]

                # 1- Resizing
                #  Case 1: If there are more than one channel, Albumentations.Resize() would only work on each
                #  channel separately, working on all dimensions at once would cause it to fail
                if len(mri.shape) >= 4:
                    temp_mri_channels = np.zeros((image_height_size, image_width_size, original_file_layer_size, mri.shape[3]))
                    for i in range(int(mri.shape[3])):
                        transformed = transform(image=mri[:, :, :, i])
                        temp_mri = transformed["image"]
                        temp_mri_channels[:, :, :, i] = temp_mri[:, :, :]

                    mri = temp_mri_channels
                else:
                    # Case 2: One channel file, Albumentations.Resize() would work normally
                    transformed = transform(image=mri)
                    mri = transformed["image"]

                # 2- Model predictions
                #   Case 1: If mri file has less layers than the desired layers, pad the rest of the desired layers
                #   with zeros
                if original_file_layer_size < file_layer_size:
                    if len(mri.shape) == 4:  # file has multiple channels
                        temp_mri = np.zeros((image_height_size, image_width_size, file_layer_size, mri.shape[3]))
                        temp_mri[:, :, :original_file_layer_size, :] = mri[:, :, :, :]
                    else:  # file has only one channel
                        temp_mri = np.zeros((image_height_size, image_width_size, file_layer_size, 1))
                        # Reshaping to format suitable for binary 3D segmentation model
                        mri_reshaped = mri.reshape((original_mri_image_height_size, original_mri_image_width_size, original_file_layer_size, 1))
                        temp_mri[:, :, :original_file_layer_size, :] = mri_reshaped[:, :, :, :]

                    mri = temp_mri
                    prediction = generate_model_prediction(model=model, mri=mri, binary_mode=binary_mode)  # Returns type np.uint8
                    # Extract the layers that coincide with the original file number of layers
                    prediction = prediction[:, :, :original_file_layer_size]

                    # Case 1.1: If original file image height or width values are different than the model input
                    #  height and width sizes, resize the prediction to original height and width
                    if (original_mri_image_height_size != image_height_size) or (original_mri_image_width_size != image_width_size):
                        transform_original_dimensions = A.Compose(
                            p=1.0,
                            transforms=[
                                A.Resize(
                                    height=original_mri_image_height_size,
                                    width=original_mri_image_width_size,
                                    interpolation=1,
                                    p=1
                                )
                            ]
                        )

                        # Make sure the prediction file is of type np.uint8, or the resizing may yield non-integer values
                        prediction = prediction.astype(np.uint8)
                        transformed = transform_original_dimensions(image=prediction)
                        prediction = transformed["image"].astype(np.uint8)

                        output_filename_path = os.path.join(output_directory, filename)
                        # Numpy array has to be set as Nifti Image object before saving
                        nib.save(nib.Nifti1Image(prediction, np.eye(4)), output_filename_path)

                    # Case 1.2: If original file image height or width are not different than the model input
                    #  then save the prediction
                    else:
                        output_filename_path = os.path.join(output_directory, filename)
                        # Numpy array has to be set as Nifti Image object before saving
                        nib.save(nib.Nifti1Image(prediction, np.eye(4)), output_filename_path)

                # Case 2: Original file has more layers than the model input layers, split the file into multiple files
                #  and concatenate the predictions
                else:
                    num_chunks = int(np.ceil(original_file_layer_size / file_layer_size))
                    prediction = np.zeros((image_height_size, image_width_size, original_file_layer_size))

                    if len(mri.shape) >= 4:  # file has multiple channels
                        chunks = np.zeros((num_chunks, image_height_size, image_width_size, file_layer_size, mri.shape[3]))
                    else:  # file has only one channel
                        chunks = np.zeros((num_chunks, image_height_size, image_width_size, file_layer_size, 1))

                    # Reshaping to format suitable for binary 3D segmentation model
                    if binary_mode:
                        mri = mri.reshape((mri.shape[0], mri.shape[1], mri.shape[2], 1))

                    # Split the file into chunks
                    for i in range(num_chunks):
                        if i == num_chunks - 1:  # last iteration (padding the last chunk with zeros and using it)
                            chunks[i, :, :, :original_file_layer_size % file_layer_size, :] = mri[:, :, i*file_layer_size: i*file_layer_size + (original_file_layer_size % file_layer_size), :]
                        else:
                            chunks[i, :, :, :, :] = mri[:, :, i*file_layer_size: (i+1)*file_layer_size, :]

                    # Generate predictions per chunk
                    for i in range(num_chunks):
                        if i == num_chunks - 1:  # Last chunk, only get the remainder of the layers
                            prediction[:, :, i*file_layer_size:] = generate_model_prediction(model=model, mri=chunks[i, :, :, :], binary_mode=binary_mode)[:, :, :original_file_layer_size - (i*file_layer_size)]  # Returns type np.uint8
                        else:
                            prediction[:, :, i*file_layer_size: (i+1)*file_layer_size] = generate_model_prediction(model=model, mri=chunks[i, :, :, :], binary_mode=binary_mode)  # Returns type np.uint8

                    # Case 2.1: If original file image height or width values are different than the model input
                    #  height and width sizes, resize the prediction to original height and width
                    if (original_mri_image_height_size != image_height_size) or (original_mri_image_width_size != image_width_size):
                        transform_original_dimensions = A.Compose(
                            p=1.0,
                            transforms=[
                                A.Resize(
                                    height=original_mri_image_height_size,
                                    width=original_mri_image_width_size,
                                    interpolation=1,
                                    p=1
                                )
                            ]
                        )

                        # Make sure the prediction file is of type np.uint8, or the resizing may yield non-integer values
                        prediction = prediction.astype(np.uint8)
                        transformed = transform_original_dimensions(image=prediction)
                        prediction = transformed["image"].astype(np.uint8)

                        output_filename_path = os.path.join(output_directory, filename)
                        # Numpy array has to be set as Nifti Image object before saving
                        nib.save(nib.Nifti1Image(prediction, np.eye(4)), output_filename_path)

                    # Case 2.2: If original file image height or width values are not different than the model input
                    # then save the prediction
                    else:
                        output_filename_path = os.path.join(output_directory, filename)
                        # Numpy array has to be set as Nifti Image object before saving
                        nib.save(nib.Nifti1Image(prediction, np.eye(4)), output_filename_path)

    # Checking file shape of input and output directories to make sure the predictions have the same dimensions
    #  as the input folder files
    print(f"\nChecking input file shapes in {input_directory} ...\n")
    shape_count_dict = calculate_shape_count_dict(directory=input_directory)
    print("\nInput file shape count dictionary:")
    pprint(shape_count_dict)

    print(f"\nChecking output file shapes in {output_directory} ...\n")
    shape_count_dict = calculate_shape_count_dict(directory=output_directory)
    print("\nOutput file shape count dictionary:")
    pprint(shape_count_dict)

    # Process end
    print("\n\n\nProcess done.")


if __name__ == '__main__':
    generate_prediction_files_for_challenge_submission(
        model_path=model_path,
        model_loss=model_loss,
        input_directory=input_directory,
        output_directory=output_directory,
        image_height_size=image_height_size,
        image_width_size=image_width_size,
        file_layer_size=file_layer_size,
        num_classes=num_classes
    )
