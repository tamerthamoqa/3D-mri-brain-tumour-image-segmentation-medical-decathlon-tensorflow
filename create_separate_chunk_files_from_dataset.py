import argparse
import os
import numpy as np
import nibabel as nib
import albumentations as A
from tqdm import tqdm

# TODO: Add a special case for resizing the Task04_Hippocampus data where both image and mask files would be inserted
#  into the ALbumentations transform pipeline, as having two separate albumentations pipelines for the image and mask
#  would yield slightly distorted masks


parser = argparse.ArgumentParser()
parser.add_argument('--input_directory', type=str, required=True,
                    help="(Required) Path to dataset folder (must contain 'train/' and 'val/' subfolders with 'images' and 'masks' folders each)"
                    )
parser.add_argument('--output_directory', type=str, required=True,
                    help="(Required) Path to output dataset folder that will contain the splitted files (must contain 'train/' and 'val/' subfolders with 'images' and 'masks' folders each)"
                    )
parser.add_argument('--image_height_size', type=int, default=240,
                    help='Required image height size per file layer slice to be resized to in the output dataset image folder. Default: (240)'
                    )
parser.add_argument('--image_width_size', type=int, default=240,
                    help='Required image width size per file layer slice to be resized to in the output dataset image folder. Default: (240)'
                    )
parser.add_argument('--file_layer_size', type=int, default=160,
                    help='Required file layer size per file chunk of an original file to in the output dataset image folder. Default: (160)'
                    )
parser.add_argument('--use_last_chunk', type=bool, default=True,
                    help='Whether to use the last file chunk of the original file that would be split into "ceil(original_file_layer_size / file_layer_size) files". If True: the last chunk will be used and the rest of the layers would be padded with zeros. Default: (True)'
                    )

args = parser.parse_args()
input_directory = args.input_directory
output_directory = args.output_directory
image_height_size = args.image_height_size
image_width_size = args.image_width_size
file_layer_size = args.file_layer_size
use_last_chunk = args.use_last_chunk


def split_chunks_and_save_as_separate_files(input_directory_images, input_directory_masks, output_directory_images,
                                            output_directory_masks, transform, file_layer_size, use_last_chunk):
    # A) Image files (should be converted to type np.float32, may contain more than one channel, for resizing purposes
    # each channel would need to be resized separately).
    for (root, dirs, files) in os.walk(input_directory_images, topdown=True):
        if len(files) != 0:
            for filename in files:
                filename_path = os.path.join(root, filename)

                # Set as type np.float32
                mri = nib.load(filename_path).get_fdata().astype(np.float32)

                # 1- Resizing
                #   Case 1: If there are more than one channel, Albumentations.Resize() would only work on each
                #   channel separately, working on all dimensions at once would cause it to fail
                if len(mri.shape) >= 4:
                    temp_mri_channels = np.zeros((image_height_size, image_width_size, mri.shape[2], mri.shape[3]))
                    for i in range(int(mri.shape[3])):
                        transformed = transform(image=mri[:, :, :, i])
                        temp_mri = transformed["image"]
                        temp_mri_channels[:, :, :, i] = temp_mri[:, :, :]

                    mri = temp_mri_channels
                else:
                    # Case 2: One channel file, Albumentations.Resize() would work normally
                    transformed = transform(image=mri)
                    mri = transformed["image"]

                # 2- Splitting file to separate files based on file_layer_size
                #   Case 1: If original file has less layers than the desired layers, pad the rest of the desired layers
                #   with zeros
                if mri.shape[2] < file_layer_size:
                    if len(mri.shape) == 4:  # file has multiple channels
                        temp_mri = np.zeros((image_height_size, image_width_size, file_layer_size, mri.shape[3]))
                        temp_mri[:, :, :mri.shape[2], :] = mri[:, :, :, :]
                    else:  # file has only one channel
                        temp_mri = np.zeros((image_height_size, image_width_size, file_layer_size))
                        temp_mri[:, :, :mri.shape[2]] = mri[:, :, :]

                    mri = temp_mri

                    output_filename_path = os.path.join(output_directory_images, filename)

                    # Numpy array has to be set as Nifti Image object before saving
                    nib.save(nib.Nifti1Image(mri, np.eye(4)), output_filename_path)
                    print(output_filename_path)

                # Case 2: Original file has more layers than the desired layers
                else:
                    # Case 2.1: Use the last chunk and pad the rest of the desired length of layers with zeros
                    if use_last_chunk:
                        num_chunks = int(np.ceil(mri.shape[2] / file_layer_size))
                        chunks = np.zeros((num_chunks, mri.shape[0], mri.shape[1], file_layer_size))

                        for i in range(num_chunks):
                            if i == num_chunks - 1:  # last iteration (if we wish to pad the last chunk with zeros and use it)
                                chunks[i, :, :, :mri.shape[2] % file_layer_size] = mri[:, :, i*file_layer_size: i*file_layer_size + (mri.shape[2] % file_layer_size)]
                            else:
                                chunks[i, :, :, :] = mri[:, :, i*file_layer_size: (i+1)*file_layer_size]

                            output_filename = filename.split(".")[0] + "_" + str(i) + ".nii.gz"
                            output_filename_path = os.path.join(output_directory_images, output_filename)

                            # Numpy array has to be set as Nifti Image object before saving
                            if len(mri.shape) >= 4:
                                nib.save(nib.Nifti1Image(chunks[i, :, :, :, :], np.eye(4)), output_filename_path)
                            else:
                                nib.save(nib.Nifti1Image(chunks[i, :, :, :], np.eye(4)), output_filename_path)
                            print(output_filename_path)

                    # Case 2.2: Ignore the last chunk
                    else:
                        num_chunks = int(np.floor(mri.shape[2] / file_layer_size))
                        chunks = np.zeros((num_chunks, mri.shape[0], mri.shape[1], file_layer_size))

                        for i in range(num_chunks):
                            chunks[i, :, :, :] = mri[:, :, i*file_layer_size: (i+1)*file_layer_size]

                            output_filename = filename.split(".")[0] + "_" + str(i) + ".nii.gz"
                            output_filename_path = os.path.join(output_directory_images, output_filename)

                            # Numpy array has to be set as Nifti Image object before saving
                            if len(mri.shape) >= 4:
                                nib.save(nib.Nifti1Image(chunks[i, :, :, :, :], np.eye(4)), output_filename_path)
                            else:
                                nib.save(nib.Nifti1Image(chunks[i, :, :, :], np.eye(4)), output_filename_path)
                            print(output_filename_path)

    # B) Mask files (should be converted to type np.uint8, only one channel would exist so no need for the special
    # case of resizing each separate channel)
    for (root, dirs, files) in os.walk(input_directory_masks, topdown=True):
        if len(files) != 0:
            for filename in files:
                filename_path = os.path.join(root, filename)

                # Set as type np.uint8 (VERY IMPORTANT to avoid resizing causing non-integer values)
                mask = nib.load(filename_path).get_fdata().astype(np.uint8)

                # 1- Resizing
                # One channel file, Albumentations.Resize() would work normally
                transformed = transform(image=mask)
                mask = transformed["image"]

                # 2- Splitting file to separate files based on file_layer_size
                #   Case 1: If original file has less layers than the desired layers, pad the rest with zeros
                if mask.shape[2] < file_layer_size:
                    temp_mask = np.zeros((image_height_size, image_width_size, file_layer_size))
                    temp_mask[:, :, :mask.shape[2]] = mask[:, :, :]

                    mask = temp_mask

                    output_filename_path = os.path.join(output_directory_masks, filename)

                    # Numpy array has to be set as Nifti Image object before saving
                    nib.save(nib.Nifti1Image(mask, np.eye(4)), output_filename_path)
                    print(output_filename_path)

                # Case 2: Original file has more layers than the desired layers
                else:
                    # Case 2.1: Use the last chunk and pad the rest of the desired length of layers with zeros
                    if use_last_chunk:
                        num_chunks = int(np.ceil(mask.shape[2] / file_layer_size))
                        chunks = np.zeros((num_chunks, mask.shape[0], mask.shape[1], file_layer_size))

                        for i in range(num_chunks):
                            if i == num_chunks - 1:  # last iteration (if we wish to pad the last chunk with zeros and use it)
                                chunks[i, :, :, :mask.shape[2] % file_layer_size] = mask[:, :, i*file_layer_size: i*file_layer_size + (mask.shape[2] % file_layer_size)]
                            else:
                                chunks[i, :, :, :] = mask[:, :, i*file_layer_size: (i+1)*file_layer_size]

                            output_filename = filename.split(".")[0] + "_" + str(i) + ".nii.gz"
                            output_filename_path = os.path.join(output_directory_masks, output_filename)

                            # Numpy array has to be set as Nifti Image object before saving
                            nib.save(nib.Nifti1Image(chunks[i, :, :, :], np.eye(4)), output_filename_path)
                            print(output_filename_path)

                    # Case 2.2: Ignore the last chunk
                    else:
                        num_chunks = int(np.floor(mask.shape[2] / file_layer_size))
                        chunks = np.zeros((num_chunks, mask.shape[0], mask.shape[1], file_layer_size))

                        for i in range(num_chunks):
                            chunks[i, :, :, :] = mask[:, :, i*file_layer_size: (i+1)*file_layer_size]

                            output_filename = filename.split(".")[0] + "_" + str(i) + ".nii.gz"
                            output_filename_path = os.path.join(output_directory_masks, output_filename)

                            # Numpy array has to be set as Nifti Image object before saving
                            if len(mask.shape) >= 4:
                                nib.save(nib.Nifti1Image(chunks[i, :, :, :, :], np.eye(4)), output_filename_path)
                            else:
                                nib.save(nib.Nifti1Image(chunks[i, :, :, :], np.eye(4)), output_filename_path)
                            print(output_filename_path)


def create_separate_files_from_dataset(input_directory, output_directory, image_height_size, image_width_size,
                                       file_layer_size, use_last_chunk):
    """Loops through all files in the specified input_directory, resizes each slice to the specified image_height_size
     and image_width_size , then splits a file into file_layer_size chunks of the original file. If the original file
     has less layers than the specified file_layer_size then the original file will be padded with zeros.
      If use_last_chunk is set to True, the last chunk will be stored and saved with zero padding.
      A dictionary of shape counts in the output 'train' and 'val' directories would be computed to check if the
      resulting shapes are correct.
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

    input_train_directory_images = os.path.join(input_directory, "train", "images")
    input_train_directory_masks = os.path.join(input_directory, "train", "masks")
    input_val_directory_images = os.path.join(input_directory, "val", "images")
    input_val_directory_masks = os.path.join(input_directory, "val", "masks")

    output_train_directory_images = os.path.join(output_directory, "train", "images")
    output_train_directory_masks = os.path.join(output_directory, "train", "masks")
    output_val_directory_images = os.path.join(output_directory, "val", "images")
    output_val_directory_masks = os.path.join(output_directory, "val", "masks")

    if not os.path.exists(output_train_directory_images):
        os.makedirs(output_train_directory_images)
    if not os.path.exists(output_train_directory_masks):
        os.makedirs(output_train_directory_masks)
    if not os.path.exists(output_val_directory_images):
        os.makedirs(output_val_directory_images)
    if not os.path.exists(output_val_directory_masks):
        os.makedirs(output_val_directory_masks)

    # Going through the 'train' folders
    print("\n\nProceeding through 'train' folders ...\n")
    split_chunks_and_save_as_separate_files(
        input_directory_images=input_train_directory_images,
        input_directory_masks=input_train_directory_masks,
        output_directory_images=output_train_directory_images,
        output_directory_masks=output_train_directory_masks,
        transform=transform,
        file_layer_size=file_layer_size,
        use_last_chunk=use_last_chunk
    )
    # Going through the 'val' folders
    print("\n\nProceeding through 'val' folders: ...\n")
    split_chunks_and_save_as_separate_files(
        input_directory_images=input_val_directory_images,
        input_directory_masks=input_val_directory_masks,
        output_directory_images=output_val_directory_images,
        output_directory_masks=output_val_directory_masks,
        transform=transform,
        file_layer_size=file_layer_size,
        use_last_chunk=use_last_chunk
    )

    print("\nIterating through output 'train' and 'val' folders to compute count of the shapes of the files in those directories ...\n")
    train_mri_paths_images = [os.path.join(output_train_directory_images, x) for x in os.listdir(output_train_directory_images)]
    train_mri_paths_masks = [os.path.join(output_train_directory_masks, x) for x in os.listdir(output_train_directory_masks)]
    val_mri_paths_images = [os.path.join(output_val_directory_images, x) for x in os.listdir(output_val_directory_images)]
    val_mri_paths_masks = [os.path.join(output_val_directory_masks, x) for x in os.listdir(output_val_directory_masks)]

    # concatenate list of paths
    all_paths = train_mri_paths_images + train_mri_paths_masks + val_mri_paths_images + val_mri_paths_masks

    shape_count_dict = {}
    for path in tqdm(all_paths):
        mri = nib.load(path).get_fdata()
        shape = mri.shape

        if shape not in shape_count_dict.keys():
            shape_count_dict[shape] = 1
        else:
            shape_count_dict[shape] += 1

    print("Output file shape count dictionary:")
    print(shape_count_dict)
    print("\n\nProcess done.")


if __name__ == '__main__':
    create_separate_files_from_dataset(
        input_directory=input_directory,
        output_directory=output_directory,
        image_height_size=image_height_size,
        image_width_size=image_width_size,
        file_layer_size=file_layer_size,
        use_last_chunk=use_last_chunk
    )
