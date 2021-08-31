# This module was originally imported from seth814's 'Semantic-Shapes' GitHub repository (also has a
#  dedicated youtube playlist), the module was heavily modified after import.
#  https://github.com/seth814/Semantic-Shapes/blob/master/data_generator.py


import tensorflow
import numpy as np
import albumentations as A
import nibabel as nib
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, mri_paths, mask_paths, mri_width, mri_height, mri_depth, batch_size=1, shuffle=True,
                 num_channels=4, augment=False, standardization=True, num_classes=4, weighted_classes=True,
                 sample_weights=None):
        self.mri_paths = mri_paths
        self.mask_paths = mask_paths
        self.mri_width = mri_width
        self.mri_height = mri_height
        self.mri_depth = mri_depth
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_channels = num_channels  # For batch generation purposes
        self.augment = augment  # Must be only true for train dataset
        if self.augment:
            self.transform = A.Compose(
                p=1.0,
                transforms=[
                    A.HorizontalFlip(p=0.5)
                ]
            )
        self.standardization = standardization  # Must be true for both train and val datasets
        self.num_classes = num_classes
        self.weighted_classes = weighted_classes  # Must be true for both train and val datasets if weighted class training is enabled
        if self.weighted_classes and sample_weights is None:
            self.sample_weights = self.calculate_class_weights()
        elif self.weighted_classes and sample_weights is not None:
            # For validation set data generator, import the sample weights of the train datagenerator as the
            #  sample_weights
            self.sample_weights = sample_weights
        self.on_epoch_end()

    def calculate_class_weights(self):
        """Used this blog post as a reference:
         https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/

        wj = n_samples / (n_classes * n_samplesj)

        wj is the weight for each class (j signifies the class)
        n_samples is the total number of samples or rows in the dataset
        n_classes is the total number of unique classes in the target
        n_samplesj is the total number of rows of the respective class
        """
        print("\nCalculating class weights from training set ...")

        n_samples_total = 0
        sample_weights = [0] * self.num_classes
        n_samples = [0] * self.num_classes

        for mask_path in tqdm(self.mask_paths):
            mask = nib.load(mask_path).get_fdata()

            n_samples_total += len(mask.flatten())  # Sum total number of voxels

            for i in range(self.num_classes):   # Sum total of class labels per class
                n_samples[i] += len(mask[mask == i])

        for i in range(self.num_classes):  # Calculate sample weights per class
            sample_weights[i] = n_samples_total / (self.num_classes * n_samples[i])

        print("\n" + str(sample_weights) + "\n")
        return sample_weights

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.mri_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.mri_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        mri_paths = [self.mri_paths[k] for k in indexes]
        mask_paths = [self.mask_paths[k] for k in indexes]

        mris, masks = self.__data_generation(mri_paths, mask_paths)

        if self.weighted_classes:
            # For class weighting, returning a third array of sample weights would automatically enable class weighted
            #  loss calculation
            return mris, masks, self.sample_weights  # Returns numpy arrays
        else:
            return mris, masks  # Returns numpy arrays

    def standardize(self, mri):
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

    def __data_generation(self, mri_paths, mask_paths):
        # Numpy array shapes are (mri_height, mri_width, num_channels)
        #  Casting masks as np.int types yields the following exception by Tensorflow so the masks are casted
        #   to np.float32 instead:
        #       TypeError: Input 'y' of 'Mul' Op has type float32 that does not match type int32 of argument 'x'
        if self.num_channels > 1:
            mris = np.empty((self.batch_size, self.mri_height, self.mri_width, self.mri_depth, self.num_channels), dtype=np.float32)
        else:
            # There has to be an extra channel dimension of one specified even for single channel mri files for
            # input to a 3D convolutional Tensorflow model
            mris = np.empty((self.batch_size, self.mri_height, self.mri_width, self.mri_depth, 1), dtype=np.float32)

        if self.num_classes > 2:  # Multiclass segmentation
            masks = np.empty((self.batch_size, self.mri_height, self.mri_width, self.mri_depth, self.num_classes),  dtype=np.float32)
        else:  # Binary segmentation
            masks = np.empty((self.batch_size, self.mri_height, self.mri_width, self.mri_depth, 1),  dtype=np.float32)

        for i, (mri_path, mask_path) in enumerate(zip(mri_paths, mask_paths)):
            mri = nib.load(mri_path).get_fdata().astype(np.float32)
            mask = nib.load(mask_path).get_fdata().astype(np.uint8)

            if self.num_classes > 2:  # Multiclass segmentation
                mask = to_categorical(mask, num_classes=self.num_classes)

            # Have the same augmentation transformation operations for the MRI nifti array and its corresponding mask
            if self.augment:
                transformed = self.transform(image=mri, mask=mask)
                mri = transformed["image"]
                mask = transformed["mask"]

            # Standardize each mri slice along the depth axis in each channel to mean 0 and standard deviation 1
            if self.standardization:
                mri = self.standardize(mri=mri)

            mris[i, ] = mri.astype(np.float32)
            masks[i, ] = mask.astype(np.float32)

        return mris, masks  # Returns numpy arrays
