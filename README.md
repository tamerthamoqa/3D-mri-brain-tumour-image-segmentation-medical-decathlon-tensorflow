# 3D-image-segmentation-medical-decathlon-tensorflow
Code repository for training U-Net 3D image segmentation models using the [medical segmentation decathlon challenge](https://decathlon-10.grand-challenge.org/) datasets.

This is __not meant as an official submission__ for the challenge since the challenge proposition describes that "teams cannot __manually__ tweak parameters of algorithms/models
on a task specific basis. Any parameter tuning has to happen automatically and algorithmically. As an example, the learning rate or the depth of a network cannot be manually
changed between tasks, but they can be found automatically through cross-validation. Any team which is found to use different human-defined and task-specific parameters will
be terminated".

Although I have explored the datasets' height, width, and depth statistics per each task in the 'misc/datasets_exploration_and_visualization' folder; I have manually set the model
input shapes as according to my judgement since I was also testing how much I could squeeze inside the TITAN RTX GPU's 24 gigabytes of Video RAM memory for the purposes of 3D Image
Segmentation.


### Medical Segmentation Decathlon datasets download links:
* [Official Google Drive link](https://goo.gl/QzVZcm)
* [Academic Torrents](https://academictorrents.com/details/274be65156ed14828fb7b30b82407a2417e1924a)


### Training segmentation model

```
    usage: train_unet_segmentation_model.py [-h] --train_data_dir TRAIN_DATA_DIR
                                            --val_data_dir VAL_DATA_DIR
                                            [--model_architecture {upsampling_dropout,conv3dtranspose_dropout,upsampling_batchnormalization,conv3dtranspose_batchnormalization}]
                                            [--unet_resize_factor UNET_RESIZE_FACTOR]
                                            [--unet_dropout_rate UNET_DROPOUT_RATE]
                                            --num_classes NUM_CLASSES
                                            --num_channels NUM_CHANNELS
                                            [--weighted_classes WEIGHTED_CLASSES]
                                            [--train_multi_gpu TRAIN_MULTI_GPU]
                                            [--num_gpus NUM_GPUS]
                                            [--training_epochs TRAINING_EPOCHS]
                                            [--model_path MODEL_PATH]
                                            [--resume_train RESUME_TRAIN]
                                            [--loss {dice,log_dice}]
                                            [--optimizer {sgd,adam,nadam}]
                                            [--lr LR]
                                            [--use_nesterov_sgd USE_NESTEROV_SGD]
                                            [--use_amsgrad_adam USE_AMSGRAD_ADAM]
                                            [--train_batch_size TRAIN_BATCH_SIZE]
                                            [--val_batch_size VAL_BATCH_SIZE]
                                            [--mri_width MRI_WIDTH]
                                            [--mri_height MRI_HEIGHT]
                                            [--mri_depth MRI_DEPTH]
                                            [--num_workers NUM_WORKERS]
    
    Training U-Net 3D image segmentation model.
    
    optional arguments:
      -h, --help            show this help message and exit
      --train_data_dir TRAIN_DATA_DIR
                            (Required) Path to the train dataset folder
      --val_data_dir VAL_DATA_DIR
                            (Required) Path to the val dataset folder
      --model_architecture {upsampling_dropout,conv3dtranspose_dropout,upsampling_batchnormalization,conv3dtranspose_batchnormalization}
                            Which model architecture to build the binary 3D U-Net
                            segmentation with: ('upsampling_dropout','conv3dtransp
                            ose_dropout','upsampling_batchnormalization','conv3dtr
                            anspose_batchnormalization'), default:
                            'conv3dtranspose_batchnormalization'
      --unet_resize_factor UNET_RESIZE_FACTOR
                            (integer value) Resize factor of the number of filters
                            (channels) per Convolutional layer in the U-Net model
                            (must be >= 1, such that 1 means retaining the
                            original number of filters (channels) per
                            Convolutional layer in the U-Net model) (default: 2
                            (half the original size))
      --unet_dropout_rate UNET_DROPOUT_RATE
                            Dropout rate for the Dropout layers in the U-Net
                            model, must be < 1 and > 0 (default: 0.3)
      --num_classes NUM_CLASSES
                            (Required) Number of classes in dataset:
                            (Task01_BrainTumour: 4, Task02_Heart: 2, Task03_Liver:
                            3, Task04_Hippocampus: 3, Task05_Prostate: 3,
                            Task06_Lung: 2, Task07_Pancreas: 3,
                            Task08_HepaticVessel: 3, Task09_Spleen: 2,
                            Task10_Colon: 2)
      --num_channels NUM_CHANNELS
                            (Required) Number of channels in image mri file in
                            dataset (modality): (Task01_BrainTumour: 4,
                            Task02_Heart: 1, Task03_Liver: 1, Task04_Hippocampus:
                            1, Task05_Prostate: 2, Task06_Lung: 1,
                            Task07_Pancreas: 1, Task08_HepaticVessel: 1,
                            Task09_Spleen: 1, Task10_Colon: 1)
      --weighted_classes WEIGHTED_CLASSES
                            If set to True, train model with sample weighting; the
                            sample weights per class would be calculated from the
                            training set by the Data Generator (default: True)
      --train_multi_gpu TRAIN_MULTI_GPU
                            If set to True, train model with multiple GPUs.
                            (default: False)
      --num_gpus NUM_GPUS   Set number of available GPUs for multi-gpu training, '
                            --train_multi_gpu' must be also set to True (default:
                            1)
      --training_epochs TRAINING_EPOCHS
                            Required training epochs (default: 200)
      --model_path MODEL_PATH
                            Path to model checkpoint (default:
                            "unet_3d_segmentation_model.h5")
      --resume_train RESUME_TRAIN
                            If set to True, resume model training from model_path
                            (default: False)
      --loss {dice,log_dice}
                            Required segmentation loss function for training the
                            multiclass segmentation model: ('dice','log_dice'),
                            (default: 'log_dice')
      --optimizer {sgd,adam,nadam}
                            Required optimizer for training the model:
                            ('sgd','adam','nadam'), (default: 'adam')
      --lr LR               Learning rate for the optimizer (default: 0.0001)
      --use_nesterov_sgd USE_NESTEROV_SGD
                            Use Nesterov momentum with SGD optimizer: ('True',
                            'False') (default: False)
      --use_amsgrad_adam USE_AMSGRAD_ADAM
                            Use AMSGrad with adam optimizer: ('True', 'False')
                            (default: False)
      --train_batch_size TRAIN_BATCH_SIZE
                            Batch size for train dataset datagenerator(default: 1)
      --val_batch_size VAL_BATCH_SIZE
                            Batch size for val dataset datagenerator (default: 1)
      --mri_width MRI_WIDTH
                            Input mri slice width (default: 240)
      --mri_height MRI_HEIGHT
                            Input mri slice height (default: 240)
      --mri_depth MRI_DEPTH
                            Input mri depth, must be a multiple of 16 for the unet
                            model (default: 160)
      --num_workers NUM_WORKERS
                            Number of workers for fit_generator (default: 4)
```

### Useful tools:
* 3D MRI Scan viewer (supports nifti files): [link](https://socr.umich.edu/HTML5/BrainViewer/)

### References:

* Simpson, A., Antonelli, M., Bakas, S., Bilello, M., Farahani, K., Ginneken, B., Kopp-Schneider, A., Landman, B., Litjens, G., Menze, B., Ronneberger, O., Summers, R., Bilic, P., Christ, P., Do, R., Gollub, M., Golia-Pernicka, J., Heckers, S., Jarnagin, W., McHugo, M., Napel, S., Vorontsov, E., Maier-Hein, L., & Cardoso, M.J. (2019). A large annotated medical image dataset for the development and evaluation of segmentation algorithms. ArXiv, abs/1902.09063. [link](https://arxiv.org/abs/1902.09063)

### Hardware Specifications
* TITAN RTX Graphics Card (24 gigabytes Video RAM).
* i9-9900KF Intel CPU overclocked to 5 GHz.
* 32 Gigabytes DDR4 RAM at 3200 MHz.
