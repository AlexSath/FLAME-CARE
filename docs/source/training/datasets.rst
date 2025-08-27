==================
Dataset Management
==================

Dataset management in this repository happens in two steps: indexing the available images, and converting those images
into a dataset for CARE-based image denoising.

⚠️ Dataset management notebooks in FLAME-CARE expect input images in **N,Y,X,C Dimensions**. Code may not be functional 
for input images that do not match these dimensions.

1. ``create_care_dataset.ipynb`` (Creation of the Dataset):
^^^^^^^^^^^^

This process is governed by ``create_care_dataset.ipynb``, which serves many functions:
 a. Indexing available images found in the provided root directory. If an available image is not already provided a 
    unique image id in ``raw_image_index.csv``, it is provided one. Then, all available image indexes are combined 
    and associated with a dataset with a unique name and ID. Information is stored in ``<dataset_name>.json`` found 
    in the ``datasets`` directory in the FLAME-CARE source code.

        ./datasets/<date>_<n_images>I_<model_type>_<input_frames>to<GT_frames>.json
        
        ./datasets/<date>_<n_images>I_<model_type>_<input_frames>to<GT_frames>.png

 b. Screening all available ``.tif`` files to verify they have a sufficient number of FLAME frames (pixel dwell frames)
    for training the desired denoising model.
 c. Pre-processing available ``.tif`` files by summing the desired number of frames for CARE denoising and saving
    the processed images in ``train`` and ``test`` datasets.

        /.../destination_directory/train/image28_frames5.tif

        /.../destination_directory/train/image28_frames40.tif

 d. Pixel dataset statistics are aggregated. This becomes important for image normalization in the second step of
    dataset pre-processing. In image analysis, it is generally believed that images should be normalized according to
    mean pixel values across all possible images in the distribution as opposed to all pixels in the singular image of
    interest. As an example, an image of a forest may have so much green that image-level normalization may completely
    remove green contrast since at the image level green signal seems average; however, the same image normalized to a
    dataset of all images captured by smartphones reveals that this image is especially green.

⚠️ Copmutation of dataset pixel statistics relies on the assumption that all images in the dataset have the same number
of channels. In other words, it assumes that all images are either RGB, or CMYK, or Grayscale. If images with a different
number of channels are used, the code will return an error. This is registered as `issue #6 <https://github.com/AlexSath/FLAME-CARE/issues/6>`_ 
in the FLAME-CARE repository.

⚠️ If images in the dataset have different X-Y dimensions (such as when bidirectional scanning correction is found in 
``.tileData.txt``), every image will be framed in an X-Y array of dimensions equal to the largest image X and image Ys 
found in the dataset. This is common when some images in the dataset have had a ``bidirectionalCorrection`` applied.
See the "API Notes" and the ``FLAMEImage`` class for more information about this correction.


Some key parameters to understand for ``create_care_dataset.ipynb`` (found in the first code cell):
 * ``INPUT_DIREC``: This is the directory where ``.tif`` images will be searched for. Image uniqueness will be assessed
   by relative save path. Based on previous FLAME data acquisition in the Balu lab, a typical save path for an image
   might be:

        SXXX_YYMMDD_<expeirment_name>_PL/<sample_name>/<tile_acquisition_scheme>
         |-- im0001.tif
         |-- im0001.tileData.txt
         |-- im0002.tif
         |-- im0002.tileData.txt
    
 * ``OUTPUT_DIREC``: This is the directory where ``train`` and ``test`` subsets will be saved.
 * ``DATASET_DIREC``: The path to the ``datasets`` directory found within the FLAME-CARE source code, which stores the
   ``raw_image_index.csv`` as well as dataset JSON files.
 * ``DS_TYPE``: The type of the dataset being generated. No need to change since only denoising models are supported.
 * ``INPUT_N_FRAMES`` and ``OUTPUT_N_FRAMES``: The nature of a denoising model is that the input to the model will be images 
   with low signal-to-noise ratio (SNR) and ground truth images will be images with high SNR. With the FLAME imaging
   modality, SNR increases with increasing frame accumulations. Therefore, ``INPUT_N_FRAMES`` will be a low number
   whereas ``OUTPUT_N_FRAMES`` will be a higher number. Experimentally, the Balu Lab has seen success using 5 and 40
   as low and high number of frames, respectively; however, see the **Denoising Appendix** for more information about 
   what number of frames may be appropriate.


2. ``care_data_configuration.ipynb`` (Readying Dataset for Training)
^^^^^^^^^^^^^^^^

As the last step of the data pre-processing pipeline for model training, ``care_data_configuration.ipynb`` does the
following:

   a. All images are min-max normalized according to a pixel distribution. As recommended by the original CARE paper,
     1-99 percentile normalization is used.
   b. After normalization, images are split into patches.
   c. The channel dimension is then removed. This is because we (the Balu Lab) made a decision to feed image data
     one channel at a time into the CARE model. Visit the homepage or talk to Alex Vallmitjana for more information.
   d. Finally, the training data is saved as an ``.npz`` (multi-dimensional numpy data structure) along with a corresponding
     JSON containing pre-processing metadata.
   e. Save image containing example input and ground truth patches for model training.

Key parameters for this notebook (found in the first code cell):

   * ``DATASET_NAME``: Indicates which dataset should be further processed
   * ``DATASET_DIREC``: Path to the ``datasets`` directory containing ``raw_image_index.csv`` and the dataset JSON file.
   * ``INPUT_DATA_DIREC``: Path to the directory containing this dataset's ``train`` and ``test`` subsets. Should be 
     found inside the same directory as ``OUTPUT_DIREC`` in the previous Jupyter Notebook.
   * ``PATCH_SIZE``: The dimension of the square patches to be extracted from the image.
   * ``PATCH_MULTIPLE``: A scalar multiple to increase the numbe of patches extracted.
   * ``BACKGROUND_PATCH_THRESHOLD``: Briefly, this is a parameter used by the ``csbdeep`` package (default CARE package) that
     determines the amount of background signal acceptable within an extracted patch. This prevents the extraction of patches
     that don't have much signal. Read more in CSBDeep's `own documentation <csbdeep.bioimagecomputing.com/doc/datagen.html#csbdeep.data.no_background_patches>`_.
   * ``CHANNELS_ONE_BY_ONE``: Whether to remove the channel dimension from extracted patches. For all models intended for
     deployment, this should be ``True`` (see 2c. above). 

This notebook will result in an NPZ with the following naming scheme:

    <dataset_name>_patch<patch_size>_<number_patches_per_image>PpI_<number_channels_per_patch>Chan.npz


3. Denoising Appendix
^^^^^^^^^

The only way to definitively determine the appropriate number of input and output number of frames for CARE denoising
is to do so empirically. However, any ML Denoising Investigator should consider the following when making a decision 
regarding the number of frames to include in input and ground-truth:

* Input number of frames determines the acquisition time required for your denoising model, and therefore dictates
  the magnitude of acquisition speed increase provided by CARE processing.
* The delta between the input and output number of frames determines the scale of the SNR gap the trained model is
  being asked to recreate. The higher the SNR gap, the more information the trained model has to "invent" during
  inference and the higher the chance for hallucination. `This blog <https://blog.yanlincs.com/ml-tech/one-step-diffusion-models>`_
  may be useful for more information.
* If a high delta between the input and output number of frames is required, an enterprizing ML scientist may seek
  to split the SNR gap into multiple steps, thereby performing gradual, step-wise denoising. This is a great idea, and
  it is the basis for `Stable Diffusion Models <https://blog.segmind.com/beginners-guide-to-stable-diffusion-steps-parameter/>`_.
  Many-step denoising is not currently supported by this codebase, however.