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
 b. Screening all available ``.tif`` files to verify they have a sufficient number of FLAME frames (pixel dwell frames)
    for training the desired denoising model.
 c. Pre-processing available ``.tif`` files by summing the desired number of frames for CARE denoising and saving
    the processed images in ``train`` and ``test`` datasets.

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