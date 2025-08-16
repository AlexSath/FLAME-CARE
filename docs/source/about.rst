=====
About
=====

Dataset Workflow.
^^^^^^^^^^^^^^^^^

1. First, available images in the local file directory are provided a unique index in ``index_available_images.ipynb``. The ``FLAMEImage`` class will attempt to error-check for frames, channels, and Zs, but this has not been extensively tested. The safest course is to use images of format 'FCYX'. Image indexes can be found at:
    
    ./datasets/raw_image_index.csv

The raw image index contains a unique index for each ``.tif`` available in the local file directory while provided a relative path to that file. This way, this index should be flexible to users downloading the same images to a different local filesystem with a different root directory. 

2. Second, the ``create_care_dataset.ipynb`` notebook is run, which contains multiple functions:
    
    1. Calculation of aggregate pixel statistics for indexed images while reserving some IDs for a test dataset. Each unique image dataset (with its unique combination of image IDs) is provided its own config ``.json`` and image sample ``.png`` at:
        
        ./datasets/<date>_<n_images>I_<model_type>_<input_frames>to<GT_frames>.json
        
        ./datasets/<date>_<n_images>I_<model_type>_<input_frames>to<GT_frames>.png
    
    2. Save to the local filesystem the aggregated input and GT frames calculated from the FLAME image dataset. A local folder is created sharing the name of the ``.json`` file The output images are saved according to their unique index and frame count. Images have still NOT been normalized, and should be dim 'YXC'. Example:
        
        /.../destination_directory/train/image28_frames5.tif

        /.../destination_directory/train/image28_frames40.tif

3. Third the care dataset is preprocessed using the `CSBDeep Dataset <https://csbdeep.bioimagecomputing.com/doc/datagen.html>`_ tooling. Input images are normalized according to pixel statistics (by default, 1-99% percentiles min-max-norm) calculated in step 2.1. Then patches are extracted for training (increase patch_multiplyer to extract more patches per image) and the extracted patches are saved as a dataset ready for training in an ``.npz`` file.
     - **NOTE1**: If images in the dataset have different X-Y dimensions (such as when bidirectional scanning correction is found in ``.tileData.txt``), every image will be framed in an X-Y array of dimensions equal to the largest image X and image Ys found in the dataset. The 0 pixels padding out the frame should not generate empty patches thanks to CSBdeep's ``patch_filter`` is set to ```no_background_patches(<patch_filter>)``. `Read more <csbdeep.bioimagecomputing.com/doc/datagen.html#csbdeep.data.no_background_patches>`_.
     - **NOTE2**: Assumes that each input channel will be predicted upon separately (e.g: R, G, and B each getting inferred on separately at inference-time). This means that patch dimensions will always be (Y, X, 1) in the resulting ``.npz``.