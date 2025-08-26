API Notes
=========

FLAMEImage
^^^^^^^^

The ``FLAMEImage`` class is designed to dynamically detect image data in the ``.tileData.txt`` that FLAME microscopes
ouptut along with images in ``.tif`` format. If the tile data isn't detected, then the class will attempt to read the
image data in an unprocessed fashion directly from the ``.tif`` file using Python's ``tifffile`` package.

That being said, some notes should be made of the ``.tileData.txt``. A few key fields will change behavior of the 
``FLAMEImage`` class.

``bidirectionalCorrection``
~~~~~~~~~

This is present to correct for a horizontal image artifact borne from line scanning
during image acquisition. This will change the final dimensions of the image. For example, if you have a bidirectional
correction set to 4 on images that are 1200x1200, the resulting image will be 1200x1196 (1200 - 4).

``nFrames``
~~~~~~~~

nFrames is a key metric in the tile data that indicates the number of accumulated frames used to genereate
the saved ``.tif`` file. "Frames" in a FLAME imaging context correlates the dwell time for each pixel. The more frames
in an image, the longer the microscope collected light data at that X-Y coordinate, resulting in higher signal-to-noise
ratio. 

**However, the number of frames listed in the tile data may not correspond to the number of frames in the
saved image**. This is because FLAME images are commonly saved such that all frame data is summed into a single channel.
In other words, you can have an iamge with ``nFrames`` being listed as 40 in the tile data, but these 40 frames **will
be saved as a single channel in the saved image**.

``FLAMEImage`` objects will automatically check whether the number of frames detected in the tile data match the number
of frames found in the corresponding ``.tif`` file. Of course, if the FLAME microscope summed the frames as discussed in
the previous paragraph, the ``FLAMEImage`` class will detect a discrepancy.

For this reason, the ``FLAMEImage`` class has workarounds at initialization to avoid checking frame counts. Namely
the ``checkFrames`` and ``overrideNFrames`` parameters.

To initialize a regular ``FLAMEImage``:

::

    this_image = FLAMEImage(
        impath = os.path.join(root, f),
        jsonext = "tileData.txt"
    )

To initialize a different ``FLAMEImage`` while overriding number of frames:

::

    this_image = FLAMEImage(
        impath = os.path.join(root, f),
        jsonext = "tileData.txt",
        overrideNFrames = 1,
        checkFrames = False,
        checkZs = True
    )


CAREInferenceEngine
~~~~~~~~~~~~~~~~~~~