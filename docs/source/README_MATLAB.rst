====================================================
Use this readme if running inference through MATLAB.
====================================================

The following instructions should be used if you would like to run CARE inference for FLAME images through
MATLAB.

There are two possible implementations. Both require the ``CARE_on_image.py`` script. The focus of this README
will be the implementation with MATLAB integration. This integration **makes inference faster** when needing
to infer on many images sequentially through MATLAB. It accomplishes this by loading the inference engine
before it is needed and keeping it loaded into memory (as opposed to re-initializing it every time it is 
needed). While this is an unusual implementation, it is made possible by MATLAB having built-in tools
to communicate with Python processes through variables in the MATLAB workspace. In this case, only file
paths to images are shared betwythoneen the Python process and the MATLAB process. In other words, the logic and flow
of information between two processes is as follows:
#. the MATLAB process will save an image to the disk and provided the savepath to the Python process.
#. Python will use an ONNX pre-loaded from MLFlow during Python process initialization to infer on data from the provided path.
#. Python will overwrite the data path with the post-inference data
#. MATLAB will load data (now processed / inferred upon) from the same path.

**IMPORTANT:** Be sure to confirm the compatibility of your MATLAB with the python used in the conda environment
for this repository. If using a default setup, the repository uses Python 3.12, which as of July 2025 requires
MATLAB 2024b or newer.

Read more: `MATLAB Python Version Compatibility <https://www.mathworks.com/support/requirements/python-compatibility.html`_

To understand how Python Connects to the MATLAB Engine more generally:
 - `MATLAB Engine Documentation <https://www.mathworks.com/help/matlab/matlab_external/connect-python-to-running-matlab-session.html>`_
 - `Python and MATLAB Dictionary Interoperability <https://www.mathworks.com/help/matlab/matlab_external/use-matlab-dictionaries-in-python.html>`_
 - `Python and MATLAB Table Interoperability (+ Numpy DTypes) <https://www.mathworks.com/help/matlab/matlab_external/use-matlab-tables-and-timetables-in-python.html>`_



1. Setting up MATLAB Engine (Workspace) Variables
#################################################

For coordination with the MATLAB process, the Python process will expect certain variables to be present in
the MATLAB workspace::

    PYTHON_INFERENCE_ACTIVE = true % placeholder variable for when it should remain active
    PYTHON_SETUP_COMPLETE = false % placeholder variable for when the python setup is complete, and the python process is ready to be fed images.
    PYTHON_CURRENT_IMAGE = NaN % Set CURRENT_IMAGE to NaN when there is no image to be run
    while ~PYTHON_SETUP_COMPLETE % wait until the python setup is complete
    end


2. Initialize Script w/ MATLAB Integration
##########################################

Once the variables are initialized as in step 1, the next step is to initialize the python process.

This will open a new terminal window where the python process is housed. While some output will
be outputted to this window, **most relevant outputs for the python script will be saved in the log directory in the user's home directory**.

**OPTION 1 (Recommended):**::

    proc_id = feature('getpid')
    system('conda activate flame')
    system(['python ' path_to_script ' --matlab --matlab_pid ' proc_id ' <args>'])


**OPTION 2:**
Run *CARE_on_image.py* as normal, but with the *--matlab* command::

    (<conda_env>) user@computer:/path/to/repository$ python CARE_on_image.py --matlab --matlab_pid proc_id <args>


3. Sequential Image Inference
#############################

In whatever master loop is handling your image, you should infer on your images in the following way::

    PYTHON_CURRENT_IMAGE = path_to_image
    while ~isnan(PYTHON_CURRENT_IMAGE) % when python is done inferring on the image, it will set PYTHON_CURRENT_IMAGE back to NaN.
    end

Note that the Python process can handle images that have been saved in raw flame format, as long as there is a
`tileData.txt` in the same folder with the same name.


4. Ending the Python Process
############################

When inference is done inside of the MATLAB process, it is easy to terminate the running Python process::

    PYTHON_INFERENCE_ACTIVE = false % sends message to Python process to exit


