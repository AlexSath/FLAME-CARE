======================
Command-Line Interface
======================

Help Menu
^^^^^^^^^

  usage: CARE_on_data.py [-h] [--matlab] [--matlab_pid MATLAB_PID] --data-path DATA_PATH --model-name {CARE-1Channel} [--model-version MODEL_VERSION]
                        --mlflow-tracking-direc MLFLOW_TRACKING_DIREC [--mlflow-tracking-ip MLFLOW_TRACKING_IP] [--mlflow-tracking-port MLFLOW_TRACKING_PORT]

  Use CARE (content-aware image restoration) to denoise data. Data can be a single tiff or a folder of tiffs. Uses MLFlow (v. 2-22-2) Registered Models to
  track and retrieve models.

  options:
    -h, --help            show this help message and exit

  Meta Parameters:
    Parameters determining how the Python inference session is run.

    --matlab              Whether to coordinate with a running MATLAB/FLAME `sessionPostProcessing` thread.
    --matlab_pid MATLAB_PID
                          The MATLAB process id to be used during engine linkage. Only required if '--matlab' requested
    --data-path DATA_PATH
                          The path to the data to infer on if '--matlab' is not requested.

  Model Information:
    Variables to configure the name and version of the requested CARE model.

    --model-name {CARE-1Channel}
                          The name of the 'Registered Model' to pull from
    --model-version MODEL_VERSION
                          The version of the registered model to pull from. If none is provided, most recent is used.

  MLFlow Tracking:
    Variables to configure MLFlow tracking for model retrieval.

    --mlflow-tracking-direc MLFLOW_TRACKING_DIREC
                          Directory with 'mlruns' folder. Try '<mount>/SynologyDrive/CARE_for_MATLAB/mlruns'.
    --mlflow-tracking-ip MLFLOW_TRACKING_IP
                          IP address where to host the MLFlow server. If omitted, default is ``localhost`` (127.0.0.1)
    --mlflow-tracking-port MLFLOW_TRACKING_PORT
                          Port where to run the MLFlow server. If omitted, default is port ``5050``


Examples
^^^^^^^^

1. Using the most recent ``CARE-1Channel`` model (registered in MLFlow tracking server at ``mlflow-tracking-direc``) to infer on a folder of images.

::
  python CARE_on_data.py --data-path path/to/data/directory --model-name CARE-1Channel --mlflow-tracking-direc path/to/tracking/directory

2. Inferring on a single image using a specific version of ``CARE-1Channel`` model registered in an MLFlow tracking server hosted on a specified ip and port.

::
  python CARE_on_data.py --data-path path/to/image.tiff --model-name CARE-1Channel --model-version 2 --mlflow-tracking-direc path/to/tracking/directory --mlflow-tracking-ip 146.12.67.1 --mlflow-tracking-port 6700


