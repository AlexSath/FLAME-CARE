import os, json, logging, argparse, subprocess
from sys import argv
from pathlib import Path # Python 3.5+
from datetime import datetime

import numpy as np
import mlflow
from mlflow.client import MlflowClient
from matlab import engine as matlab_engine

from flame.image import FLAMEImage
from flame.engine import CAREInferenceSession
from flame.error import FLAMEMLFlowError, FLAMECmdError, FLAMEIOError, FLAMEPyMatlabError, CAREInferenceError
from flame.io import assert_path_exists, assert_direc_exists, assert_file_exists
from flame.utils import set_up_tracking_server, update_matlab_variables

INIT_TIME = datetime.now().strftime('%Y%m%d-%H%M%S')
LOG_DIREC = os.path.join(Path.home(), "logs")
if not os.path.isdir(LOG_DIREC): os.mkdir(LOG_DIREC)
LOGGER = logging.getLogger("MAIN")
logging.basicConfig(
    filename=os.path.join(LOG_DIREC, f"{INIT_TIME}_CARE-on-image.log"),
    encoding="utf-8",
    level=logging.DEBUG
)

def main():
    print(f"Python system logs can be found at {LOG_DIREC}")
    parser = argparse.ArgumentParser(
        prog="CARE_on_data.py",
        description= "Use CARE (content-aware image restoration) to denoise data." \
            + " Data can be a single tiff or a folder of tiffs." \
            + " Uses MLFlow (v. 2-22-2) Registered Models to track and retrieve models."
    )

    meta_group = parser.add_argument_group("Meta Parameters", description="Parameters determining how the Python inference session is run.")
    meta_group.add_argument("--matlab", action="store_true", help="Whether to coordinate with a running MATLAB/FLAME `sessionPostProcessing` thread.")
    meta_group.add_argument("--matlab_pid", required=("--matlab" in argv), help="The MATLAB process id to be used during engine linkage. Only required if '--matlab' requested.")
    meta_group.add_argument("--data-path", required=("--matlab" not in argv), help="The path to the data to infer on if '--matlab' is not requested.")
    
    model_group = parser.add_argument_group("Model Information", description="Variables to configure the name and version of the requested CARE model.")
    model_group.add_argument("--model-name", required=True, type=str, choices=["CARE-1Channel"], help="The name of the 'Registered Model' to pull from.")
    model_group.add_argument("--model-version", type=int, required=False, help="The version of the registered model to pull from. If none is provided, most recent is used.")

    mlflow_group = parser.add_argument_group("MLFlow Tracking", description="Variables to configure MLFlow tracking for model retrieval.")
    mlflow_group.add_argument("--mlflow-tracking-direc", required=True, help="Directory with 'mlruns' folder. Try '<mount>/SynologyDrive/CARE_for_MATLAB/mlruns'.")
    mlflow_group.add_argument("--mlflow-tracking-ip", required=False, default="127.0.0.1", help="IP address where to host the MLFlow server.")
    mlflow_group.add_argument("--mlflow-tracking-port", required=False, default="5050", help="Port where to run the MLFlow server.")

    args = parser.parse_args()

    """PARAMETER INPUT VALIDATION"""
    if args.data_path is not None:
        DATA_PATH_TYPE = None
        assert_path_exists(args.data_path)
        try: 
            assert_file_exists(args.data_path)
            DATA_PATH_TYPE = "file"
        except FLAMEIOError as e: DATA_PATH_TYPE = "directory"
    assert_direc_exists(args.mlflow_tracking_direc)
    
    """SET UP MLFLOW SERVER PROCESS"""
    MLFLOW_SERVER_PROCESS = set_up_tracking_server(
        ip=args.mlflow_tracking_ip,
        port=args.mlflow_tracking_port,
        direc=args.mlflow_tracking_direc,
        log_path=os.path.join(LOG_DIREC, f"{INIT_TIME}_mlflow_server.log")
    )

    """DOWNLOAD MLFLOW ARTIFACTS FOR INFERENCE"""
    try:
        TRACKING_URI = f"http://{args.mlflow_tracking_ip}:{args.mlflow_tracking_port}"
        CLIENT = MlflowClient(tracking_uri=TRACKING_URI)
        REGISTERED_MODEL = CLIENT.get_registered_model(args.model_name)

        if args.model_version is None:
            THIS_RUN_ID = REGISTERED_MODEL.latest_versions[-1].run_id
        else:
            THIS_RUN_ID = REGISTERED_MODEL.latest_versions[args.model_version - 1].run_id
    except Exception as e:
        LOGGER.error(
            f"Could not get {args.model_name} v.{args.model_version} from registered models at provided tracking server.\n" \
          + f"{e.__class__.__name__}: {e}"
        )
        raise FLAMEMLFlowError(
            f"Could not get {args.model_name} v.{args.model_version} from registered models at provided tracking server.\n" \
          + f"{e.__class__.__name__}: {e}"
        )
        
    """INITIALIZING MODEL INFERENCE ENGINE"""
    try:
        ENGINE = CAREInferenceSession.from_mlflow_uri(
            tracking_uri=TRACKING_URI,
            run_id=THIS_RUN_ID
        )
    except Exception as e:
        LOGGER.error(f"Could not initialize CAREInferenceSession from {THIS_RUN_ID}.{e.__class__.__name__}: {e}")
        raise FLAMECmdError(f"Could not initialize CAREInferenceSession from {THIS_RUN_ID}.{e.__class__.__name__}: {e}")
    
    """IF IN MATLAB MODE"""
    if args.matlab == True:
        try:
            MATLAB_ENGINE = matlab_engine.connect_matlab(f"MATLAB_{args.matlab_pid}")
            LOGGER.info(f"Connected to MATLAB engine of PID {args.matlab_pid}.")
        except Exception as e:
            LOGGER.exception(f"Could not connect to MATLAB engine of PID {args.matlab_pid}.\n{e.__class__.__name__}: {e}")
            raise FLAMEPyMatlabError(f"Could not connect to MATLAB engine of PID {args.matlab_pid}.\n{e.__class__.__name__}: {e}")
        
        # If in MATLAB mode, initialize the matlab variables.
        matlab_engine_variable_names = [
            "PYTHON_INFERENCE_ACTIVE", "PYTHON_SETUP_COMPLETE", "PYTHON_CURRENT_IMAGE"
        ]
        matlab_engine_variable_dict = {
            x: None for x in matlab_engine_variable_names
        }
        try:
            update_matlab_variables(
                matlab_eng=MATLAB_ENGINE,
                variable_dict=matlab_engine_variable_dict,
                skip_missing=False
            )
        except Exception as e:
            LOGGER.exception(f"Could not sync variables to matlab engine.\n{e.__class__.__name__}: {e}")
            raise FLAMEPyMatlabError(f"Could not sync variables to matlab engine.\n{e.__class__.__name__}: {e}")
        
        MATLAB_ENGINE["PYTHON_SETUP_COMPLETE"] = True
        
        # Setup is complete at this point, so start the main inference loop:
        while MATLAB_ENGINE["PYTHON_INFERENCE_ACTIVE"]:
            LOGGER.info(f"Python CARE inference is active...")
            if not np.isnan(MATLAB_ENGINE["PYTHON_CURRENT_IMAGE"]):
                try:
                    assert os.path.isfile(MATLAB_ENGINE["PYTHON_CURRENT_IMAGE"]), f"Provided path in 'PYTHON_CURRENT_IMAGE' must be a file."
                    LOGGER.info(f"Inferring on {MATLAB_ENGINE['PYTHON_CURRENT_IMAGE']}...")
                    MATLAB_ENGINE["PYTHON_CURRENT_IMAGE"] = np.nan
                except Exception as e:
                    LOGGER.exception(f"Could not run inference on provided path {MATLAB_ENGINE['PYTHON_CURRENT_IMAGE']}.\n{e.__class__.__name__}: {e}")
                    raise CAREInferenceError(f"Could not run inference on provided path {MATLAB_ENGINE['PYTHON_CURRENT_IMAGE']}.\n{e.__class__.__name__}: {e}")

            try: # update MATLAB variables to sync processes after every iteration of the while loop
                update_matlab_variables(
                    matlab_eng=MATLAB_ENGINE,
                    variable_dict=matlab_engine_variable_dict,
                    skip_missing=False
                )
            except Exception as e:
                LOGGER.exception(f"Could not sync variables to matlab engine.\n{e.__class__.__name__}: {e}")
                raise FLAMEPyMatlabError(f"Could not sync variables to matlab engine.\n{e.__class__.__name__}: {e}")

    else: #IF NOT IN MATLAB MODE:
        if DATA_PATH_TYPE == "file":
            pass
        elif DATA_PATH_TYPE == "directory":
            pass
        else:
            LOGGER.exception(f"Did not recognize '--data-path' of type {DATA_PATH_TYPE} ({type(args.data_path)})")
            raise FLAMECmdError(f"Did not recognize '--data-path' of type {DATA_PATH_TYPE} ({type(args.data_path)})")
    
    subprocess.Popen.kill(MLFLOW_SERVER_PROCESS)

if __name__ == "__main__":
    main()