import os, json, datetime, logging, argparse
from sys import argv

from ..flame import FLAMEImage, CAREInferenceSession

LOGGER = logging.getLogger("MAIN")

def main():
    parser = argparse.ArgumentParser(
        prog="CARE_on_data.py",
        description= "Use CARE (content-aware image restoration) to denoise data.\n" \
            + " - Data can be a single tiff or a folder of tiffs.\n" \
            + " - Uses MLFlow (v. 2-22-2) Registered Models to track and retrieve models.\n"
    )

    meta_group = parser.add_argument_group("Meta Parameters")
    meta_group.add_argument("--matlab", action="store_true", help="Whether to coordinate with a running MATLAB/FLAME `sessionPostProcessing` thread.")
    meta_group.add_argument("--data-path", nargs=1, required=("--matlab" not in argv), help="The path to the data to infer on if '--matlab' is not requested.")
    
    model_group = parser.add_argument_group("Model Information", description="Variables to configure the name and version of the requested CARE model.")
    model_group.add_argument("model-name", nargs=1, required=True, type=str, choices=["CARE-1Channel"], help="The name of the 'Registered Model' to pull from")
    model_group.add_argument("--model-version", nargs=1, type=int, required=False, help="The version of the registered model to pull from. If none is provided, most recent is used.")

    mlflow_group = parser.add_argument_group("MLFlow Tracking", description="Variables to configure MLFlow tracking for model retrieval.")
    mlflow_group.add_argument("mlflow-tracking-direc", nargs=1, required=True, help="Directory with 'mlruns' folder. Try '<mount>/SynologyDrive/CARE_for_MATLAB/mlruns'.")
    mlflow_group.add_argument("--mlflow-tracking-ip", nargs=1, required=False, default="127.0.0.1", help="IP address where to host the MLFlow server.")
    mlflow_group.add_argument("--mlflow-tracking-port", nargs=1, required=False, default="5050", help="Port where to run the MLFlow server")
