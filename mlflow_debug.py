# %%
import os
from datetime import datetime
import json
import logging

from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from csbdeep.utils import axes_dict, plot_history, plot_some

from matplotlib import pyplot as plt
import tensorflow as tf
import tf2onnx
import onnx
import numpy as np

import mlflow
from mlflow.models import infer_signature
from mlflow.pyfunc import log_model
from mlflow.onnx import log_model as onnx_log_model

from flame.mlflow_pyfunc import MLFLOW_CARE_Model
from flame.error import CAREInferenceError

# %%
tf.config.run_functions_eagerly(False)

# %%
mlflow.set_tracking_uri(uri="http://127.0.0.1:5050")
EXPERIMENT_NAME = "CARE Denoising 1 Channel"
mlflow.set_experiment(EXPERIMENT_NAME)

# %%
DATA_DIREC = "/mnt/d/data/processed/20250527_112I_denoising_5to40F"
PATCH_CONFIG_JSON = os.path.join(DATA_DIREC, "patch_config.json")
SAVE_DIREC = "/mnt/d/models/"
UNET_KERN_SIZE = 3
TRAIN_BATCH_SIZE = 16
INFER_BATCH_SIZE = 1
RANDOM_STATE = 8888

# %%
logger = logging.getLogger("main")
logging.basicConfig(
    filename=f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_logger.log",
    encoding="utf-8",
    level=logging.DEBUG
)

# %%
# ensure that data directory and patch config json paths are valid
assert os.path.isdir(DATA_DIREC)
assert os.path.isfile(PATCH_CONFIG_JSON)

# %% [markdown]
# ### Creating training config by building on patch_config

# %%
try:
    config_json = json.load(open(PATCH_CONFIG_JSON, 'r'))
    logger.info(f"Successfully loaded patch config from {PATCH_CONFIG_JSON}")
except Exception as e:
    logger.error(f"Could not load patch config json from {PATCH_CONFIG_JSON}.\nERROR: {e}")
    raise CAREInferenceError(f"Could not load patch config json from {PATCH_CONFIG_JSON}.\nERROR: {e}")

# %%
try:
    MODEL_NAME = f"FLAME_CARE_" \
        + f"{config_json['FLAME_Dataset']['input']['n_frames']}F" \
        + f"-" \
        + f"{config_json["FLAME_Dataset"]['output']['n_frames']}F"
    logger.info(f"Training a model with NAME: '{MODEL_NAME}'...")
except Exception as e:
    logger.error(f"Failed to dynamically load model name.\nERROR: {e}")
    raise CAREInferenceError(f"Failed to dynamically load model name.\nERROR: {e}")


# %%
try:
    RUN_ID = mlflow.search_runs(MODEL_NAME).shape[0]
    MODEL_DIREC = os.path.join(SAVE_DIREC, MODEL_NAME, str(RUN_ID))
    # exist_ok being True *SHOULD* (?) be fine because RUN_ID will not iterate upwards unless training either started or finished.
    os.makedirs(MODEL_DIREC, exist_ok = True)
    logger.info(f"Training run id is {RUN_ID}.")
    logger.info(f"Model saving to {MODEL_DIREC}")
except Exception as e:
    logger.error(f"Failed to load run id and/or set up model save directory.\nERROR: {e}")
    raise CAREInferenceError(f"Failed to load run id and/or set up model save directory.\nERROR: {e}")

# %%
config_json['Train_Config'] = {
    'npz_path': os.path.join(DATA_DIREC, config_json['Patch_Config']['name']),
    'name': MODEL_NAME,
    'model_direc': MODEL_DIREC,
    'unet_kern_size': UNET_KERN_SIZE,
    'train_batch_size': TRAIN_BATCH_SIZE,
    'random_state': RANDOM_STATE,
}

# %%
# verifying npz path...
NPZ_PATH = config_json['Train_Config']['npz_path']
assert os.path.isfile(config_json['Train_Config']['npz_path']), f"NPZ path {NPZ_PATH} is not a file"

# %% [markdown]
# ### Training and Validation Data

# %%
(X, Y), (X_val, Y_val), axes = load_training_data(
    NPZ_PATH,
    validation_split=0.1,
    verbose=True
)

# %%
c = axes_dict(axes)['C']
channels_in, channels_out = X.shape[c], Y.shape[c]

# %% [markdown]
# ### CARE Model

# %%
config_json

# %%
config_json['CARE_Model'] = {
    'name': MODEL_NAME,
    'experiment_name': EXPERIMENT_NAME,
    'run_id': RUN_ID,
    'base_dir': SAVE_DIREC,
    'run_dir': MODEL_DIREC
}
config_json['CARE_Model']['CSBDeep_Config'] = {
    'axes': axes,
    'n_channel_in': channels_in,
    'n_channel_out': channels_out,
    'probabilistic': False, # default from CSBDeep
    'allow_new_parameters': False, # default from CSBDeep
    'unet_kern_size': UNET_KERN_SIZE,
    'train_batch_size': TRAIN_BATCH_SIZE,
    'unet_input_shape': tuple(config_json['Patch_Config']['patch_shape']),
    'allow_new_parameters': True
}

# %%
config = Config(
    **config_json['CARE_Model']['CSBDeep_Config']
)

config_json['CARE_Model']['Model_Arch'] = vars(config)

# %%
JSON_CONFIG_PATH = os.path.join(MODEL_DIREC, "model_config.json")
json.dump(config_json, open(JSON_CONFIG_PATH, 'w+'))

# %% [markdown]
# ### Training the Model

# %%
model = CARE(
    config,
    str(RUN_ID),
    basedir=os.path.join(SAVE_DIREC, MODEL_NAME)
)

# %%
history = model.train(X, Y, validation_data=(X_val, Y_val), epochs=1)

# %%
# model.keras_model.save(os.path.join(MODEL_DIREC, 'saved_model.keras'))

# %% [markdown]
# ### Some quick visualizations

# %%
print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);
plt.savefig(os.path.join(MODEL_DIREC, "training_history.png"))

# %% [markdown]
# ### Model Evaluation from Validation Set

# %%
_P = model.keras_model.predict(X_val[:5])

# %%
plt.figure(figsize=(20,12))
if config.probabilistic:
    _P = _P[...,:(_P.shape[-1]//2)]
plot_some(X_val[:5],Y_val[:5],_P,pmax=99.5)
plt.suptitle('5 example validation patches\n'      
             'top row: input (source),  '          
             'middle row: target (ground truth),  '
             'bottom row: predicted from source');

plt.savefig(os.path.join(MODEL_DIREC, "val_set_predict_sample.png"))

# %% [markdown]
# ### Logging Model in MLFlow Database

# %%
val_loss = history.history['val_loss']
val_mae = history.history['val_mae']
val_mse = history.history['val_mse']

# %%
pyfunc_model = MLFLOW_CARE_Model(
    config_json_path=os.path.join(MODEL_DIREC, 'model_config.json')
)

# %%
# with mlflow.start_run():

mlflow.start_run()

logger.info(f"Run started")
print("Run started")

# Log the hyperparameters
mlflow.log_params(config_json)

# Log the validation performance metrics
mlflow.log_metric("val_loss", np.min(val_loss))
mlflow.log_metric("val_mae", np.min(val_mae))
mlflow.log_metric("val_mse", np.min(val_mse))

# infer the model signature
signature = infer_signature(X, pyfunc_model.predict(X))

# Log the model
# model_info = log_model(
#     artifact_path='care model',
#     python_model=pyfunc_model,
#     # model_code_path=os.path.join(os.getcwd(), "flame", "model.py"),
#     code_paths=[
#         os.path.join(os.getcwd(), "flame", "__init__.py"),
#         os.path.join(os.getcwd(), "flame", "mlflow_pyfunc.py"),
#         os.path.join(MODEL_DIREC, "model_config.json"),
#         os.path.join(MODEL_DIREC, "config.json")
#     ],
#     conda_env=os.path.join(os.getcwd(), "environment.yml"),
#     signature=signature,
#     input_example=X
# )

code_paths=[
        os.path.join(os.getcwd(), "flame", "__init__.py"),
        os.path.join(os.getcwd(), "flame", "mlflow_pyfunc.py"),
        os.path.join(MODEL_DIREC, "model_config.json"),
        os.path.join(MODEL_DIREC, "config.json")
    ]
model_info = log_model(artifact_path='care model', python_model=pyfunc_model, code_paths=code_paths, conda_env=os.path.join(os.getcwd(), "environment.yml"), signature=signature, input_example=X)
    
logger.info(f"Run finished.")
print("Run finished.")
mlflow.end_run()

# %% [markdown]
# ### Export to ONNX

# %%
input_shape = list(X.shape)
batch_dim = axes_dict(axes)['S']
input_shape[batch_dim] = None
print(input_shape)

# %%
input_signature = [
    tf.TensorSpec(
        input_shape, 
        tf.float32, 
        name='patch'
    )
]

# %%
onnx_model, _ = tf2onnx.convert.from_keras(
    model.keras_model,
    input_signature,
    opset=13
)

# %%
onnx.save(onnx_model, os.path.join(MODEL_DIREC, f"{MODEL_NAME}.onnx"))


