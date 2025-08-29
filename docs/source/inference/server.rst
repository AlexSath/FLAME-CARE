Start MLFlow Server
===================

``start_mlflow_server.py``
^^^^^^^^^^^

        usage: start_mlflow_server.py [-h] [--ip IP] [--port PORT] [--tracking-direc TRACKING_DIREC]

        options:
        -h, --help            show this help message and exit
        --ip IP
        --port PORT
        --tracking-direc TRACKING_DIREC

Examples
~~~~~~~

::
    python start_mlflow_server.py --ip 127.0.0.1 --port 5050 --tracking-direc </path/to/mlflow/mlruns>