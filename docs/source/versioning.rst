===============
Version Control
===============

Introduction
^^^^^^^^^^^^

Version Control is a key component of Machine Learning Operations (MLOps). It allows for understanding of where
models come from as experimental parameters are iterated through. It enables easy access to the best models
based on well-documented evaluation techniques.

To enable version control, this project utilizes two tools. 

I. MLFlow
~~~~~~~~~

`MLFlow <https://mlflow.org/docs/2.2.2/>`_ is a platform for managing machine learning lifecycles. As projects grow 
along with datasets, scientists can often grow overwhelmed by the number of parameters that need to be tracked to 
understand the training, inference, and performance of any given model in their project. **The purpose of MLFlow 
in CARE for FLAME is to provide infrastructure to track these parameters and simplify decision-making in choosing a 
model for FLAME image denoising in the clinic**.

MLFlow will be installed as a part of the ``care`` Conda environment built in either of the install tutorials.

Once MLFlow is installed, it can interface with any model repository. MLFlow model storage is split into **metadata**
and **artifact** storage. While both can and will be stored in Synology Drive in this project, each has their own 
range of possibilities. See `backend <https://www.mlflow.org/docs/latest/ml/tracking/backend-stores>`_ and 
`artifact <https://mlflow.org/docs/latest/ml/tracking/artifact-stores>`_ stores for more information.

II. Synology Drive
~~~~~~~~~~~~~~~~~

To enable file sharing, this project was built and tested using Synology Drive. While MLFlow can detect and use models
in any location on the local machine's filesystem, Synology Drive enables someone on one computer to train a model which
then gets automatically synced with another machine that has the same Synology Drive folder mounted on their filesystem.
Therefore, Synology Drive enables synced model training and sharing for all members of Balu Lab.

1. Synology Drive Mount
^^^^^^^^^^^^^^^^^^^^^^^

To mount Synology Drive, download the `Synology Drive Client <https://www.synology.com/en-global/support/download/RS1221RP+?version=7.2#utilities>`_

Once downloaded, create a Sync Task.

Add the Synology Drive server using the IP provided by other members of Balu Lab.

Hit ``Create`` and ensure that the ``NLOM-DATA/CARE_for_MATLAB/mlruns`` folder is mounted to your machine:
* If ``NLOM-DATA`` is mounted, then all directories within ``NLOM-DATA`` will appear in a new ``SynologyDrive`` folder 
  on the machine.
* If ``NLOM-DATA/CARE_for_MATLAB`` is mounted, then all directories within ``CARE_for_MATLAB`` will appear in the 
  ``SynologyDrive`` folder.

Assuming that ``NLOM-DATA/CARE_for_MATLAB`` was mounted, and the following was selected in ``Sync Rules``:

.. image:: ../../images/versioning/SynologyDrive_sync.png

Then the user would see the following in their file tree:

.. image:: ../../images/versioning/SynologyDrive_mount.png

2. MLFlow
^^^^^^^^^

To download MLFlow, follow the instructions for setting up the ``care`` conda environment in the :doc:`/training` for details.

a. Starting Tracking Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run the following command:

::
    mlflow server --host 127.0.0.1 --port 5050 --serve-artifacts --backend-store-uri <path/to/CARE_for_MATLAB/mlruns> --default-artifact-root <path/to/CARE_for_MATLAB/mlruns> --artifacts-destination <path/to/CARE_for_MATLAB/mlruns>



b. Viewing Stored Models 
~~~~~~~~~~~~~~~~~~~~~~~~

c. Model Registry
~~~~~~~~~~~~~~~~~

