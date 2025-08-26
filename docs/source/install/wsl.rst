==============================
WSL (Training & CLI Inference)
==============================

WSL stands for **W**indows **S**ubsystem for **L**inux. It is required for training because training is performed through
the TensorFlow package. TensorFlow can only interface with the GPU on a Windows Computer using the WSL virtual machine 
as an interface (as of Aug 2025). Thus, python packages and dependencies must be install on WSL for training.

The WSL interface can also be used for inference. However, to run inference, **CUDA, cuDNN, and TensorRT will have to be
installed in WSL separately from any native Windows installations**. While inference is possible thorugh WSL, this tutorial 
is designed for training; therefore, it won't include installation instructions for these packages in WSL for now.

1. Install WSL (Ubuntu 24.04)
^^^^^^^^^^^^^^

a. Open PowerShell
~~~~~~~~~~

Hit the Windows Button or visit the Start Menu and type "PowerShell".

b. Install
~~~~~~~~~~

List possible WSL distributions and verify that ``Ubuntu-24.04`` is present:

::
    wsl --list --online

Next, install the desired distribution:

::
    wsl --install Ubuntu-24.04


Setting up WSL
^^^^^^^^^^^^^^

::
    sudo apt install wslu