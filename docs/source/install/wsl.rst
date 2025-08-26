==============================
WSL (Training & CLI Inference)
==============================

WSL stands for **W**indows **S**ubsystem for **L**inux. It is required for training because training is performed through
the TensorFlow package. TensorFlow can only interface with the GPU on a Windows computer using the WSL virtual machine 
as an interface (as of Aug 2025). Thus, Python packages and dependencies must be installed on WSL for training.

The WSL interface can also be used for inference. However, to run inference, **CUDA, cuDNN, and TensorRT will have to be
installed in WSL separately from any native Windows installations**. While inference is possible thorugh WSL, this tutorial 
is designed for training; therefore, it won't include installation instructions for these dependencies in WSL for now.

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

c. Open WSL
~~~~~~~

Close and re-open the PowerShell prompt. Then, open WSL according to the image below:

.. image:: ../../images/install/wsl/open_ubuntu.png
    :alt: Image showing dropdown to open a new shell in PowerShell with Ubuntu-24.04 boxed.


2. Install Python Dependencies
^^^^^^^^^^^^^^

a. Download Source Code
~~~~~~~~~~~

Either using GitHub CLI or the GitHub ZIP downloader, download the source code from 
`the FLAME-CARE repository <https://github.com/AlexSath/FLAME-CARE>`_.

⚠️ It is recommended that the source code be installed to the Windows filesystem (File Explorer on Windows machine) as
opposed to the WSL filesystem. This way both Windows and WSL will have easy access to the code.

Below is an image of how to download the ZIP:

.. image:: ../../images/install/git_zip_download.png
    :alt: Image of GitHub GUI with ZIP download button circled.

If issues are still experienced when downloading the source code, visit the `Windows Install Page 
<https://flame-care.readthedocs.io/en/latest/install/windows.html>`_ for more information.

b. Access Windows Filesystem through ``mnt``
~~~~~~~~~~

If following the instructions, the source code should be installed on the Windows Filesystem. To access it in WSL,
simply use the "Mount" path found at ``/mnt``. Example:

.. image:: ../../images/install/wsl/mount.png
    :alt: Image showing the wsl /mnt path

``cd`` (**C**hange **D**irectory) and ``ls`` (List Files) can then be used to navigate to the folder where the source
code was installed. 
