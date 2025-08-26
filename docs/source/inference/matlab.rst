=================
Usage with MATLAB
=================

MATLAB inference is primarily developed by Alex Vallmitjana.

The recommended workflow for running inference in MATLAB is as follows:

1. After each image is processed they are saved in a temporal folder.
2. When all are done, the script calls your CARE in that folder
3. Afterwards images are reconstructed depending on the original dimensions/flim formatting

For an example of this implementation, please view ``runCARE.m`` in the ``matlab`` source code folder.

As the core, MATLAB will run the following system commands:

::

    cmd1='conda activate care';
    cmd2=['python', <codepath>, 'command', 'line', 'parameters', ..., 'here'];
    system([cmd1 ' && ' cmd2]);

