swacmod
=======

Development repo for the Surface Water Accounting Model (SWAcMod). In the following documentation, we'll refer to the ``ROOT`` directory as the one obtained by cloning this repository, i.e. the one containing this file.

Installation on Windows systems:
-------------------------------------------------------

This installation assumes that git and python are installed.

- https://git-scm.com/
- https://www.python.org/

1. Install Microsoft C++ Build tools.

- Download the installer from https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Choose Desktop development with C++.
- In the right hand pane, make sure that MSVC and Windows SDK are selected.
- Click "Install" in the bottom right hand corner.

2. Create a folder to install swacmod and navigate into it. For example:

.. code-block:: bash

    mkdir swac
    cd swac

3. Download the source code and install dependencies:

.. code-block:: bash

    git clone git@github.com:GWSci/SWAcMod.git .
    setup_windows.bat

**Troubleshooting:** If the ``git clone`` command fails then you might need to set up a key for authentication. Check the link below for instructions:

https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

4. Run a test model

.. code-block:: bash

    run.bat -i .\input_files\input.yml -o .\output_files\

After the initial installation, models can be run with a command similar to step 4. Steps 1--3 do not need to be repeated.

Regarding Python environments, the intent is that users do not have to manage environments themselves. The ``setup_windows.bat`` script creates an environment using venv. The ``run.bat`` script will activate and exit the environment automatically.

Installation on Linux systems:
------------------------------

This installation assumes that git and python are installed, and that your shell is bash.

- https://git-scm.com/
- https://www.python.org/

1. Create a folder to install swacmod and navigate into it using the terminal. For example:

.. code-block:: bash

    mkdir swac
    cd swac

2. Download the source code and install dependencies:

.. code-block:: bash

    git clone git@github.com:GWSci/SWAcMod.git .
    ./setup_linux.sh

**Troubleshooting:** If the ``git clone`` command fails then you might need to set up a key for authentication. Check the link below for instructions:

https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

**Troubleshooting:** If the ``setup_linux.sh`` command fails, then you might need to install python3-venv. The error message will probably tell you how to do this, but in any case the command is: ``apt install python3.10-venv``.

3. Run a test model

.. code-block:: bash

    ./run.sh -i ./input_files/input.yml -o ./output_files/

**Troubleshooting:** If there is a cython compilation error complaining that Python.h does not exist then you might need to install ``python-dev``. Run the command ``sudo apt-get install python3-dev`` to install it, and then run ``touch swacmod/cymodel.pyx`` to force a recompilation on the next run. Then try running the test model again.

After the initial installation, models can be run with a command similar to step 3. Steps 1--2 do not need to be repeated.

Regarding Python environments, the intent is that users do not have to manage environments themselves. The ``setup_linux.sh`` script creates an environment using venv. The ``run.sh`` script will activate and exit the environment automatically.

Installation on Mac systems:
----------------------------

This installation assumes that git and homebrew are installed.

- https://git-scm.com/
- https://brew.sh/

1. Create a folder to install swacmod and navigate into it using the terminal. For example:

.. code-block:: bash

    mkdir swac
    cd swac

2. Download the source code and install dependencies:

.. code-block:: bash

    git clone git@github.com:GWSci/SWAcMod.git .
    ./setup_mac.sh

**Troubleshooting:** If the ``git clone`` command fails then you might need to set up a key for authentication. Check the link below for instructions:

https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

3. Run a test model

.. code-block:: bash

    ./run.sh -i ./input_files/input.yml -o ./output_files/

After the initial installation, models can be run with a command similar to step 3. Steps 1--2 do not need to be repeated.

Regarding Python environments, the intent is that users do not have to manage environments themselves. The ``setup_mac.sh`` script creates an environment using venv. The ``run.sh`` script will activate and exit the environment automatically.

Command Line Arguments:
------------------------------------------------------

To see the optional arguments run the model with the -h argument

.. code-block:: bash

    python swacmod_run.py -h

    usage: swacmod_run.py [-h] [-t] [-d] [-r] [-i INPUT_YML] [-o OUTPUT_DIR]
                      [-f {hdf5,h5,csv}] [-s]

    optional arguments:
      -h, --help            show this help message and exit
      -t, --test            run the whole model, but do not output any file
      -d, --debug           verbose logging
      -r, --reduced         output reduced format files
      -i, --input_yml       path to input yaml file inside input directory
      -o, --output_dir      path to output directory
      -f, --format          output file format, choose between ``hdf5`` (or ``h5``) and ``csv``
      -s, --skip_prompt     skip user prompts and warnings

For example,

.. code-block:: bash

    python swacmod_run.py -d -r -i path_to_input/input001.yml -o path_to_output/ -f csv -s

Flags can also be combined, the above is equivalent to

.. code-block:: bash

    python swacmod_run.py -drs -i path_to_input/input001.yml -o path_to_output/ -f csv