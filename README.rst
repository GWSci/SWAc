swacmod
=======

Development repo for the Surface Water Accounting Model (SWAcMod). In the following documentation, we'll refer to the ``ROOT`` directory as the one obtained by cloning this repository, i.e. the one containing this file.

Installation on Windows systems (tested on Windows 7):
------------------------------------------------------

- install `Microsoft Visual C++ Compiler <https://web.archive.org/web/20210106040224/https://download.microsoft.com/download/7/9/6/796EF2E4-801B-4FC4-AB28-B59FBF6D907B/VCForPython27.msi>`_ (for Python 2.7)
- install `Anaconda <https://repo.anaconda.com/archive/Anaconda2-2019.10-Windows-x86_64.exe>`_ (for Python 2.7, 64-bit)
- install `Git <https://git-scm.com/download/win>`_
- launch ``Start`` > ``All Programs`` > ``Git`` > ``Git GUI`` > ``Clone Existing Repository``
- when prompted, use as parameters:

    source location: https://github.com/AlastairBlack/SWAcMod

    target: C:/somewhere/SWAcMod

- launch ``Start`` > ``All Programs`` > ``Anaconda2`` > ``Anaconda Prompt`` and install ``SWAcMod`` with

.. code-block:: bash

    > cd Desktop/SWAcMod
    > python setup.py install
    > pip install flopy==3.2.12

At this point you may experience an error.  Admin rights to download files are required to obtain FloPy.  If this command fails
you can continue the below process however when running SWAc you will be unable to write MODFLOW output files from SWAc.
This will be noted by a message after the SWAc run stating NO FLOPY MODULE

To **run a test version of the model** (including a log file): navigate to ``ROOT``, then run

.. code-block:: bash

    $ python swacmod_run.py -i .\input_files\input.yml -o .\output_files\

These are lower case -i and -o.  The input_files folder contains a full worked example of both csv and yml formatted input files.

Finally, to see the optional arguments run the model with the -h argument

.. code-block:: bash

    $ python swacmod_run.py -h

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

    $ python swacmod_run.py -d -r -i path_to_input/input001.yml -o path_to_output/ -f csv -s

Flags can also be combined, the above is equivalent to

.. code-block:: bash

    $ python swacmod_run.py -drs -i path_to_input/input001.yml -o path_to_output/ -f csv


.. note::
   Python package maintainers may no longer support Python2, if using Python2 then restrict the python environment to the package versions listed in requirements.txt, FloPy is noted as one such package.
   
   The swac code is agnostic to Python 2 and 3 however this readme is specified for python2.  An alternative C++ redistributable may be required for Python3.x 

Installation on Linux systems:
------------------------------

This installation assumes that git and python are installed, and that your shell is bash.

- https://git-scm.com/
- https://www.python.org/

1. Create a folder to install swacmod and navigate into it using the terminal. For example:

.. code-block:: bash

    $ mkdir swac
    $ cd swac

2. Download the source code and install dependencies:

.. code-block:: bash

    $ git clone git@github.com:GWSci/SWAcMod.git .
    $ ./setup_linux.sh

**Troubleshooting:** If the ``git clone`` command fails then you might need to set up a key for authentication. Check the link below for instructions:

https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

**Troubleshooting:** If the ``setup_linux.sh`` command fails, then you might need to install python3-venv. The error message will probably tell you how to do this, but in any case the command is: ``apt install python3.10-venv``.

3. Run a test model

.. code-block:: bash

    $ ./run.sh -i ./input_files/input.yml -o ./output_files/

Regarding Python environments, the intent is that users do not have to manage environments themselves. The ``setup_linux.sh`` script creates an environment using venv. The ``run.sh`` script will activate and exit the environment automatically.

Installation on Mac systems:
----------------------------

This installation assumes that git and homebrew are installed.

- https://git-scm.com/
- https://brew.sh/

1. Create a folder to install swacmod and navigate into it using the terminal. For example:

.. code-block:: bash

    $ mkdir swac
    $ cd swac

2. Download the source code and install dependencies:

.. code-block:: bash

    $ git clone git@github.com:GWSci/SWAcMod.git .
    $ ./setup_mac.sh

**Troubleshooting:** If the ``git clone`` command fails then you might need to set up a key for authentication. Check the link below for instructions:

https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

3. Run a test model

.. code-block:: bash

    $ ./run.sh -i ./input_files/input.yml -o ./output_files/

Regarding Python environments, the intent is that users do not have to manage environments themselves. The ``setup_mac.sh`` script creates an environment using venv. The ``run.sh`` script will activate and exit the environment automatically.
