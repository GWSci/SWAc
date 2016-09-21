swacmod
=======

Development repo for the Surface Water Accounting Model (SWAcMod). In the following documentation, we'll refer to the ``ROOT`` directory as the one obtained by cloning this repository, i.e. the one containing ``setup.py``.

Installation on Windows

- install Git using default options
- launch Git Bash
- install virtualenv with:

    $ pip install virtualenv

- create a new virtualenv with:

    $ cd Dektop
    $ virtualenv SWAcMod_env

- clone repo in SWAcMod_env/SWAcMod/
- activate virtualenv

    $ cd SWAcMod_env
    $ source Scripts/activate

- install module

    $ python setup.py install

Installation on UNIX

The dependencies that need to be installed are `numpy`, ``dateutil`` and ``PyYAML``. In order to install them, I recommend first installing ``pip``, a python package manager, if you haven't already done so (instructions `here <https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py>`_). Then run

.. code-block:: bash

    $ pip install numpy python-dateutil pyyaml

Then download this Git repo.
To **confirm that all modules are installed correctly**: navigate to ``ROOT``, launch a python shell and run

.. code-block:: python

    >>> from swacmod import io, swacmod, model

This should not produce errors.

To **run the end-to-end tests**: navigate to ``ROOT``, then run

.. code-block:: bash

    $ python setup.py test

The tests will not produce output files, they will only print to ``stdout``.
If they succeed, you will see an output like the following:

.. code-block:: bash

    running test
    running egg_info
    writing swacmod.egg-info/PKG-INFO
    writing top-level names to swacmod.egg-info/top_level.txt
    writing dependency_links to swacmod.egg-info/dependency_links.txt
    reading manifest file 'swacmod.egg-info/SOURCES.txt'
    writing manifest file 'swacmod.egg-info/SOURCES.txt'
    running build_ext
    test_get_output (tests.tests.EndToEndTests)
    Test for get_output() function. ... ok
    test_validate_all (tests.tests.EndToEndTests)
    Test for validate_all() function. ... ok
    test_validate_functions (tests.tests.EndToEndTests)
    Test that all parameters and series have a validation function. ... ok

    ----------------------------------------------------------------------
    Ran 3 tests in 3.023s

    OK

Finally, to **run the model**: navigate to ``ROOT``, then run

.. code-block:: bash

    $ python -m swacmod.swacmod

This will produce an ``output.csv`` file for each node and a ``.log`` file in the ``output_files/`` directory.
