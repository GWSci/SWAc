swacmod
=======

Development repo for the Surface Water Accounting Model (SWAcMod). In the following documentation, we'll refer to the ``ROOT`` directory as the one obtained by cloning this repository, i.e. the one containing ``setup.py``.

Installation on Windows

- install Microsoft Visual C++ Compiler for Python 2.7 (https://www.microsoft.com/en-us/download/details.aspx?id=44266)
- install Anaconda for Python 2.7, 32-bit (https://www.continuum.io/downloads)
- install Git (https://git-scm.com/download/win)
- clone repo using Git GUI in SWAcMod/
- from an Anaconda Prompt, install SWAcMod with

    $ python setup.py install

To **confirm that all modules are installed correctly**: navigate to ``ROOT``, launch a python shell and run

.. code-block:: python

    >>> from swacmod import input_output, swacmod, model

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
