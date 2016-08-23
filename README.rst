swacmod
=======

Development Repo for Surface Water Accounting Model (SWAcMod). In the following documentation, we'll refer to the ``ROOT`` directory as the one obtained by cloning this repository, i.e. the one containing ``setup.py``.

The only dependency that needs to be installed is ``xlrd``, used to read the Excel input file. In order to install it, I recommend first installing ``pip``, a python package manager, if you haven't already done so (instructions `here <https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py>`_). Then run

.. code-block:: bash

    $ pip install xlrd

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
    test_get_output (swacmod.tests.EndToEndTests)
    Test for get_output() function. ... ok
    test_load_input (swacmod.tests.EndToEndTests)
    Test for load_input_from_excel() function. ... ok
    test_load_params (swacmod.tests.EndToEndTests)
    Test for load_params_from_excel() function. ... ok

    ----------------------------------------------------------------------
    Ran 3 tests in 1.281s

    OK

Finally, to **run the model**: navigate to ``ROOT``, then run

.. code-block:: bash

    $ python -m swacmod.swacmod

This will produce ``output.csv`` file and a ``.log`` file in the ``output_files/`` directory.
