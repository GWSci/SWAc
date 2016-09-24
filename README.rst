swacmod
=======

Development repo for the Surface Water Accounting Model (SWAcMod). In the following documentation, we'll refer to the ``ROOT`` directory as the one obtained by cloning this repository, i.e. the one containing this file.

Installation on win32 systems (tested on Windows 7)

- install `Microsoft Visual C++ Compiler <https://www.microsoft.com/en-us/download/details.aspx?id=44266>`_ (for Python 2.7)
- install `Anaconda <https://www.continuum.io/downloads>`_ (for Python 2.7, 32-bit)
- install `Git <https://git-scm.com/download/win>`_
- launch ``All Programs`` > ``Git`` > ``Git GUI`` > ``Clone Existing Repository`` with 

    ``source location``: ``https://github.com/AlastairBlack/SWAcMod``
    
    ``target``: ``C:/Users/Marco/Desktop/SWAcMod``
    
- launch ``All Programs`` > ``Anaconda2`` > ``Anaconda Prompt`` and install ``SWAcMod`` with

.. code-block:: bash

    > cd Desktop/SWAcMod
    > python setup.py install

To **confirm that all modules are installed correctly**: navigate to ``ROOT``, launch a python shell and run

.. code-block:: python

    >>> from swacmod import input_output, utils, validation, finalization

This should not produce errors.

To **run the end-to-end tests**: navigate to ``ROOT``, then run

.. code-block:: bash

    $ python setup.py test

The tests will not produce output files, they will only print to ``stdout``.
If they succeed, you will see an output like the following:

.. code-block:: bash

    running test
    running egg_info
    writing requirements to swacmod.egg-info/requires.txt
    writing swacmod.egg-info/PKG-INFO
    writing top-level names to swacmod.egg-info/top_level.txt
    writing dependency_links to swacmod.egg-info/dependency_links.txt
    reading manifest file 'swacmod.egg-info/SOURCES.txt'
    writing manifest file 'swacmod.egg-info/SOURCES.txt'
    running build_ext
    model.pyx modified, recompiling
    test_get_output (tests.tests.EndToEndTests)
    Test for get_output() function. ... ok
    test_val_num_nodes (tests.tests.EndToEndTests)
    Test for val_num_nodes() function. ... ok
    test_val_start_date (tests.tests.EndToEndTests)
    Test for val_start_date() function. ... ok
    test_validate_all (tests.tests.EndToEndTests)
    Test for validate_all() function. ... ok
    test_validate_functions (tests.tests.EndToEndTests)
    Test that all parameters and series have a validation function. ... ok

    ----------------------------------------------------------------------
    Ran 5 tests in 3.064s

    OK

To **run the model without output files** (but with a log file): navigate to ``ROOT``, then run

.. code-block:: bash

    $ python swacmod_run.py test

Finally, to simply **run the model**: navigate to ``ROOT``, then run

.. code-block:: bash

    $ python swacmod_run.py

This will write the output files in the ``output_files/`` directory.
