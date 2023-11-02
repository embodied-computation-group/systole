# Guidelines for contributing

Systole is a scientific community-driven software project, and we welcome contributions from interested individuals or groups. In these guidelines, we provide potential contributors information regarding the aim of the Systole project.

Systole has been developed with a focus on interactive reports, fast feature extraction and artefacts detection/correction for respiratory, electrocardiography and photoplethysmography signals. We do not intend to cover other physiological signals (EMG, EGG, pupil...) and we prefer to redirect the interested users to other existing packages.

We are welcoming potential contributions on Systole that can include:

* Fixing issues/errors with the current codebase.
* Improving the documentation of the main functions, tutorials and examples.
* New examples of use/tutorials for meaningful physiological analysis pipelines.
* Visualization functions.
* Feature extraction algorithms for ECG, PPG or respiration time series.
* New recording classes.
* Correction algorithms.

Ideally, these contributions should use/implement methods that have already been published. The relevant papers should be properly cited in the documentation string of the functions as well as in the tutorial/example describing the use of such methods.

# Opening issues

We appreciate being notified of problems with the existing Systole code. We prefer that issues be filed on [Github Issue Tracker](https://github.com/LegrandNico/systole/issues), rather than on social media or by direct email to the developers.

Please verify that your issue is not being currently addressed by other issues or pull requests by using the GitHub search tool to look for keywords in the project issue tracker.

# Contributing code via pull requests

While issue reporting is valuable, we strongly encourage users who are inclined to do so to submit patches for new or existing issues via pull requests. This is particularly the case for simple fixes, such as typos or tweaks to documentation, which do not require a heavy investment of time and attention.

Contributors are also encouraged to contribute new code to enhance Systole's functionality, also via pull requests. Please consult the [Systole documentation](https://LegrandNico.github.io/systole/#) to ensure that any new contribution does not strongly overlap with existing functionality.

The preferred workflow for contributing to Systole is to fork the [GitHub repository](https://github.com/LegrandNico/systole), clone it to your local machine, and develop on a feature branch.

## Steps:

1. Fork the [project repository](https://github.com/LegrandNico/systole) by clicking on the 'Fork' button near the top right of the main repository page. This creates a copy of the code under your GitHub user account.

2. Clone your fork of the Systole repo from your GitHub account to your local disk, and add the base repository as a remote:

   ```bash
   $ git clone git@github.com:<your GitHub handle>/systole.git
   $ cd systole
   $ git remote add upstream git@github.com:LegrandNico/systole.git
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a ``feature`` branch. It's good practice to never routinely work on the ``main`` branch of any repository.

4. Project requirements are in ``requirements.txt``, and libraries used for testing (including running the tutorial notebooks) are in ``requirements-test.txt``.

   You may (probably in a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/)) run:

   ```bash
   $ pip install -e .
   $ pip install -r requirements-dev.txt
   ```
5. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes locally.
   After committing, it is a good idea to sync with the base repository in case there have been any changes:

   ```bash
   $ git fetch upstream
   $ git rebase upstream/main
   ```

   Then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

6. Go to the GitHub web page of your fork of the Systole repo. Click the 'Pull request' button to send your changes to the project's maintainers for review. This will send an email to the committers.

## Pull request checklist

We recommended that your contribution complies with the following guidelines before you submit a pull request:

*  If your pull request addresses an issue, please use the pull request title to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is created.

*  All public methods must have informative docstrings with sample usage when appropriate.

*  Please prefix the title of incomplete contributions with `[WIP]` (to indicate a work in progress). WIPs may be useful to (1) indicate you are working on something to avoid duplicated work, (2) request a broad review of functionality or API, or (3) seek collaborators.

*  All other tests pass when everything is rebuilt from scratch.

*  When adding additional functionality, provide at least one example in the docstring of the function. Depending on the method being implemented, it can also be useful to add an example in the `/example/` folder of sub-folders, and/or to improve the tutorials notebooks in the `/source/notebooks/` folder. Have a look at other examples for reference.

* Documentation and high-coverage tests are necessary for enhancements to be accepted.

* Type [hints](https://docs.python.org/3/library/typing.html) for all parameters and output variables. We use [MyPy](https://github.com/python/mypy) for static type checking.

* No `pre-commit` errors.

## Style guide

We have configured a pre-commit hook that checks for `black`-compliant and `flake8`-compliant code style, `isort`-compliant import sorting, and `mypy`-compliant type hints.

For documentation strings, we *prefer* [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html) to comply with the style that predominates in our upstream dependencies.

---

##### This file has been adapted from the [PYMC guide to contributing](https://github.com/pymc-devs/pymc/blob/main/CONTRIBUTING.md).

---