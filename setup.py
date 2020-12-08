#
# seirmo setuptools script
#
from setuptools import setup, find_packages


def get_version():
    """
    Get version number from the solver module.
    The easiest way would be to just ``import solver ``, but note that this may  # noqa
    fail if the dependencies have not been installed yet. Instead, we've put
    the version number in a simple version_info module, that we'll import here
    by temporarily adding the oxrse directory to the pythonpath using sys.path.
    """
    import os
    import sys

    sys.path.append(os.path.abspath('solver'))
    from version_info import VERSION as version
    sys.path.pop()

    return version


def get_readme():
    """
    Load README.md text for use as description.
    """
    with open('README.md') as f:
        return f.read()


setup(
    # Module name (lowercase)
    name='solver',

    # Version
    version=get_version(),

    description='This is a project that creates a solver of ODE systems.',  # noqa

    long_description=get_readme(),

    license='BSD 3-Clause "New" or "Revised" License',

    # author='',

    # author_email='',

    maintainer='',

    maintainer_email='',

    url='https://github.com/FarmHJ/numerical-solver.git',

    # Packages to include
    packages=find_packages(include=('solver', 'solver.*')),

    # List of dependencies
    install_requires=[
        # Dependencies go here!
        'numpy>=1.8'
    ],
    extras_require={
        'docs': [
            # Sphinx for doc generation. Version 1.7.3 has a bug:
            'sphinx>=1.5, !=1.7.3',
            # Nice theme for docs
            'sphinx_rtd_theme',
        ],
        'dev': [
            # Flake8 for code style checking
            'flake8>=3',
        ],
    },
)