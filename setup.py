from distutils.core import setup

setup(
    name='ctr_framework',
    version='1',
    packages=[
        'ctr_framework',
    ],
    install_requires=[
        'numpy',
        'dash==1.2.0',
        'dash-daq==0.1.0',
        'pint',
        'guppy3',
        'sphinx-rtd-theme',
        'sphinx-code-include',
        'jupyter-sphinx',
        'numpydoc',
    ],
)
