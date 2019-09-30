# This is the SETUP file
from distutils.core import setup

setup(
    name='ProcessEntropy',
    version='0.2dev',
    packages=['ProcessEntropy',],
    license='MIT license',
    description='A toolkit for calculating process entropy quickly. With specific applications to tweets.',
    long_description=open('README.md').read(),
    install_requires=[
        "numba",
        "nltk",
    ],
)