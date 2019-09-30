# This is the SETUP file
from distutils.core import setup

setup(
    name='ProcessEntropy',
    version='0.2dev',
    packages=['ProcessEntropy',],
    scripts=['example_usage', 'test_code'] ,
    license='MIT license',
    description='A toolkit for calculating process entropy quickly. With specific applications to tweets.',
    long_description=open('README.md').read(),
    url="https://github.com/tobinsouth/ProcessEntropy",
    install_requires=[
        "numba",
        "nltk",
    ],
)