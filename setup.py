# histlite/setup.py

from setuptools import setup
import sys
import setuptools

__version__ = '0.9.0'


setup(
    name='histlite',
    packages = ['histlite'],
    version=__version__,
    author='Mike Richman',
    author_email='mike.d.richman@gmail.com',
    description='A somewhat "lite" histogram library',
    long_description='',
    install_requires=['numpy', 'scipy', 'matplotlib'],
)
