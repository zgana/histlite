# histlite/setup.py

from setuptools import setup, find_packages
import sys
import setuptools

__version__ = '2022.8.26'


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='histlite',
    version=__version__,
    author='Mike Richman',
    author_email='mike.d.richman@gmail.com',
    packages = find_packages(),
    description='A somewhat "lite" histogram library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/zgana/histlite',
    install_requires=['numpy', 'scipy'],
)
