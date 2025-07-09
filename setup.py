# histlite/setup.py

from setuptools import setup, find_packages
import os
import sys
import setuptools

__version__ = '2022.8.26'


_here = os.path.abspath(os.path.dirname(__file__))
ABOUT = {}
with open(os.path.join(_here, 'histlite', 'version.py')) as f:
    exec(f.read(), ABOUT)

setup(
    name=ABOUT['package_name'],
    version=ABOUT['version'],
    author=ABOUT['author'],
    author_email=ABOUT['author_email'],
    packages = find_packages(),
    description=ABOUT['description'],
    long_description=ABOUT['long_description'],
    long_description_content_type='text/markdown',
    url=ABOUT['url'],
    install_requires=['numpy', 'scipy'],
)
