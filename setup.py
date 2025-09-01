# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 11:50:24 2025

@author: p-sik
"""

import setuptools

def get_version():
    with(open("src/Reader4D/__init__.py", "r")) as fh:
        for line in fh:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")
                
def get_long_description():
    with open("README.md", "r") as fh: description = fh.read()
    return(description)
    
setuptools.setup(
    name="Reader4D",
    version=get_version(),
    author="Pavlina Sikorova",
    author_email="pavlinasik@isibrno.cz",
    description=\
        "Fast readers for 4D-STEM datasets (Timepix3, CSR/HDF5/ADVB).",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/pavlinasik/Reader4D/",
    project_urls={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
    license='MIT',
    package_dir={"":"src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    include_package_data=True)