from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="offensivenn",
    version="1.1.1",
    author="Tharindu Ranasinghe",
    author_email="rhtdranasinghe@gmail.com",
    description="Offensive Language Identification with Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TharinduDR/https://github.com/TharinduDR/OffensiveNN",
    packages=find_packages(exclude=("examples", "docs", )),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "tensorflow"
        "pandas"
        "numpy"
        "gensim"
        "sklearn"
    ],
)