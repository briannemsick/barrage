from setuptools import find_packages, setup


long_description = """
Barrage is an opinionated supervised deep learning tool built on top of
TensorFlow 2.x designed to standardize and orchestrate the training and scoring of
complicated models. Barrage is built around a JSON config and the
TensorFlow 2.x library using the Tensorflow.Keras API.
"""

setup(
    name="barrage",
    version="0.1.0a3",
    description="A supervised deep learning tool.",
    long_description=long_description,
    author="Brian Nemsick",
    author_email="brian.nemsick@gmail.com",
    url="https://github.com/briannemsick/barrage/",
    license="MIT",
    python_requires=">=3.6.0",
    install_requires=[
        "cytoolz>=0.9.0.1",
        "jsonschema>=3.0.0",
        "numpy>=1.16.0",
        "pandas>=0.24.0",
    ],
    extras_require={
        "tensorflow": ["tensorflow==2.0.0b1"],
        "tensorflow-gpu": ["tensorflow-gpu==2.0.0b1"],
        "tests": [
            "black",
            "coveralls",
            "flake8",
            "mypy",
            "pre-commit",
            "pytest",
            "pytest-cov",
        ],
    },
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    include_package_data=True,
    packages=find_packages(exclude=["docs*", "examples*", "tests*"]),
)
