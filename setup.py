from setuptools import find_packages, setup

long_description = """
Barrage is an opinionated supervised deep learning tool built on top of
TensorFlow 2.x designed to standardize and orchestrate the training and scoring of
complicated models. Barrage is built around a JSON config and the
TensorFlow 2.x library using the Tensorflow.Keras API.
"""

setup(
    name="barrage",
    version="0.6.0",
    description="A supervised deep learning tool.",
    long_description=long_description,
    author="Brian Nemsick",
    author_email="brian.nemsick@gmail.com",
    url="https://github.com/briannemsick/barrage/",
    license="MIT",
    python_requires=">=3.6",
    install_requires=[
        "click",
        "cytoolz>=0.9.0",
        "jsonschema>=3.0",
        "numpy",
        "pandas",
        "tensorflow>=2.4.0, <2.5",
    ],
    extras_require={
        "tests": [
            "black",
            "codecov",
            "flake8",
            "flake8-bugbear",
            "flake8-comprehensions",
            "flake8-print",
            "isort",
            "mypy",
            "pre-commit",
            "pre-commit-hooks",
            "pytest",
            "pytest-cov",
        ]
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
    entry_points={"console_scripts": ["barrage = barrage.console:cli"]},
    include_package_data=True,
    packages=find_packages(exclude=["docs*", "examples*", "tests*"]),
)
