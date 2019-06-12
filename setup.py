from setuptools import find_packages, setup


setup(
    name="barrage",
    version="0.1.0a0",
    description="A supervised deep learning tool.",
    author="Brian Nemsick",
    author_email="brian.nemsick@gmail.com",
    url="https://github.com/briannemsick/barrage/",
    license="MIT",
    python_requires=">=3.6.0",
    install_requires=["jsonschema>=3.0.0", "numpy>=1.16.0", "pandas>=0.24.0"],
    extras_require={
        "cpu": ["tensorflow==2.0.0b0"],
        "gpu": ["tensorflow-gpu==2.0.0b0"],
        "tests": ["black", "flake8", "mypy", "pre-commit", "pytest", "pytest-cov"],
    },
    packages=find_packages(),
    include_package_data=True,
)
