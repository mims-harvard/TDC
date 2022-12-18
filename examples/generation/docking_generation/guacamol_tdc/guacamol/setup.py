import io
import re
from os import path

from setuptools import setup

# Get the version from guacamol/__init__.py
# Adapted from https://stackoverflow.com/a/39671214
__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open("guacamol/__init__.py", encoding="utf_8_sig").read(),
).group(1)

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="guacamol",
    version=__version__,
    author="BenevolentAI",
    author_email="guacamol@benevolent.ai",
    description="Guacamol: benchmarks for de novo molecular design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BenevolentAI/guacamol",
    packages=["guacamol", "guacamol.data", "guacamol.utils"],
    license="MIT",
    install_requires=[
        "joblib>=0.12.5",
        "numpy>=1.15.2",
        "scipy>=1.1.0",
        "tqdm>=4.26.0",
        "FCD==1.1",
        # FCD doesn't pin the tensorflow and Keras dependencies, so we have to do the honours
        "tensorflow==1.15.4",
        "Keras==2.1.0",
        "h5py==2.10.0",
    ],
    python_requires=">=3.6",
    extras_require={
        "rdkit": ["rdkit>=2018.09.1.0"],
    },
    include_package_data=True,
    zip_safe=False,
)
