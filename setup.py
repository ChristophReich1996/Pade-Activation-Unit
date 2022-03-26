from setuptools import setup

setup(
    name="pau",
    version="0.1",
    url="https://github.com/ChristophReich1996/Pade-Activation-Unit",
    license="MIT License",
    author="Christoph Reich",
    author_email="ChristophReich@gmx.net",
    description="PyTorch Pade-Activation-Unit",
    packages=["pau",],
    install_requires=["torch>=1.0.0"],
)