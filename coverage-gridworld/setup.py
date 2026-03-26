from setuptools import setup

setup(
    name="coverage_gridworld",
    version="1.1.0",
    author="Mateus Karvat",
    description="Coverage-focused gridworld environment with enemies, walls, and rotating fields of view.",
    install_requires=["gymnasium", "numpy", "pygame"],
    python_requires=">=3.10",
)
