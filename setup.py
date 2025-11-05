
```python
from setuptools import setup, find_packages

setup(
    name="hkr",
    version="1.0.0",
    description="Heat Kernel Regularization MVP Framework",
    author="Joshua & ChatGPT-5",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["torch>=2.0.0", "tqdm", "matplotlib"],
    python_requires=">=3.9",
)
