from setuptools import setup, find_packages

setup(
    name="medvill",
    version="2.0.0",
    description="MedViLL – Medical Vision-Language Learning (modernized)",
    packages=find_packages(exclude=["scripts", "configs", "data"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.40.0",
        "einops>=0.7.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "sacrebleu>=2.3.1",
        "omegaconf>=2.3.0",
        "tqdm>=4.65.0",
    ],
)
