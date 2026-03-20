"""Setup script for cognitive_llm package."""

from setuptools import find_packages, setup

setup(
    name="cognitive-llm",
    version="0.1.0",
    description="Neuroscience-inspired architectural blocks for small language models",
    author="Riyad Mehdiyev",
    url="https://github.com/RiyadMehdi7/cognitive-llm",
    packages=find_packages(exclude=["tests*", "experiments*", "paper*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "transformers>=4.53.0",
        "datasets>=2.18.0",
        "accelerate>=0.30.0",
        "peft>=0.10.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "full": [
            "bitsandbytes>=0.46.1",
            "trl>=0.8.6",
            "evaluate>=0.4.0",
            "scipy>=1.11.0",
            "matplotlib>=3.8.0",
            "wandb>=0.16.0",
            "lm-eval>=0.4.2",
        ],
    },
)
