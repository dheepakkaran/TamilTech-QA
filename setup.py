"""Setup script for the TamilTech-QA project."""
from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent
REQUIREMENTS = (HERE / "requirements.txt").read_text(encoding="utf-8").splitlines()
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

setup(
    name="tamiltech-qa",
    version="0.1.0",
    description=(
        "TamilTech-QA: the first Tanglish (Tamil-English code-switched) technical "
        "question-answering dataset, QLoRA fine-tuned model, and evaluation toolkit."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    author="TamilTech-QA Authors",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "notebooks", "scripts")),
    install_requires=[req for req in REQUIREMENTS if req and not req.startswith("#")],
    entry_points={
        "console_scripts": [
            "tamiltech-train=src.training.trainer:main",
            "tamiltech-eval=src.evaluation.metrics:main",
            "tamiltech-scrape=src.data_collection.youtube_scraper:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
