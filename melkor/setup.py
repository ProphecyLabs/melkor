from typing import List
from setuptools import setup, find_packages

requirements: List[str] = [
    line.strip() for line in open("requirements.txt", "r").readlines() if len(line) > 1
]

setup(
    name="melkor",
    version="0.1.0",
    description="Ames housing prediction",
    url="https://github.com/ProphecyLabs/melkor",
    packages=find_packages(exclude=["contrib", "docs", "tests", "notebooks", "data"]),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "dev": [],
        "test": ["pytest", "requests", "flake8==3.8.3", "mypy"],
    },
    package_data={},
    data_files=[],
    entry_points={},
)
