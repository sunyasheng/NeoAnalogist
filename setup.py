from setuptools import find_packages, setup

setup(
    name="gpt-scientist",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests",
        "pyyaml",
        "PyPDF2",
    ],
    python_requires=">=3.7",
    # Project structure
    package_data={
        "": ["*.yaml", "*.yml"],
    },
    # Entry points
    entry_points={
        "console_scripts": [
            "arxiv-download=agent.opt.arxiv:arxiv_download",
            "paper2md=app:main",
        ],
    },
)
