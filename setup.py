import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ygct",
    version="0.1.0",
    author="Your Name",
    author_email="your@email.com",
    description="A brief description of ygct",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/ygct",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=["requirement1", "requirement2"],
    entry_points={
        "console_scripts": [
            "ygct=ygct.main:main",
        ],
    },
)
