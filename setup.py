# install the package: python -m pip install -e .
import os
import setuptools

tf_name = "transfinder"
tf_init = os.path.join(tf_name, "__init__.py")

# readme
with open("README.md", "r") as f:
    long_description = f.read()

# requirements
with open("requirements.txt", "r") as f:
    requirements = [req.strip() for req in f.readlines() if not req.startswith("#")]

with open(tf_init, "r") as f:
    for line in f:
        if line.startswith("__version__"):
            _, _, tf_version = line.replace('"', "").split()
        elif line.startswith("__date__"):
            _, _, tf_vdate = line.replace('"', "").split()
        else:
            pass

# install
setuptools.setup(
    name=tf_name,
    version=tf_version,
    author="Dezi Liu",
    author_email="adzliu@ynu.edu.cn",
    description="Transient detection and classification pipeline",
    long_description=long_description,
    url="https://github.com/LiuDezi/TransFinder/",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    include_package_data=True, 
    python_requires=">=3.10",
    install_requires=requirements,
)
