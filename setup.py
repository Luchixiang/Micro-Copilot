import setuptools
from setuptools import setup

install_deps = [
    'numpy>=1.20.0',
    'scipy',
    'natsort',
    'tifffile',
    'tqdm',
    'numba>=0.53.0',
    'llvmlite',
    'torch>=1.6',
    'opencv-python-headless',
    'fastremap',
    'imagecodecs',
    'roifile',
    'Pillow',
    'PyWavelets',
]

image_deps = ['nd2', 'pynrrd']

gui_deps = [
    'pyqtgraph>=0.11.0rc0', "pyqt6", "pyqt6.sip", 'qtpy', 'superqt',
    'geopandas', 'matplotlib', 'pandas', 'scikit-image', 'seaborn', 'shapely',
    'torchvision'
]

microscope_deps = ['pycromanager']
llm_deps = ['openai']

distributed_deps = [
    'dask',
    'dask_image',
    'scikit-learn',
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="cellquant", version="0.1.0", license="BSD-3-Clause", author="Chixiang Lu",
    author_email="luchixiang@gmail.com",
    description="Micro-copilot microscopy image segmentation and lysosome analysis", long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=setuptools.find_namespace_packages(include=['cellquant', 'cellquant.*']),
    install_requires=install_deps, extras_require={
        'gui': gui_deps,
        'microscope': microscope_deps,
        'llm': llm_deps,
        'image': image_deps,
        'test': ['pytest', 'pytest-cov'],
        'distributed': distributed_deps,
        'all': gui_deps + microscope_deps + llm_deps + distributed_deps + image_deps,
    }, include_package_data=True, classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ], entry_points={'console_scripts': [
        'micro-copilot = cellquant.__main__:main',
        'cellquant = cellquant.__main__:main',
    ]},
    package_data={
        'cellquant': ['logo/*.png', 'logo/*.ico', 'weights/*'],
        'cellquant.gui': ['*.html'],
    })

