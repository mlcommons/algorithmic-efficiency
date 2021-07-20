"""MLCommons Algorithmic Efficiency.

For a Jax GPU install:
pip install -e .[jax-gpu] -f https://storage.googleapis.com/jax-releases/jax_releases.html'
"""
from setuptools import find_packages
from setuptools import setup


jax_core_deps = [
    'jax',
    'flax',
    'optax',
    'tensorflow-cpu',
    'tensorflow_datasets',
]


setup(
    name='algorithmic_efficiency',
    version='0.0.1',
    description='MLCommons Algorithmic Efficiency',
    author='MLCommons Algorithmic Efficiency Working Group',
    author_email='algorithms@mlcommons.org',
    url='https://github.com/mlcommons/algorithmic-efficiency',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
        'absl-py',
        'numpy',
    ],
    extras_require={
        'jax-cpu': jax_core_deps + ['jaxlib'],
        # Note for GPU support the installer must be run with
        # `-f 'https://storage.googleapis.com/jax-releases/jax_releases.html'`.
        'jax-gpu': jax_core_deps + ['jaxlib==0.1.65+cuda110'],
        'pytorch': [
            'torch',
            'torchvision',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='mlcommons algorithmic efficiency',
)
