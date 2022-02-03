"""MLCommons Algorithmic Efficiency.

For a Jax GPU install:
pip install -e .[jax-gpu] -f https://storage.googleapis.com/jax-releases/jax_releases.html'
"""
from setuptools import find_packages
from setuptools import setup


jax_core_deps = [
    'flax==0.3.5',
    'optax==0.0.9',
    'tensorflow_datasets==4.4.0',
    'tensorflow-cpu==2.5.0',
]


setup(
    name='algorithmic_efficiency',
    version='0.0.1',
    description='MLCommons Algorithmic Efficiency',
    author='MLCommons Algorithmic Efficiency Working Group',
    author_email='algorithms@mlcommons.org',
    url='https://github.com/mlcommons/algorithmic-efficiency',
    license='Apache 2.0',
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=[
        'absl-py==0.14.0',
        'numpy>=1.19.2',
        'pandas>=1.3.1',
        'ctcdecode>=1.0.2',
        'levenshtein>=0.12.0',
        'librosa>=0.8.1',
        'six>=1.15.0',
    ],
    extras_require={
        'jax-cpu': jax_core_deps + ['jax==0.2.28', 'jaxlib==0.1.76'],
        # Note for GPU support the installer must be run with
        # `-f 'https://storage.googleapis.com/jax-releases/jax_releases.html'`.
        'jax-gpu': jax_core_deps + ['jax[cuda]==0.2.28', 'jaxlib==0.1.76+cuda111.cudnn82'],
        'pytorch': [
            'torch==1.9.1+cu111',
            'torchvision==0.10.1+cu111',
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
