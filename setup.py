"""Setup file for algorithmic_efficiency, use setup.cfg for configuration."""

from setuptools import setup

<<<<<<< HEAD

jax_core_deps = [
    'flax==0.3.5',
    'optax==0.0.9',
    'tensorflow_datasets==4.4.0',
    'tensorflow==2.5.0',
]


# Assumes CUDA 11.x and a compatible NVIDIA driver and CuDNN.
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
        'clu==0.0.6',
        'jraph==0.0.2.dev',
        'numpy>=1.19.2',
        'pandas==1.3.4',
        'scikit-learn==1.0.1',
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
=======
if __name__ == "__main__":
  setup()
>>>>>>> main
