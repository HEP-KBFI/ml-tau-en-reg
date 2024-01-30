from setuptools import setup, find_packages

setup(
    name='enreg',
    version='0.0.1',
    description='Software for tau_h energy regression',
    url='',
    author=['Laurits Tani'],
    author_email='laurits.tani@cern.ch',
    license='GPLv3',
    packages=find_packages(),
    package_data={
        'enreg': [
            'config/*',
            'tests/*',
            'scripts/*'
        ]
    },
    install_requires=[
        'awkward',
        'fastjet',
        'hydra-core',
        'matplotlib',
        'mplhep',
        'numba',
        'pandas',
        'pyarrow',
        'scikit-learn',
        'scikit-optimize',
        'scipy',
        'seaborn',
        'setGPU',
        'tensorboard',
        'tqdm',
        'uproot',
        'vector',
    ],
)
