from setuptools import setup, find_packages

setup(
    name="achilles",
    url="https://github.com/esteinig/achilles",
    author="Eike J. Steinig",
    author_email="eikejoachim.steinig@my.jcu.edu.au",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scikit-image",
        "scikit-learn",
        "h5py",
        "scipy",
        "tqdm",
        "seaborn",
        "matplotlib",
        "colorama",
        "pymongo",
        "mongoengine",
        "keras",
        "click",
        "pyyaml",
        "pytest",
        "wget",
        "watchdog",
        "paramiko",
        'numpy',
        'tqdm',
        'colorama',
        'pymongo',
        'mongoengine',
        'ont_fast5_api',
        'pandas',
        'paramiko',
        'scp',
        'scikit-image',
        'scipy',
        'watchdog',
        'apscheduler',
        'click',
        'deprecation',
        'pyfastaq'
    ],
    entry_points="""
        [console_scripts]
        achilles=achilles.terminal.client:terminal_client
        poremongo=poremongo.terminal.client:terminal_client
        pm=poremongo.terminal.client:terminal_client
    """,
    version="0.3",
    license="MIT",
    description="Achilles is a platform to train and evaluate neural networks"
    " for nanopore signal classification.",
)
