from setuptools import setup, find_packages

setup(
    name="achilles",
    url="https://github.com/esteinig/achilles",
    author="Eike J. Steinig",
    author_email="eikejoachim.steinig@my.jcu.edu.au",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "keras",
        "pandas",
        "numpy",
        "scipy",
        "scikit-image",
        "scikit-learn",
        "h5py",
        "tqdm",
        "seaborn",
        "matplotlib",
        "colorama",
        "mongoengine",
        "click",
        "pyyaml",
        "pytest",
        "wget",
    ],
    entry_points="""
        [console_scripts]
        achilles=achilles.terminal.client:terminal_client
    """,
    version="0.3",
    license="MIT",
    description="Achilles is a platform to train and evaluate neural networks"
    " for nanopore signal classification.",
)
