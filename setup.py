import setuptools

setuptools.setup(
      name='achilles',
      version='0.1',
      description='Chirons student. Species classification of raw nanopore signal with ANNs.',
      url='http://github.com/esteinig/achilles',
      author='Eike J. Steinig',
      author_email='eikejoachim.steinig@my.jcu.edu.au',
      license='MIT',
      packages=['asclepius', "ont_fast5_api"],
      scripts=["achilles"],
)