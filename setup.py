from setuptools import setup

setup(name='DrugDataLoader',
      version='0.1',
      description='drug data loader',
      url='https://github.com/kexinhuang12345/DrugDataLoader',
      author='Kexin Huang',
      author_email='kexinhuang@hsph.harvard.edu',
      license='MIT',
      packages=['DrugDataLoader'],
      zip_safe=False,
      install_requires=['numpy','pandas','wget','tqdm','fuzzywuzzy'])

