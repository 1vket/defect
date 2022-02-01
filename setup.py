from setuptools import setup, find_packages

install_requires = [
  'numpy',
  'torch',
  'scipy',
  'pyopenjtalk',
  'librosa',
]

packages = [
  'nwskwii',
]

setup(
  name='nwskwii',
  version='0.1',
  packages=packages,
  install_requires=install_requires,
)
