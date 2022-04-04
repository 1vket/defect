from setuptools import setup, find_packages

install_requires = [
  'numpy',
  'torch',
  'scipy',
  'pyopenjtalk',
  'librosa',
  'pyyaml',
  'attrdict',
]

packages = [
  'defect',
]

setup(
  name='defect',
  version='1.1',
  packages=packages,
  install_requires=install_requires,
)
