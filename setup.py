from setuptools import setup



# Version Numbers:
# x.y.z.w
# w: Minor changes; Does not impact API.
# z: Minor changes: Backwards compatible.
# y: Semi-minor changes: Not all parts backwards compatible, you probably will not need to change your code.
# x: Major changes: Upgrades will likely require changes to your code.

setup(name='EvoSoroCore',
      author='David Matthews',
      version='0.1.0.0',
      packages=['evosorocore'],
      description='Evolution toolkit for soft robots',
      install_requires=['numpy', 'scipy', 'networkx', 'lxml']
      )
