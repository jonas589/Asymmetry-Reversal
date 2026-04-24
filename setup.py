from setuptools import setup, find_packages
import os

version = '0a'

if __name__ == '__main__':
    setup(name='star_rev',
          version=version,
          author='Jonas Káral',
          author_email='jkar0463@uni.sydney.edu.au',
          description='Some routine for JWKB mode coupling and stretched echelle diagrams',
          license='MIT',
          url='https://gitlab.com/darthoctopus/zeta.git',
          packages=find_packages(),
          install_requires=['numpy', 'scipy', 'zeta', 'nifty-ls']
          )
