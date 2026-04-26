from setuptools import setup, find_packages
import os

version = '0a'

if __name__ == '__main__':
    setup(name='star_rev',
          version=version,
          author='Jonas Káral',
          author_email='jkar0463@uni.sydney.edu.au',
          description='Some routine for JWKB mode coupling and fitting mixed mode stellar power spectra with lorentzians by minimizing the -ln(L) function',
          license='MIT',
          url='https://github.com/jonas589/Asymmetry-Reversal.git',
          packages=find_packages(),
          install_requires=['numpy', 'scipy', 'zeta @ git+https://gitlab.com/darthoctopus/zeta.git']
          )
