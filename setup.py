from distutils.core import setup, Extension
from setuptools import setup, Extension

config = {
    'include_package_data': True,
    'description': 'DeepLIFT',
    'url': 'NA',
    'download_url': 'https://github.com/kundajelab/deeplift',
    'version': '0.2',
    'packages': ['deeplift'],
    'setup_requires': [],
    'install_requires': ['numpy>=1.9', 'keras==0.3.2', 'theano>=0.8'],
    'scripts': [],
    'name': 'DeepLIFT'
}

if __name__== '__main__':
    setup(**config)
