from distutils.core import setup

if __name__== '__main__':
    setup(include_package_data=True,
          description='DeepLIFT (Deep Learning Important FeaTures)',
          url='NA',
          download_url='NA',
          version='0.6',
          packages=['deeplift',
                    'deeplift.layers', 'deeplift.visualization',
                    'deeplift.conversion'],
          setup_requires=[],
          install_requires=['numpy>=1.9', 'tensorflow>=1.7'],
          scripts=[],
          name='deeplift')
