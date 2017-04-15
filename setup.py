from distutils.core import setup

if __name__== '__main__':
    setup(include_package_data=True,
          description='Interpretable deep learning',
          url='NA',
          download_url='NA',
          version='0.5.0-tensorflow',
          packages=['deeplift',
                    'deeplift.blobs', 'deeplift.visualization',
                    'deeplift.conversion'],
          setup_requires=[],
          install_requires=['numpy>=1.9', 'tensorflow>=1.0.1'],
          scripts=[],
          name='deeplift')
