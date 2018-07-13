from distutils.core import setup

if __name__== '__main__':
    setup(include_package_data=True,
          description='DeepLIFT (Deep Learning Important FeaTures)',
          url='https://github.com/kundajelab/deeplift',
          version='0.6.5',
          packages=['deeplift',
                    'deeplift.layers', 'deeplift.visualization',
                    'deeplift.conversion'],
          setup_requires=[],
          install_requires=['numpy>=1.9'],
          extras_require={
            'tensorflow': ['tensorflow>=1.7'],
            'tensorflow with gpu': ['tensorflow-gpu>=1.7']},
          scripts=[],
          name='deeplift')
