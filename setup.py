from distutils.core import setup

if __name__== '__main__':
    setup(include_package_data=True,
          description='DeepLIFT (Deep Learning Important FeaTures)',
          long_description="""Algorithms for computing importance scores in deep neural networks.

Implements the methods in "Learning Important Features Through Propagating Activation Differences" by Shrikumar, Greenside & Kundaje, as well as other commonly-used methods such as gradients, guided backprop and integrated gradients. See https://github.com/kundajelab/deeplift for documentation and FAQ.
          """,
          url='https://github.com/kundajelab/deeplift',
          version='0.6.13.0',
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
