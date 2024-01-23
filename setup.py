from setuptools import setup

setup(
    name='pyestimate',
    version='0.3.1',    
    description='Sinusoidal parameters estimators (1D, 2D, single or multiple sinusoids, ...)',
    url='https://github.com/alexishumblet/pyestimate',
    author='Alexis Humblet',
    author_email='alexishumblet@gmail.com',
    license='BSD 3-clause',
    packages=['pyestimate'],
    install_requires=['scipy',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
    ],
)