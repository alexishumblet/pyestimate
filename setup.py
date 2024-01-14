from setuptools import setup

setup(
    name='estimate',
    version='0.1.0',    
    description='Various algoritms in statistical signal processing',
    url='https://github.com/alexishumblet/estimate',
    author='Alexis Humblet',
    author_email='alexishumblet@gmail.com',
    license='BSD 3-clause',
    packages=['estimate'],
    install_requires=['scipy',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
    ],
)