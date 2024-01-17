from setuptools import setup

setup(
    name='pyestimate',
    version='0.1.1',    
    description='Various algorithms in statistical signal processing',
    url='https://github.com/alexishumblet/pyestimate',
    author='Alexis Humblet',
    author_email='alexishumblet@gmail.com',
    license='BSD 3-clause',
    packages=['pyestimate'],
    install_requires=['scipy',
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
    ],
)