from setuptools import find_packages, setup

setup(
    name='rsna_heme',
    version='0.1.0',
    url='https://github.com/johncolby/rsna_heme',
    author='John Colby',
    author_email='john.b.colby@gmail.com',
    description='RSNA intracranial hemorrhage challenge',
    packages=['rsna_heme'],
    install_requires=[
        'gluoncv',
        'ipykernel',
        'ipython',
        'ipywidgets',
        'kaggle',
        'mxboard',
        #'mxnet',
        'opencv-python',
        'pandas',
        'pydicom',
        'sklearn'
    ],
    include_package_data=True
)
