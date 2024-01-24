from distutils.core import setup

_dependencies = [
    'cycler==0.10.0',
    'kiwisolver==1.0.1',
    'matplotlib==3.5.2',
    'numpy==1.22.4',
    'pandas==1.4.2',
    'pyparsing==3.0.9',
    'python-dateutil==2.8.1',
    'pytz==2022.1',
    'scikit-learn==1.1.1',
    'scipy==1.8.1',
    'seaborn==0.9.0',
    'six==1.11.0',
    'tqdm==4.25.0']

setup(
    name='Tesseract',
    version='0.9',
    description='Tesseract: A library for performing '
                'time-aware classifications.',
    maintainer='Feargus Pendlebury',
    maintainer_email='Feargus.Pendlebury[at]rhul.ac.uk',
    url='',
    packages=['tesseract'],
    setup_requires=_dependencies,
    install_requires=_dependencies
)
