from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(
    name='mlsuite',
    version='0.0.1',
    description='Machine Learning Suite',
    long_description=readme,
    author='Daniel Morales',
    author_email='dnlmrls9@gmail.com',
    url='https://github.com/DanielMorales9/MLSuite',
    install_requires=['numpy', 'matplotlib', 'scipy']
)
