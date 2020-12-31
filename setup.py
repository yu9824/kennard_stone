from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

setup(
    name='kennard_stone',
    version='0.0.3',
    description='',
    author='yu-9824',
    author_email='yu.9824@gmail.com',
    install_requires=install_requirements,
    url='https://github.com/yu-9824/kennard_stone',
    license=license,
    packages=find_packages(exclude=['example'])
)
