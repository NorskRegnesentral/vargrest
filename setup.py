from setuptools import setup, find_packages

setup(
    name='vargrest',
    version=open('vargrest/VERSION.txt').read(),
    description='Package',
    author='Norwegian Computing Center',
    packages=find_packages(),
    include_package_data=True,
)
