from setuptools import setup, find_packages

setup(
    name='vargrest',
    version=open('vargrest/VERSION.txt').read(),
    description='Variogram estimation for ResQml models converted by nrresqml',
    author='Norwegian Computing Center',
    license='GPL-3.0',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/NorskRegnesentral/vargrest',
    packages=find_packages(),
    package_data={
        'vargrest': ['VERSION.txt']
    },
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
