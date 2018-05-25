import sys
sys.path.append("/usr/local/lib/python3.5/dist-packages/")
if sys.version_info < (3,5):
    sys.exit ('Sorry, you should run this on Python > 3.5')

from setuptools import setup

setup(
    name='reliefweb_tag',
    version='0.1',
    packages=['reliefweb_tag'],
    url='https://github.com/reliefweb/reliefweb-tag-assistant/',
    license='ReliefWeb',
    author='Miguel Hernandez',
    author_email='hernandez@reliefweb.int',
    description='ReliefWeb Tag Assistant',
    long_description='Tag urls using ReliefWeb tags',
    keywords='reliefweb humanitarian updates news articles neural classification tagging multitagging',
    zip_safe=False,
    install_requires=[
        'numpy',
        'pandas',
        'sklearn',
        'keras',
        'tensorflow',
	'cucco',
	'flask',
    ],
    python_requires='>3.5.0',
    include_package_data=True
)
