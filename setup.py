import sys

from setuptools import setup

sys.path.append("/usr/local/lib/python3.5/dist-packages/")
if (sys.version_info.major, sys.version_info.minor) != (3, 6):
    sys.exit('Sorry, you should run this on Python 3.6')

setup(
    name='reliefweb_tag',
    version='0.1',
    packages=['reliefweb_tag'],
    url='https://github.com/reliefweb/reliefweb-tag-assistant/',
    license='ReliefWeb',
    author='Miguel Hernandez',
    author_email='hernandez@reliefweb.int',
    description='ReliefWeb Tag Assistant',
    long_description=open('README.md').read(),
    keywords='reliefweb humanitarian updates news articles neural classification tagging multitagging',
    zip_safe=False,
    install_requires=[
        # see requirements.txt
    ],
    python_requires='>3, <3.7',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Natural Language :: English',
        'Intended Audience :: Developers',
    ],
)
