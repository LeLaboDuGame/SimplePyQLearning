import setuptools
from setuptools import setup

setup(
    name='simplepyqlearning',
    version='1.1.0',
    packages=setuptools.find_packages(),
    url='https://github.com/AdrienDumontet/SimplePyAI',
    license='Let the package like is it',
    author_email='',
    author='LeLaboDuGame, https://twitch.tv/LeLaboDuGame',
    description='A simple python lib to do QLearning',
    install_requires=[
        "numpy",
        "tqdm",
    ]
)
