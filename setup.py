from setuptools import setup

setup(
    name='MachineLearning',
    url='https://github.com/RobertJN64/MachineLearning',
    author='Robert Nies',
    author_email='robertjnie@gamil.com',
    # Needed to actually package something
    packages=['MachineLearning'],
    # Needed for dependencies
    install_requires=['pygame'],
    # *strongly* suggested for sharing
    version='2.0',
    # The license can be anything you like
    license='MIT',
    description='Machine learning libraries that focus on transparency over speed. Most rely on genetic algs.',
    long_description=open('README.txt').read(),
    scripts = ['scripts/test.py', 'scripts/files/titanic.json']
)