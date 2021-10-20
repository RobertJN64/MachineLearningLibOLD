from setuptools import setup

setup(
    name='MachineLearning',
    url='https://github.com/RobertJN64/MachineLearning',
    author='Robert Nies',
    author_email='robertjnies@gamil.com',
    # Needed to actually package something
    packages=['MachineLearning'],
    # Needed for dependencies
    install_requires=[],
    #removed pygame, not technically needed anymore
    # *strongly* suggested for sharing
    version='2.5.0',
    # The license can be anything you like
    license='MIT',
    description='Machine learning libraries that focus on transparency over speed. Most rely on genetic algs. Install PythonExtended + matplotlib for graping',
    long_description=open('README.md').read(),
    data_files=[('tests', ['tests/test.py']),
                ('tests/files', ['tests/files/titanic.json'])],
)