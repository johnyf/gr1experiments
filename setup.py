from setuptools import setup


name = 'tugs'
description = (
    'GR(1) synthesizer using Cython.')
url = 'https://github.com/johnyf/{name}'.format(name=name)
README = 'README.md'
VERSION_FILE = '{name}/_version.py'.format(name=name)
MAJOR = 0
MINOR = 1
MICRO = 1
version = '{major}.{minor}.{micro}'.format(
    major=MAJOR, minor=MINOR, micro=MICRO)
s = (
    '# This file was generated from setup.py\n'
    "version = '{version}'\n").format(version=version)
install_requires = [
    'dd >= 0.1.3',
    'omega >= 0.0.3']
tests_require = [
    'nose >= 1.3.4']


def run_setup():
    with open(VERSION_FILE, 'w') as f:
        f.write(s)
    setup(
        name=name,
        version=version,
        description=description,
        long_description=open(README).read(),
        author='Ioannis Filippidis',
        author_email='jfilippidis@gmail.com',
        url=url,
        license='BSD',
        install_requires=install_requires,
        tests_require=tests_require,
        packages=[name],
        package_dir={name: name},
        keywords=['synthesis', 'temporal logic'],
        entry_points={
            'console_scripts':
                ['tugs = tugs.solver:command_line_wrapper']})


if __name__ == '__main__':
    run_setup()
