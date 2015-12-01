from setuptools import setup
# inline
# import git


name = 'tugs'
description = (
    'GR(1) synthesizer using Cython.')
url = 'https://github.com/johnyf/{name}'.format(name=name)
README = 'README.md'
VERSION_FILE = '{name}/_version.py'.format(name=name)
MAJOR = 0
MINOR = 1
MICRO = 1
VERSION = '{major}.{minor}.{micro}'.format(
    major=MAJOR, minor=MINOR, micro=MICRO)
VERSION_TEXT = (
    '# This file was generated from setup.py\n'
    "version = '{version}'\n")
install_requires = [
    'dd >= 0.1.3',
    'omega >= 0.0.3']
tests_require = [
    'nose >= 1.3.4']


def git_version(version):
    try:
        import git
        repo = git.Repo('.git')
        repo.git.status()
    except ImportError, git.GitCommandNotFound:
        print('gitpython or git not found: Assume release.')
        return ''
    sha = repo.head.commit.hexsha
    if repo.is_dirty():
        return '.dev0+{sha}-dirty'.format(sha=sha)
    # commit is clean
    # is it release of `version` ?
    try:
        tag = repo.git.describe(
            match='v[0-9]*', exact_match=True,
            tags=True, dirty=True)
        assert tag[1:] == version, (tag, version)
        return ''
    except git.GitCommandError:
        return '.dev0+{sha}'.format(sha=sha)


def run_setup():
    # version
    version = VERSION + git_version(VERSION)
    s = VERSION_TEXT.format(version=version)
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
