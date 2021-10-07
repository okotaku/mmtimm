from setuptools import find_packages, setup


def take_package_name(name):
    if name.startswith('-e'):
        return name[name.find('=') + 1:name.rfind('-')]
    else:
        return name.strip()


def _requires_from_file(filename):
    return [take_package_name(l_) for l_ in open(filename).read().splitlines()]


def load_links_from_file(filepath):
    res = []
    with open(filepath) as fp:
        for pkg_name in fp.readlines():
            if pkg_name.startswith('-e'):
                res.append(pkg_name.split(' ')[1])
    return res


exec(open('timmextension/version.py').read())
setup(
    name='timmextension',
    version=__version__,  # noqa
    description='backbones base on timm',
    author='Takumi Okoshi',
    dependency_links=load_links_from_file('requirements.txt'),
    install_requires=_requires_from_file('requirements.txt'),
    packages=find_packages())
