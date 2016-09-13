import re
import numpy
import itertools
import distutils.sysconfig
from glob import glob
from setuptools import setup, find_packages, Extension
from setuptools.dist import Distribution
from Cython.Distutils import build_ext
from pkg_resources import resource_string


# remove the "-Wstrict-prototypes" compiler option (not valid for C++)
CFG_VARS = distutils.sysconfig.get_config_vars()
for key, value in CFG_VARS.items():
    if isinstance(value, basestring):
        CFG_VARS[key] = value.replace("-Wstrict-prototypes", "")


class BinaryDistribution(Distribution):
    """
    Subclass the setuptools Distribution to flip the purity flag to false.
    See http://lucumr.pocoo.org/2014/1/27/python-on-wheels/
    """
    def is_pure(self):
        return False


# dependency links
SKIP_RE = re.compile(r'^\s*(?:-\S+)\s+(.*)$')

# Regex groups: 0: URL part, 1: package name, 2: package version
EGG_RE = re.compile(r'^(git\+https?://[^#]+)(?:#egg=([a-z0-9_.]+)(?:-([a-z0-9_.-]+))?)?$')

# Regex groups: 0: URL part, 1: package name, 2: branch name
URL_RE = re.compile(r'^\s*(https?://[\w\.]+.*/([^\/]+)/archive/)([^\/]+).zip$')

# our custom way of specifying extra requirements in separate text files
EXTRAS_RE = re.compile(r'^extras\-(\w+)\-requirements\.txt$')


def parse_reqs(reqs):
    """Parse requirements.txt files into lists of requirements and dependencies
    """
    pkg_reqs = []
    dep_links = []
    for req in reqs:
        # find things like `--find-links <URL>`
        dep_link_info = SKIP_RE.match(req)
        if dep_link_info is not None:
            url = dep_link_info.group(1)
            dep_links.append(url)
            continue
        # add packages of form:
        # git+https://github.com/Livefyre/pymaptools#egg=pymaptools-0.0.3
        egg_info = EGG_RE.match(req)
        if egg_info is not None:
            url, _, _ = egg_info.group(0, 2, 3)
            # if version is None:
            #     pkg_reqs.append(egg)
            # else:
            #     pkg_reqs.append(egg + '==' + version)
            dep_links.append(url)
            continue
        # add packages of form:
        # https://github.com/escherba/matplotlib/archive/qs_fix_build.zip
        zip_info = URL_RE.match(req)
        if zip_info is not None:
            url, pkg = zip_info.group(0, 2)
            pkg_reqs.append(pkg)
            dep_links.append(url)
            continue
        pkg_reqs.append(req)
    return pkg_reqs, dep_links


def build_extras(glob_pattern):
    """Generate extras_require mapping
    """
    fnames = glob(glob_pattern)
    result = dict()
    dep_links = []
    for fname in fnames:
        extras_match = re.search(EXTRAS_RE, fname)
        if extras_match is not None:
            extras_file = extras_match.group(0)
            extras_name = extras_match.group(1)
            with open(extras_file, 'r') as fhandle:
                result[extras_name], deps = parse_reqs(fhandle.readlines())
                dep_links.extend(deps)
    return result, dep_links


INSTALL_REQUIRES, INSTALL_DEPS = parse_reqs(
    resource_string(__name__, 'requirements.txt').splitlines())
TESTS_REQUIRE, TESTS_DEPS = parse_reqs(
    resource_string(__name__, 'dev-requirements.txt').splitlines())
EXTRAS_REQUIRE, EXTRAS_DEPS = build_extras('extras-*-requirements.txt')
DEPENDENCY_LINKS = list(set(itertools.chain(
    INSTALL_DEPS,
    TESTS_DEPS,
    EXTRAS_DEPS
)))


CXXFLAGS = u"""
-O3
-msse4.2
-Wno-unused-value
-Wno-unused-function
""".split()

VERSION = '0.2.0'
URL = 'https://github.com/escherba/clustering-metrics'


setup(
    name="clustering-metrics",
    version=VERSION,
    author="Eugene Scherba",
    license="BSD",
    author_email="escherba@gmail.com",
    description=("Algorithms for locality-sensitive hashing on text data"),
    url=URL,
    download_url=URL + "/tarball/master/" + VERSION,
    packages=find_packages(exclude=['tests', 'scripts']),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    dependency_links=DEPENDENCY_LINKS,
    zip_safe=False,
    test_suite='nose.collector',
    cmdclass={'build_ext': build_ext},
    keywords=['clustering', 'evaluation', 'metrics', 'validation', 'analysis'],
    ext_modules=[
        Extension(
            "clustering_metrics.ext",
            [
                "clustering_metrics/ext.pyx",
            ],
            depends=[
            ],
            language="c++",
            extra_compile_args=CXXFLAGS,
            include_dirs=[
                "include",
            ]),
        Extension(
            "clustering_metrics.entropy",
            [
                "clustering_metrics/entropy.pyx",
                "clustering_metrics/gamma.c",
                "clustering_metrics/assignmentoptimal_dbl.c",
                "clustering_metrics/assignmentoptimal_lng.c"
            ],
            depends=[
                "include/gamma.h",
                "include/assignmentoptimal.h",
            ],
            language="c",
            extra_compile_args=CXXFLAGS,
            include_dirs=[
                numpy.get_include(),
                "include",
            ])
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 2.7',
        'Topic :: Internet',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Filters',
    ],
    long_description=resource_string(__name__, 'README.rst'),
    distclass=BinaryDistribution,
)
