import argparse
import re
import json
import collections
import pickle
import joblib
import os
import codecs
import logging
from gzip import open as gzip_open
from bz2 import BZ2File
from zipfile import ZipFile
from pkg_resources import resource_listdir, resource_filename
from fnmatch import fnmatch
from pymaptools.inspect import hasmethod
from pymaptools.utils import joint_context
from pymaptools.iter import isiterable


SUPPORTED_EXTENSION = re.compile(ur'(\.(?:gz|bz2|zip))$', re.IGNORECASE)


def get_extension(fname, regex=SUPPORTED_EXTENSION, lowercase=True):
    """Return a string containing its extension (if matches pattern)
    """
    match = regex.search(fname)
    if match is None:
        return None
    elif lowercase:
        return match.group().lower()
    else:
        return match.group()


def walk_files(dirname, file_pattern=u'*'):
    """Recursively walk through directory tree and find matching files
    """
    for root, _, files in os.walk(dirname):
        for name in files:
            full_name = os.path.join(root, name)
            if fnmatch(full_name, file_pattern):
                yield full_name


def read_json_lines(finput, logger=logging, encoding='utf-8'):
    ctx = joint_context(finput) \
        if isiterable(finput) \
        else open_gz(finput, 'r')
    with ctx as fhandle:
        for idx, line in enumerate(fhandle, start=1):
            try:
                obj = json.loads(line, encoding=encoding)
            except ValueError as err:
                logger.error("Could not parse line %d: %s", idx, err)
                continue
            yield obj


def ndjson2col(iterator):
    """Convert line-delimited JSON input to Pandas-compatible column dict
    """
    result = collections.defaultdict(list)
    try:
        obj = iterator.next()
    except StopIteration:
        return result
    fields = frozenset(obj.iterkeys())
    for field in fields:
        result[field].append(obj[field])
    for idx, obj in enumerate(iterator, start=2):
        missing_fields = fields - frozenset(obj.iterkeys())
        if missing_fields:
            raise RuntimeError("Missing fields %s at line %d" %
                               (list(missing_fields), idx))
        for field in fields:
            result[field].append(obj[field])
    return result


FILEOPEN_FUNCTIONS = {
    '.gz': lambda fname, mode='r', compresslevel=9: gzip_open(fname, mode + 'b', compresslevel),
    '.bz2': lambda fname, mode='r', compresslevel=9: BZ2File(fname, mode, compresslevel),
    '.zip': lambda fname, mode='r', compresslevel=None: ZipFile(fname, mode).open(os.path.basename(fname)[0:-4])
}


def open_gz(fname, mode='r', compresslevel=9):
    """Transparent substitute to open() for gzip, bz2 support

    If extension was not found or is not supported, assume it's a plain-text file
    """
    extension = get_extension(fname)
    if extension is None:
        return open(fname, mode)
    else:
        fopen_fun = FILEOPEN_FUNCTIONS[extension]
        return fopen_fun(fname, mode, compresslevel)


def parse_json(line):
    """Safe wrapper around json.loads"""
    try:
        return json.loads(line)
    except ValueError:
        return None


def default_encode_json(obj):
    """Default object encoder to JSON

    Warning: one ought to define one's own decoder for one's own
    object; this method works in many cases but may not be correct.
    """
    return obj.__dict__


def write_json_line(handle, obj, default=default_encode_json, **kwargs):
    """write a line encoding a JSON object to a file handle"""
    handle.write(u"%s\n" % json.dumps(obj, default=default, **kwargs))


class FileReader(collections.Iterator):
    """Read files sequentially and return lines from each

    This is basically a quirky reimplementation of FileInput

    ::

        >>> reader = FileReader([[1, 2, 3], [4, 5], [], [6, 7]])
        >>> list(reader)
        [1, 2, 3, 4, 5, 6, 7]
        >>> reader = FileReader([[], []])
        >>> list(reader)
        []
        >>> reader = FileReader([])
        >>> list(reader)
        []
    """
    def __init__(self, files, mode='r', openhook=None):
        self._files = iter(files)
        self._curr_handle = None
        self._curr_filename = None
        self._openhook = openhook
        self._mode = mode
        if files:
            self._advance_handle()

    def _advance_handle(self):
        self._curr_filename = self._files.next()
        if self._openhook is None:
            self._curr_handle = self._curr_filename \
                if hasmethod(self._curr_filename, 'next') \
                else iter(self._curr_filename)
        else:
            curr_handle = self._curr_handle
            if hasmethod(curr_handle, 'close'):
                curr_handle.close()
            self._curr_handle = self._openhook(self._curr_filename, self._mode)

    def __iter__(self):
        return self

    def parse(self, line):
        """Override this method if you want a custom parser"""
        return line

    def next(self):
        line = None
        if self._curr_handle is None:
            raise StopIteration()
        while line is None:
            try:
                line = self._curr_handle.next()
            except StopIteration:
                self._advance_handle()
        return self.parse(line)


class JSONLineReader(FileReader):
    """A subclass of FileReader specialized for JSON line input
    """
    def parse(self, line):
        return parse_json(line)


def read_text_resource(finput, encoding='utf-8', ignore_prefix='#'):
    """Read a text resource ignoring comments beginning with pound sign
    :param finput: path or file handle
    :type finput: str, file
    :param encoding: which encoding to use (default: UTF-8)
    :type encoding: str
    :param ignore_prefix: lines matching this prefix will be skipped
    :type ignore_prefix: str, unicode
    :rtype: generator
    """
    ctx = joint_context(codecs.iterdecode(finput, encoding=encoding)) \
        if isiterable(finput) \
        else codecs.open(finput, 'r', encoding=encoding)
    with ctx as fhandle:
        for line in fhandle:
            if ignore_prefix is not None:
                line = line.split(ignore_prefix)[0]
            line = line.strip()
            if line:
                yield line


def write_text_resource(foutput, text, encoding='utf-8'):
    """Write a text resource
    :param foutput: path or file handle
    :type foutput: str, file
    :param text: content to write
    :type text: str, unicode, iterable
    :param encoding: which encoding to use (default: UTF-8)
    :type encoding: str
    """
    if isinstance(foutput, file):
        for chunk in codecs.iterencode(text, encoding=encoding):
            foutput.write(chunk)
    else:
        with codecs.open(foutput, 'w', encoding=encoding) as fhandle:
            if isiterable(text):
                for line in text:
                    fhandle.write(u"%s\n" % line)
            else:
                fhandle.write(text)


def read_tsv_like(finput, encoding='utf-8', ignore_prefix='%', sep=' '):
    """Read TSV-like format
    """
    for line in read_text_resource(finput, encoding=encoding, ignore_prefix=ignore_prefix):
        yield line.split(sep)


def write_tsv_like(foutput, records, encoding='utf-8', sep=' '):
    """Write TSV-like format
    """
    unicode_sep = sep.decode(encoding)
    text = (unicode_sep.join(record) for record in records)
    write_text_resource(foutput, text, encoding=encoding)


class GzipFileType(argparse.FileType):
    """Same as argparse.FileType except works transparently with files
    ending with .gz extension
    """

    def __init__(self, mode='r', bufsize=-1, compresslevel=9,
                 name_pattern=SUPPORTED_EXTENSION):
        super(GzipFileType, self).__init__(mode, bufsize)
        self._compresslevel = compresslevel
        self._name_pattern = name_pattern

    def __call__(self, string):
        extension = get_extension(string, regex=self._name_pattern)
        if extension is None:
            return super(GzipFileType, self).__call__(string)
        else:
            fopen_fun = FILEOPEN_FUNCTIONS[extension]
            try:
                return fopen_fun(string, self._mode, self._compresslevel)
            except OSError as err:
                raise argparse.ArgumentTypeError(
                    "can't open '%s': %s" % (string, err))


class PathArgumentParser(argparse.ArgumentParser):

    """
    Supplement argparse.ArgumentParser with a method that checks for
    valid filesystem paths
    """

    def __is_valid_file(self, arg):
        if os.path.isfile(arg):
            return arg
        else:
            self.error("Path '%s' does not correspond to existing file" % arg)

    def __is_valid_directory(self, arg):
        if os.path.isdir(arg):
            return arg
        else:
            self.error("Path '%s' does not correspond to existing directory" % arg)

    def add_argument(self, *args, **kwargs):
        if 'metavar' in kwargs and ('type' not in kwargs or kwargs['type'] == str):
            metavar = kwargs['metavar']
            if metavar == 'FILE':
                kwargs['type'] = self.__is_valid_file
            elif metavar == 'DIR':
                kwargs['type'] = self.__is_valid_directory
            else:
                raise ValueError("Invalid metavar parameter value '%s'" % metavar)
        super(PathArgumentParser, self).add_argument(*args, **kwargs)


class DumperFacade(object):
    """Provides a consistent interface to dumper objects"""

    _dump_types = {
        'pickle': dict(load=pickle.load, dump='_dump_pickle'),
        'joblib': dict(load=joblib.load, dump='_dump_joblib')
    }

    def __init__(self, dumper_type='joblib'):
        dumper_props = self._dump_types[dumper_type]
        self.load = dumper_props['load']
        self.dump = getattr(self, dumper_props['dump'])

    @classmethod
    def keys(cls):
        return cls._dump_types.keys()

    @staticmethod
    def _dump_joblib(estimator, fname):
        joblib.dump(estimator, fname, compress=9)

    @staticmethod
    def _dump_pickle(estimator, fname):
        with open(fname, 'wb') as fhandle:
            pickle.dump(estimator, fhandle)


class SimplePicklableMixin(object):
    def save_to(self, filename):
        with open_gz(filename, "wb") as fhandle:
            pickle.dump(self, fhandle)

    @classmethod
    def load_from(cls, filename):
        with open_gz(filename, "rb") as fhandle:
            obj = pickle.load(fhandle)
            if not isinstance(obj, cls):
                raise TypeError("Loaded object not of expected type %s", cls.__name__)
            return obj


def pickle_dump(obj, output_file, protocol=2):
    """Pickle object `obj` to file `output_file`.
    `protocol` defaults to 2 so pickled objects are compatible across
    Python 2.x and 3.x.
    """
    if isinstance(output_file, basestring):
        with open_gz(output_file, 'wb') as fhandle:
            pickle.dump(obj, fhandle, protocol=protocol)
    else:
        pickle.dump(obj, output_file, protocol=protocol)


def pickle_load(input_file):
    """Load pickled object from `input_file`"""
    if isinstance(input_file, basestring):
        with open_gz(input_file, 'rb') as fhandle:
            return pickle.load(fhandle)
    else:
        return pickle.load(input_file)


class ResourceBundle(object):

    """Find and present resource paths in a convenient format

    Usage::

        >>> MY_RESOURCES = ResourceBundle(__name__, '.', '.py')
        >>> os.path.basename(MY_RESOURCES.io)
        'io.py'
    """
    @classmethod
    def list_matching_files(cls, package_or_requirement, resource_dir, extension):
        resource_pattern = "*" + extension
        dirname = resource_filename(package_or_requirement, resource_dir)
        for filename in resource_listdir(package_or_requirement, resource_dir):
            if fnmatch(filename, resource_pattern):
                bname = os.path.basename(filename)
                rname, _ = os.path.splitext(bname)
                rname = re.sub("\\W", "_", rname)
                yield rname, os.path.join(dirname, filename)

    def __init__(self, package_or_requirement, resource_dir, extension=".txt"):
        for rname, fname in self.list_matching_files(package_or_requirement, resource_dir, extension):
            self.__dict__[rname] = fname
