"""
Utility for splitting data files
"""
import os
import sys
import glob
import argparse
from tqdm import tqdm
from pymaptools.iter import isiterable
from pymaptools.io import GzipFileType


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=GzipFileType('r'), default=sys.stdin,
                        help='input file')
    parser.add_argument("--num_splits", type=int, default=50,
                        help='Number of splits')
    parser.add_argument("--output", type=str, required=True,
                        help="output directory")
    parser.add_argument("--show_progress", action='store_true',
                        help='show progress bar')
    parser.add_argument("--overwrite", action='store_true',
                        help='overwrite any existing files')
    namespace = parser.parse_args(args)
    return namespace


def split_or_whole(dirname):
    if not isiterable(dirname):
        if not os.path.exists(dirname):
            raise ValueError(u"Specified path does not exist: {}".format(dirname))
        pattern = os.path.join(dirname, 'part-*')
        fnames = glob.glob(pattern)
    else:
        fnames = dirname
    return fnames


def write_split(row_iter, dirname, total=None, show_progress=True,
                overwrite=False, num_splits=100, has_enum=False):
    """Save row iterator as split CSV
    """
    if os.path.exists(dirname):
        if not overwrite:
            raise IOError("path {} already exists".format(dirname))
        # delete contents
        fnames = split_or_whole(dirname)
        for fname in fnames:
            os.remove(fname)
        fname = os.path.join(dirname, '_SUCCESS')
        if os.path.exists(fname):
            os.remove(fname)
    else:
        os.mkdir(dirname)

    # ensure that rows are enumerated
    if not has_enum:
        row_iter = enumerate(row_iter)

    if show_progress:
        row_iter = tqdm(row_iter, total=total)

    fnames = [os.path.join(dirname, 'part-%05d' % i) for i in range(num_splits)]
    fhandles = [open(fn, 'w') for fn in fnames]

    # write lines
    [fhandles[i % num_splits].write(row) for i, row in row_iter]

    # close files
    [fh.close() for fh in fhandles]

    # let outside world know that we are done
    fname = os.path.join(dirname, '_SUCCESS')
    os.mknod(fname)


def run(args):
    write_split(args.input, args.output, show_progress=args.show_progress,
                overwrite=args.overwrite, num_splits=args.num_splits, has_enum=False)


if __name__ == "__main__":
    run(parse_args())
