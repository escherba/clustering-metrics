import unicodecsv as csv
import argparse
import sys
from pymaptools.iter import field_getter
from pymaptools.io import GzipFileType


def parse_args(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--fields', nargs='*', default=None)
    ap.add_argument('--input_delimiter', default='\t', help='input delimiter')
    ap.add_argument('--output_delimiter', default=',', help='output delimiter')
    ap.add_argument('--output_header', action='store_true')
    ap.add_argument('--input', type=GzipFileType('r'), default=sys.stdin)
    ap.add_argument('--output', type=GzipFileType('w'), default=sys.stdout)
    namespace = ap.parse_args(args)
    return namespace


def run(args):
    reader = csv.reader(args.input, delimiter=args.input_delimiter)
    writer = csv.writer(args.output, delimiter=args.output_delimiter)
    fields = args.fields
    header = reader.next()
    get = field_getter(header, fields)
    if args.output_header:
        writer.writerow(get(header))
    for row in reader:
        writer.writerow(get(row))


if __name__ == "__main__":
    run(parse_args())
