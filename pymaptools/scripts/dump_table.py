"""
Dumps MySQL tables using `mysqldump` into CSV-compatible format
without locking the table.

Typical usage:

    python src/scripts/dump_table.py --table like_events \\
        --fields id user_id entity_id entity_type ts created_at \\
        > /path/outfile
"""
import errno
import sys
import os
import re
import argparse
import subprocess
import unicodecsv as csv


_RE_ALL = re.compile(ur'^\s*INSERT INTO (\w+) \(([^\(\)]*)\) VALUES \((.*)\);\s*$', re.UNICODE)
_RE_NAME = re.compile(ur'\w+', re.UNICODE)


DELIMITERS = {
    'tab': '\t',
    'comma': ','
}


def parse_args(args=None):

    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    ap.add_argument('--table', type=str, required=True, help='table to use')
    ap.add_argument(
        '--fields', type=str, nargs='*', default=None,
        help='limit to these table fields (default: all)')
    ap.add_argument(
        '--output', type=argparse.FileType('w'), default=sys.stdout,
        help='output file (tab-delimited)')
    ap.add_argument(
        '--charset', type=str, default='utf8', help='default character set')
    ap.add_argument(
        '--no_header', action='store_true', help='do not output header')
    ap.add_argument('--host', type=str, default=os.environ.get('AURORA_HOST_READONLY', 'localhost'))
    ap.add_argument('--port', type=int, default=os.environ.get('AURORA_PORT'))
    ap.add_argument('--user', type=str, default=os.environ.get('AURORA_USER'))
    ap.add_argument('--pwd',  type=str, default=os.environ.get('AURORA_PASS'))
    ap.add_argument('--db',   type=str, default=os.environ.get('AURORA_DB'))
    ap.add_argument('--single_transaction', action='store_true',
                    help='whether to use --single-transaction in case of mysqldump')
    ap.add_argument('--delimiter', type=str, default='tab', choices=DELIMITERS.keys(),
                    help='Delimiter to use for writing output')
    ap.add_argument('--cmd', type=str, default='mysql', choices=['mysqldump', 'mysql'],
                    help='MySQL command to use')
    namespace = ap.parse_args(args)
    return namespace


def unescape_row(row):
    unescaped = []
    for v in row:
        if isinstance(v, basestring):
            unescaped.append(v.replace('\\', ''))
        else:
            unescaped.append(v)
    return unescaped


class NullClass:
    def __str__(self):
        return "NULL"

    def __repr__(self):
        return "NULL"


NULL = NullClass()


def write_mysqldump(args, stream):
    delimiter = DELIMITERS[args.delimiter]
    writer = csv.writer(args.output, delimiter=delimiter,
                        escapechar='\\', quoting=csv.QUOTE_NONE)
    fields = args.fields
    for idx, line in enumerate(iter(stream.readline, '')):
        match = _RE_ALL.match(line)
        table, column_sg, value_sg = match.groups()
        values = eval(value_sg)
        columns = _RE_NAME.findall(column_sg)
        assert(len(values) == len(columns))
        if fields is not None:
            record = dict(zip(columns, values))
            columns = fields
            values = [record[c] for c in columns]
        if idx == 0 and not args.no_header:
            writer.writerow(columns)
        writer.writerow(unescape_row(values))


def write_mysql(args, stream):
    fh = args.output
    for idx, line in enumerate(iter(stream.readline, '')):
        fh.write(line)


def write_stream(args, stream):
    if args.cmd == 'mysqldump':
        write_mysqldump(args, stream)
    elif args.cmd == 'mysql':
        write_mysql(args, stream)
    else:
        raise ValueError(args.cmd)


def create_cmd(args):
    d = {}
    if args.cmd == 'mysqldump':
        template = """
        mysqldump --quick --compress --extended-insert=false --complete-insert=true
        %(single_transaction)s --skip-lock-tables --default-character-set=%(charset)s
        --skip-add-drop-table --skip-add-locks --skip-comments --skip-triggers
        --skip-quote-names --compact --no-create-info
        %(host)s %(port)s %(user)s %(password)s %(database)s %(table)s
        """
        d['charset'] = args.charset
        d['single_transaction'] = '--single-transaction' if args.single_transaction else ''
        d['host'] = '--host=%s' % args.host if args.host else ''
        d['port'] = '--port=%s' % args.port if args.port else ''
        d['user'] = '--user=%s' % args.user if args.user else ''
        d['password'] = '--password=%s' % args.pwd if args.pwd else ''
        d['database'] = '--databases %s' % args.db if args.db else ''
        d['table'] = '--tables %s' % args.table if args.table else ''
        cmd = (template.strip() % d).split()
    elif args.cmd == 'mysql':
        template = """
            mysql\t-h%(host)s\t-P%(port)s\t-u\t%(user)s\t-p%(password)s\t--batch\t-q\t-e\tselect %(fields)s from %(table)s;\t%(database)s
        """
        d['host'] = args.host
        d['port'] = args.port
        d['user'] = args.user
        d['password'] = args.pwd
        d['fields'] = ', '.join(args.fields) if args.fields else '*'
        d['table'] = args.table
        d['database'] = args.db
        cmd = (template.strip() % d).split('\t')
    else:
        raise ValueError(args.cmd)
    return cmd


def run(args):
    if not args.db:
        sys.stderr.write("You did not specify database name\n")
        sys.exit(1)

    cmd = create_cmd(args)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        write_stream(args, proc.stdout)
    except IOError as err:
        proc.kill()
        if err.errno != errno.EPIPE:
            raise
    except KeyboardInterrupt:
        proc.kill()


if __name__ == "__main__":
    run(parse_args())
