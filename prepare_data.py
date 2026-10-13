"""Explicit local symbol-dataset export/validation; never trains or uploads."""
import argparse
import json

from src.data import export_existing, validate_export


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', help='Explicit source directory for optional annotated-data export')
    parser.add_argument('--out', default='dataset_symbols_small')
    parser.add_argument('--limit', type=int, default=2000)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.limit < 100:
        parser.error('Use at least 100 samples')
    if not args.validate_only and args.data is None:
        parser.error('Use --validate-only for the included dataset, or specify --data for an export')
    report = validate_export(args.out) if args.validate_only else export_existing(args.data, args.out, args.limit)
    print(json.dumps(report, ensure_ascii=True, indent=2))
