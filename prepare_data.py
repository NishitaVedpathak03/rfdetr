"""Explicit local symbol-dataset export/validation; never trains or uploads."""
import argparse
import json

from src.data import export_existing, validate_export, pad_symbol_boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', help='Explicit source directory for optional annotated-data export')
    parser.add_argument('--out', default='dataset_symbols_small')
    parser.add_argument('--limit', type=int, default=2000)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--box-padding', type=float, help='Explicitly update existing annotations only; 2 adds two pixels per side')
    args = parser.parse_args()
    if args.limit < 100:
        parser.error('Use at least 100 samples')
    if args.box_padding is not None:
        if args.validate_only or args.data is not None:
            parser.error('--box-padding updates --out in place; use --validate-only separately')
        print(json.dumps(pad_symbol_boxes(args.out, args.box_padding), ensure_ascii=True, indent=2))
        raise SystemExit(0)
    if not args.validate_only and args.data is None:
        parser.error('Use --validate-only for the included dataset, or specify --data for an export')
    report = validate_export(args.out) if args.validate_only else export_existing(args.data, args.out, args.limit)
    print(json.dumps(report, ensure_ascii=True, indent=2))
