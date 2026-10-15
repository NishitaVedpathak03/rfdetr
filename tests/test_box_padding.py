"""Annotation-only padding stays bounded and never changes image pixels."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data import pad_symbol_boxes, write_json
from src.profiles import COMPLEX_PIPELINE


class BoxPaddingTests(unittest.TestCase):
    def dataset(self, root):
        write_json(root / 'symbol_dataset.json', {'pipeline': COMPLEX_PIPELINE})
        for split in ('train', 'valid', 'test'):
            folder = root / split
            folder.mkdir()
            Image.new('RGB', (32, 32), 'white').save(folder / 'image.png')
            write_json(folder / '_annotations.coco.json', {
                'images': [{'id': 0, 'file_name': 'image.png', 'width': 32, 'height': 32}],
                'annotations': [{'id': 1, 'image_id': 0, 'category_id': 0, 'bbox': [0., 0., 4., 3.], 'area': 12.},
                                {'id': 2, 'image_id': 0, 'category_id': 10, 'bbox': [28., 29., 4., 3.], 'area': 12.}]})
        write_json(root / 'preparation_report.json', {'training_started': False})

    def test_clipping_labels_pixels_and_idempotency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.dataset(root)
            images = list(root.glob('*/*.png'))
            hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in images}
            report = pad_symbol_boxes(root, 2)
            self.assertFalse(report['training_started'])
            self.assertFalse(report['images_changed'])
            coco = json.loads((root / 'train/_annotations.coco.json').read_text())
            self.assertEqual(coco['annotations'][0]['bbox'], [0., 0., 6., 5.])
            self.assertEqual(coco['annotations'][1]['bbox'], [26., 27., 6., 5.])
            self.assertEqual([a['category_id'] for a in coco['annotations']], [0, 10])
            self.assertEqual(coco['annotations'][0]['tight_bbox'], [0., 0., 4., 3.])
            first = (root / 'train/_annotations.coco.json').read_bytes()
            pad_symbol_boxes(root, 2)
            self.assertEqual(first, (root / 'train/_annotations.coco.json').read_bytes())
            self.assertEqual(hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in images})
            pad_symbol_boxes(root, 0)
            restored = json.loads((root / 'train/_annotations.coco.json').read_text())
            self.assertEqual(restored['annotations'][0]['bbox'], [0., 0., 4., 3.])

    def test_invalid_padding_leaves_annotations_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.dataset(root)
            first = (root / 'train/_annotations.coco.json').read_bytes()
            for invalid in (-1, 6, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    pad_symbol_boxes(root, invalid)
            self.assertEqual(first, (root / 'train/_annotations.coco.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
