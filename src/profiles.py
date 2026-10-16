"""Supported vocabularies; the original class order is never changed."""

SYMBOLS = tuple('0123456789xyzab+-=×')
EXTENDED_SYMBOLS = SYMBOLS + tuple('√ijkn')
SMALL_PIPELINE = 'rf-detr-small-equations-v1'
EXTENDED_PIPELINE = 'rf-detr-structured-equations-v2'
COMPLEX_SYMBOLS = EXTENDED_SYMBOLS + tuple('()sclog')
COMPLEX_PIPELINE = 'rf-detr-complex-equations-v3'
CALCULUS_SYMBOLS = COMPLEX_SYMBOLS + tuple('tdfC∫∑∂∞')
CALCULUS_PIPELINE = 'rf-detr-calculus-equations-v4'

PROFILES = {
    SMALL_PIPELINE: {'name': 'small', 'classes': SYMBOLS, 'max_symbols': 24},
    EXTENDED_PIPELINE: {'name': 'extended', 'classes': EXTENDED_SYMBOLS, 'max_symbols': 48},
    COMPLEX_PIPELINE: {'name': 'complex', 'classes': COMPLEX_SYMBOLS, 'max_symbols': 96},
    CALCULUS_PIPELINE: {'name': 'calculus', 'classes': CALCULUS_SYMBOLS, 'max_symbols': 128},
}


def profile_for(pipeline):
    try:
        return PROFILES[pipeline]
    except KeyError as error:
        raise ValueError(f'Unsupported equation pipeline: {pipeline}') from error


def pipeline_for(name):
    for pipeline, profile in PROFILES.items():
        if profile['name'] == name:
            return pipeline
    raise ValueError(f'Unknown equation profile: {name}')
