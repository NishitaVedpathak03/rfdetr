"""Supported vocabularies; the original class order is never changed."""

SYMBOLS = tuple('0123456789xyzab+-=×')
EXTENDED_SYMBOLS = SYMBOLS + tuple('√ijkn')
SMALL_PIPELINE = 'rf-detr-small-equations-v1'
EXTENDED_PIPELINE = 'rf-detr-structured-equations-v2'

PROFILES = {
    SMALL_PIPELINE: {'name': 'small', 'classes': SYMBOLS, 'max_symbols': 24},
    EXTENDED_PIPELINE: {'name': 'extended', 'classes': EXTENDED_SYMBOLS, 'max_symbols': 48},
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
