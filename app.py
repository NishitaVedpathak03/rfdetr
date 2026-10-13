"""Simple upload, image comparison, predicted steps and LaTeX interface."""
import hashlib
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from ui.steps import step_description
from ui.images import render_overlay, uploaded_image


PROJECT = Path(__file__).resolve().parent
RFDETR_CHECKPOINT = PROJECT / 'checkpoints/rfdetr_small/checkpoint_best_total.pth'
st.set_page_config(page_title='Equation OCR', layout='wide')
st.markdown('''<style>
.block-container {max-width: 1150px; padding-top: 2rem; padding-bottom: 2rem;}
[data-testid="stLatex"] {max-height: 180px; overflow: auto;}
</style>''', unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=3)
def decode_image(data):
    return uploaded_image(data)


@st.cache_resource(show_spinner=False, max_entries=2)
def cached_predictor(path, stamp):
    from ui.inference import load_predictor
    return load_predictor(path)



def show_result(sample, identity):
    if st.session_state.get('viewer_identity') != identity:
        st.session_state.viewer_identity = identity
        set_step(0, sample)
        st.session_state.refresh_player = False
    interval = .8 if st.session_state.playing else None
    st.fragment(run_every=interval)(show_images_and_steps)(sample)
    show_latex(sample)


def set_step(index, sample):
    count = len(sample['steps'])
    st.session_state.step_index = max(0, min(index, count - 1)) if count else 0
    st.session_state.step_slider = st.session_state.step_index + 1
    st.session_state.playing = False
    st.session_state.last_tick = time.monotonic()
    st.session_state.refresh_player = True


def move_step(delta, sample):
    set_step(st.session_state.step_index + delta, sample)


def seek_step(sample):
    set_step(st.session_state.step_slider - 1, sample)


def toggle_play(sample):
    playing = not st.session_state.playing
    if playing and st.session_state.step_index == len(sample['steps']) - 1:
        set_step(0, sample)
    st.session_state.playing = playing
    st.session_state.last_tick = time.monotonic()
    st.session_state.refresh_player = True


def show_images_and_steps(sample):
    rejected = sample.get('rejection_reason')
    if rejected:
        st.session_state.playing = False
    steps = sample['steps']
    refresh = st.session_state.pop('refresh_player', False)
    if st.session_state.playing and time.monotonic() - st.session_state.last_tick >= .8:
        st.session_state.step_index = min(st.session_state.step_index + 1, len(steps) - 1)
        st.session_state.step_slider = st.session_state.step_index + 1
        st.session_state.last_tick = time.monotonic()
        if st.session_state.step_index == len(steps) - 1:
            st.session_state.playing = False
            refresh = True
    index = st.session_state.step_index
    step = steps[index] if steps else None
    left, right = st.columns(2)
    with left:
        st.subheader('Original image')
        st.image(sample['original'], width='stretch')
    with right:
        st.subheader('Colored image')
        colored = render_overlay(sample['original'] if rejected else sample['colored'], sample['nodes'], {}, show_boxes=False,
                                 active_id=step['node_id'] if step and not rejected else None, display_size=(1400, 1000))
        st.image(colored, width='stretch')
        if rejected:
            st.caption('Coloring unavailable: recognition failed.')
    st.subheader('Steps')
    if rejected:
        st.info('Recognition failed: ' + rejected)
        return
    if not steps:
        st.info('No regions were recognized in this image.')
        return
    node = next(n for n in sample['nodes'] if n['id'] == step['node_id'])
    st.write(f'**Step {index + 1} of {len(steps)}** · {step_description(step, node, predicted=sample["source"] == "model prediction")}')
    if len(steps) > 1:
        st.slider('Step', 1, len(steps), key='step_slider', on_change=seek_step, args=(sample,),
                  label_visibility='collapsed')
    controls = st.columns(4)
    controls[0].button('Previous', key='previous', on_click=move_step, args=(-1, sample),
                       disabled=index == 0, width='stretch')
    controls[1].button('Next', key='next', on_click=move_step, args=(1, sample),
                       disabled=index == len(steps) - 1, width='stretch')
    controls[2].button('Pause' if st.session_state.playing else 'Play', key='play_pause',
                       on_click=toggle_play, args=(sample,), disabled=len(steps) < 2, width='stretch')
    controls[3].button('Reset', key='reset', on_click=set_step, args=(0, sample), width='stretch')
    if refresh:
        st.rerun()


def show_latex(sample):
    prediction = sample['prediction']
    draft = prediction['latex']
    with st.expander('LaTeX result', expanded=True):
        if sample.get('rejection_reason'):
            st.caption('LaTeX unavailable because the predicted structure is incomplete or ambiguous.')
            return
        if not draft:
            st.caption('No LaTeX is available because no symbols were detected.')
            return
        st.caption('Reconstructed from detected symbols and spatial rules; check it against the image.')
        if len(draft) <= 6000 and prediction['latex_metadata']['max_depth'] <= 40:
            st.latex(draft)
        st.code(draft, language='latex', height=90)
        st.download_button('Download LaTeX', draft + '\n', file_name=f'{sample["name"]}_predicted.tex',
                           mime='text/plain', on_click='ignore', key='export_latex')


def main():
    st.title('Equation OCR')
    upload = st.file_uploader('Upload an equation image', type=['png', 'jpg', 'jpeg'],
                              key='upload', max_upload_size=25)
    if upload is None:
        st.session_state.pop('prediction_sample', None)
        st.session_state.pop('image_identity', None)
        st.session_state.pop('upload_digest', None)
        st.session_state.pop('viewer_identity', None)
        st.session_state.playing = False
        return
    try:
        data = upload.getvalue()
        image = decode_image(data)
        digest = hashlib.sha256(data).hexdigest()
        if st.session_state.get('upload_digest') != digest:
            st.session_state.upload_digest = digest
        checkpoint_path = RFDETR_CHECKPOINT
        if not checkpoint_path.is_file():
            st.error(f'Model checkpoint not found: {checkpoint_path}')
            return
        # Invalidate both model and per-image caches when inference code changes.
        stamp = (checkpoint_path.stat().st_mtime_ns, checkpoint_path.stat().st_size,
                 (PROJECT / 'ui/inference.py').stat().st_mtime_ns,
                 (PROJECT / 'src/visualize.py').stat().st_mtime_ns,
                 (PROJECT / 'src/data.py').stat().st_mtime_ns,
                 (PROJECT / 'src/model.py').stat().st_mtime_ns,
                 (PROJECT / 'src/layout.py').stat().st_mtime_ns,
                 ((checkpoint_path.parent / 'symbol_pipeline.json').stat().st_mtime_ns
                  if checkpoint_path.suffix == '.pth' and (checkpoint_path.parent / 'symbol_pipeline.json').is_file() else None))
        identity = (hashlib.sha256(data).hexdigest(), str(checkpoint_path), stamp)
        if st.session_state.get('image_identity') != identity:
            st.session_state.pop('prediction_sample', None)
            st.session_state.playing = False
            with st.spinner('Processing equation…'):
                from ui.inference import analyze_image
                predictor = cached_predictor(str(checkpoint_path), stamp)
                name = Path(getattr(upload, 'name', 'uploaded_equation')).stem
                sample = analyze_image(image, predictor, name)
            st.session_state.prediction_sample = sample
            st.session_state.image_identity = identity
            set_step(0, sample)
            st.session_state.refresh_player = False
        sample = st.session_state.prediction_sample
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        st.session_state.playing = False
        st.error(f'Cannot process this image: {error}')
        return
    except Exception as error:
        st.session_state.playing = False
        st.error(f'Unable to run the model: {error}')
        return
    prediction = sample['prediction']
    if prediction.get('metadata', {}).get('node_limit_reached'):
        sample['rejection_reason'] = 'the detector reached its candidate limit; this is not a trustworthy equation structure.'
    elif prediction['latex_metadata']['status'] == 'partial':
        sample['rejection_reason'] = 'the predicted equation structure is incomplete or ambiguous.'
    else:
        sample.pop('rejection_reason', None)
    show_result(sample, ('model', identity))


if __name__ == '__main__':
    main()
