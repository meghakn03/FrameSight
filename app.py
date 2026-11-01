"""Run with: streamlit run app.py"""
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from threading import RLock

import av
import cv2
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from framesight.engine import Engine

st.set_page_config(page_title='FrameSight · Video intelligence', page_icon='◉', layout='wide')
st.markdown('''<style>
.block-container {padding-top:2.5rem; max-width:1440px}
[data-testid="stMetric"] {background:#151E2E; padding:18px; border-radius:12px}
h1 {letter-spacing:-2px} .eyebrow {color:#33D6AD;letter-spacing:3px;font-size:12px}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">FRAMESIGHT / VIDEO INTELLIGENCE</div>', unsafe_allow_html=True)
st.title('Every frame. A clearer picture.')
st.caption('Detect objects, follow their movement, and turn video into measurable insights.')

with st.sidebar:
    st.header('Detection studio')
    source = st.radio('Video source', ['Upload video', 'Webcam'])
    model = st.selectbox('YOLO model', ['yolo11n.pt', 'yolo11s.pt'], help='Nano is faster; small may improve accuracy.')
    confidence = st.slider('Confidence threshold', 0.1, 0.9, 0.35, 0.05)
    size = st.select_slider('Inference resolution', [320, 480, 640, 960], value=640)
    categories = {'Person': 0, 'Bicycle': 1, 'Car': 2, 'Motorcycle': 3, 'Bus': 5,
                  'Truck': 7, 'Bird': 14, 'Cat': 15, 'Dog': 16}
    selected = st.multiselect('Filter categories', list(categories), help='Leave empty to detect all 80 COCO categories.')
    classes = [categories[c] for c in selected] or None
    st.divider()
    st.caption('YOLO11 + ByteTrack • Local inference\n\nFirst use downloads model weights. Webcam needs camera permission and localhost or HTTPS.')


def dashboard(snapshot, key):
    cols = st.columns(4)
    cols[0].metric('Objects in latest frame', sum(snapshot['current'].values()))
    cols[1].metric('Unique track IDs', sum(snapshot['unique'].values()))
    cols[2].metric('Processed frames', snapshot['frames'])
    cols[3].metric('Inference FPS', f"{snapshot['fps']:.1f}")
    left, right = st.columns([1, 1.4])
    labels = sorted(set(snapshot['unique']) | set(snapshot['observations']))
    table = pd.DataFrame([{'Category': c, 'Visible now': snapshot['current'].get(c, 0),
                           'Unique IDs': snapshot['unique'].get(c, 0),
                           'Detections': snapshot['observations'].get(c, 0)} for c in labels],
                         columns=['Category', 'Visible now', 'Unique IDs', 'Detections'])
    with left:
        st.subheader('Category breakdown')
        st.dataframe(table, hide_index=True, use_container_width=True)
    with right:
        st.subheader('Activity over time')
        if snapshot['history']:
            st.line_chart(pd.DataFrame(snapshot['history']).set_index('frame')['objects'], color='#33D6AD')
        else:
            st.info('Your activity chart will appear after the first frame.')
    st.caption('Unique IDs estimate objects seen during this run. Occlusions can create new IDs. '
               'Detections count observations across frames, not distinct objects. '
               'Exports retain the latest 50,000 detections; chart retains 1,800 frames.')
    a, b = st.columns(2)
    a.download_button('↓ Category summary CSV', table.to_csv(index=False), 'framesight-summary.csv', 'text/csv', key=f'{key}-summary')
    b.download_button('↓ Detection log CSV', pd.DataFrame(snapshot['detections']).to_csv(index=False),
                      'framesight-detections.csv', 'text/csv', key=f'{key}-log')


if source == 'Upload video':
    upload = st.file_uploader('Drop a video to start exploring', type=['mp4', 'mov', 'avi', 'mkv', 'webm'])
    limit = st.number_input('Maximum frames to process', min_value=1, max_value=100000, value=1500,
                            help='A bounded run keeps long videos manageable. Every frame is processed sequentially up to this limit.')
    if upload:
        st.caption(f'{upload.name} · {upload.size / 1024 / 1024:.1f} MB')
    if st.button('Analyze video', type='primary', disabled=upload is None):
        st.session_state.pop('upload_result', None)
        preview = st.empty()
        progress = st.progress(0, text='Preparing detector…')
        try:
            with TemporaryDirectory(prefix='framesight-') as directory:
                path = Path(directory) / ('input' + Path(upload.name).suffix)
                path.write_bytes(upload.getbuffer())
                capture = cv2.VideoCapture(str(path))
                try:
                    if not capture.isOpened():
                        raise ValueError('Cannot decode this video. Try an H.264 MP4 file.')
                    engine = Engine(model, confidence, classes, size)
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    fps = fps if fps > 0 else 30
                    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    target = min(total, int(limit)) if total > 0 else int(limit)
                    for index in range(int(limit)):
                        ok, frame = capture.read()
                        if not ok:
                            break
                        annotated = engine.process(frame, index / fps)
                        if index % 5 == 0:
                            preview.image(annotated, channels='BGR', use_container_width=True)
                            progress.progress(min((index + 1) / target, 1.0), text=f'Processing frame {index + 1} / {target}')
                    snapshot = engine.stats.snapshot()
                    if snapshot['frames'] == 0:
                        raise ValueError('The video contains no decodable frames.')
                    preview.image(annotated, channels='BGR', use_container_width=True)
                    st.session_state.upload_result = snapshot
                    progress.progress(1.0, text=f"Complete · {snapshot['frames']} frames analyzed")
                finally:
                    capture.release()
        except Exception as exc:
            st.error(f'Analysis could not finish: {exc}')
            st.info('Check your internet connection for the first model download, installed dependencies, and video format.')
    if 'upload_result' in st.session_state:
        dashboard(st.session_state.upload_result, 'upload')
    elif not upload:
        st.info('Upload a clip, choose your detection settings, then select Analyze video.')
else:
    st.subheader('Live camera')
    st.caption('Press START to allow browser camera access. Stop the camera before changing settings; select Reset to begin a fresh count.')
    if st.button('Reset webcam session'):
        st.session_state['camera_generation'] = st.session_state.get('camera_generation', 0) + 1
    generation = st.session_state.get('camera_generation', 0)
    config = (model, confidence, tuple(classes or []), size, generation)
    if st.session_state.get('camera_config') != config:
        # Callback state belongs to this session; no cached mutable model is shared between users.
        st.session_state.camera_config = config
        st.session_state.camera_state = {'engine': None, 'error': None, 'lock': RLock(), 'start': monotonic()}
    state = st.session_state.camera_state

    def callback(frame):
        with state['lock']:
            try:
                if state['engine'] is None:
                    state['engine'] = Engine(model, confidence, classes, size)
                    state['start'] = monotonic()
                image = state['engine'].process(frame.to_ndarray(format='bgr24'), monotonic() - state['start'])
                return av.VideoFrame.from_ndarray(image, format='bgr24')
            except Exception as exc:
                state['error'] = str(exc)
                return frame

    ctx = webrtc_streamer(key=f'camera-{config}', mode=WebRtcMode.SENDRECV,
        rtc_configuration={'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]},
        media_stream_constraints={'video': {'width': {'ideal': 640}, 'height': {'ideal': 480}}, 'audio': False},
        video_frame_callback=callback, async_processing=False)

    @st.fragment(run_every=1)
    def live_stats():
        with state['lock']:
            error = state['error']
            snapshot = state['engine'].stats.snapshot() if state['engine'] else None
        if error:
            st.error(f'Camera inference failed: {error}. Stop and reset the webcam to retry.')
        if snapshot:
            dashboard(snapshot, 'webcam')
        else:
            st.info('Waiting for camera frames. Remote networks may require a TURN server; see README.')
    live_stats()
