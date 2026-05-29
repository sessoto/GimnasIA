import os
import cv2
import numpy as np

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.progressbar import ProgressBar
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.utils import platform

from core.pose_detector import PoseDetector
from core.skeleton_drawer import draw_skeleton

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'assets', 'models', 'movenet_lightning.tflite'
)

_ANALYZER_MAP = {
    'squat':  'analyzers.squat_analyzer.SquatAnalyzer',
    'pushup': 'analyzers.pushup_analyzer.PushupAnalyzer',
    'pullup': 'analyzers.pullup_analyzer.PullupAnalyzer',
    'abs':    'analyzers.abs_analyzer.AbsAnalyzer',
    'lsit':   'analyzers.lsit_analyzer.LSitAnalyzer',
}


def _load_analyzer(exercise_id: str):
    path = _ANALYZER_MAP.get(exercise_id)
    if not path:
        return None
    module_path, cls_name = path.rsplit('.', 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)()


class CameraWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._texture = None
        self._cap = None
        self._detector = PoseDetector(MODEL_PATH)
        self._analyzer = None
        self._running = False
        self.on_result = None   # callback(AnalysisResult)
        with self.canvas:
            Color(0, 0, 0, 1)
            self._rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._sync_rect, size=self._sync_rect)

    def _sync_rect(self, *_):
        self._rect.pos = self.pos
        self._rect.size = self.size

    def set_analyzer(self, analyzer):
        self._analyzer = analyzer

    def start(self):
        if platform == 'android':
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA])
        self._cap = cv2.VideoCapture(0)
        self._running = True
        Clock.schedule_interval(self._update, 1.0 / 30)

    def stop(self):
        self._running = False
        Clock.unschedule(self._update)
        if self._cap:
            self._cap.release()
            self._cap = None

    def _update(self, dt):
        if not self._cap or not self._cap.isOpened():
            return
        ret, frame = self._cap.read()
        if not ret:
            return

        keypoints = self._detector.detect(frame)

        result = None
        if self._analyzer is not None:
            result = self._analyzer.analyze(keypoints)
            if self.on_result:
                self.on_result(result)

        draw_skeleton(frame, keypoints)

        # Flip vertically: OpenCV origin top-left, Kivy bottom-left
        frame = cv2.flip(frame, 0)
        h, w = frame.shape[:2]

        if self._texture is None or self._texture.size != (w, h):
            self._texture = Texture.create(size=(w, h), colorfmt='bgr')
        self._texture.blit_buffer(frame.tobytes(), colorfmt='bgr', bufferfmt='ubyte')

        self.canvas.clear()
        with self.canvas:
            Color(1, 1, 1, 1)
            Rectangle(texture=self._texture, pos=self.pos, size=self.size)


class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._exercise_id = None
        self._exercise_name = ''
        self._analyzer = None
        self._build_ui()

    def _build_ui(self):
        root = BoxLayout(orientation='vertical')

        # Top bar
        top = BoxLayout(size_hint_y=None, height=52, padding=[8, 4])
        self._back_btn = Button(
            text='← Volver', size_hint_x=None, width=110,
            background_color=(0.15, 0.15, 0.15, 1), font_size=15,
        )
        self._back_btn.bind(on_release=self._go_back)
        self._title_lbl = Label(text='Pose Analyzer', font_size=20, bold=True)
        top.add_widget(self._back_btn)
        top.add_widget(self._title_lbl)

        # Camera view
        self._cam = CameraWidget()
        self._cam.on_result = self._on_analysis

        # Score bar
        score_row = BoxLayout(size_hint_y=None, height=28, padding=[8, 2])
        self._score_lbl = Label(text='Score: --', size_hint_x=None, width=100, font_size=15)
        self._score_bar = ProgressBar(max=100, value=0)
        score_row.add_widget(self._score_lbl)
        score_row.add_widget(self._score_bar)

        # Rep counter + phase
        stats_row = BoxLayout(size_hint_y=None, height=36, padding=[8, 2])
        self._reps_lbl = Label(text='Reps: 0', font_size=18, bold=True)
        self._phase_lbl = Label(text='', font_size=18, color=(1, 0.8, 0, 1))
        stats_row.add_widget(self._reps_lbl)
        stats_row.add_widget(self._phase_lbl)

        # Feedback
        self._feedback_lbl = Label(
            text='Posicionáte frente a la cámara',
            size_hint_y=None, height=46,
            font_size=15, halign='center',
        )
        self._feedback_lbl.bind(size=lambda inst, val: setattr(inst, 'text_size', (val[0], None)))

        root.add_widget(top)
        root.add_widget(self._cam)
        root.add_widget(score_row)
        root.add_widget(stats_row)
        root.add_widget(self._feedback_lbl)
        self.add_widget(root)

    def configure(self, exercise_id: str, exercise_name: str):
        self._exercise_id = exercise_id
        self._exercise_name = exercise_name
        self._title_lbl.text = exercise_name
        self._analyzer = _load_analyzer(exercise_id)
        self._cam.set_analyzer(self._analyzer)

    def on_enter(self):
        self._cam.start()

    def on_leave(self):
        self._cam.stop()

    def _on_analysis(self, result):
        score = result.score
        self._score_lbl.text = f'Score: {score:.0f}'
        self._score_bar.value = score

        if score >= 75:
            self._score_bar.color = (0.1, 0.8, 0.1, 1)
        elif score >= 50:
            self._score_bar.color = (1.0, 0.7, 0.0, 1)
        else:
            self._score_bar.color = (0.9, 0.1, 0.1, 1)

        self._reps_lbl.text = f'Reps: {result.rep_count}'
        self._phase_lbl.text = result.phase

        msg = ' · '.join(result.feedback[:2]) if result.feedback else ''
        self._feedback_lbl.text = msg

    def _go_back(self, *_):
        self.manager.current = 'home'
