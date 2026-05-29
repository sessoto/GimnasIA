from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

EXERCISES = [
    {'id': 'squat',  'name': 'Sentadilla',  'icon': '[S]'},
    {'id': 'pushup', 'name': 'Flexiones',   'icon': '[P]'},
    {'id': 'pullup', 'name': 'Pull-up',     'icon': '[U]'},
    {'id': 'abs',    'name': 'Abdominales', 'icon': '[A]'},
    {'id': 'lsit',   'name': 'L-Sit',       'icon': '[L]'},
]


class HomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build_ui()

    def _build_ui(self):
        root = BoxLayout(orientation='vertical', padding=20, spacing=10)

        title = Label(
            text='Fitness Pose Analyzer',
            font_size=28, bold=True,
            size_hint_y=None, height=70,
        )
        subtitle = Label(
            text='Selecciona un ejercicio para comenzar',
            font_size=16,
            size_hint_y=None, height=40,
        )

        grid = GridLayout(cols=2, spacing=15, padding=10)
        for ex in EXERCISES:
            btn = Button(
                text=f"{ex['name']}",
                font_size=20,
                background_color=(0.15, 0.45, 0.75, 1),
                size_hint_y=None, height=90,
            )
            btn._exercise_id = ex['id']
            btn._exercise_name = ex['name']
            btn.bind(on_release=self._select_exercise)
            grid.add_widget(btn)

        root.add_widget(title)
        root.add_widget(subtitle)
        root.add_widget(grid)
        self.add_widget(root)

    def _select_exercise(self, btn):
        cam = self.manager.get_screen('camera')
        cam.configure(btn._exercise_id, btn._exercise_name)
        self.manager.current = 'camera'
