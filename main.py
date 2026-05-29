import os
os.environ['KIVY_NO_ENV_CONFIG'] = '1'

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, FadeTransition

from screens.home_screen import HomeScreen
from screens.camera_screen import CameraScreen


class FitnessPoseApp(App):
    def build(self):
        self.title = 'Fitness Pose Analyzer'
        sm = ScreenManager(transition=FadeTransition())
        sm.add_widget(HomeScreen(name='home'))
        sm.add_widget(CameraScreen(name='camera'))
        return sm


if __name__ == '__main__':
    FitnessPoseApp().run()
