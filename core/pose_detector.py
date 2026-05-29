import numpy as np
import cv2

try:
    import tensorflow as tf
    _Interpreter = tf.lite.Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter as _Interpreter


class PoseDetector:
    INPUT_SIZE = 192

    def __init__(self, model_path: str):
        self._interpreter = _Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        self._input_idx = self._interpreter.get_input_details()[0]['index']
        self._output_idx = self._interpreter.get_output_details()[0]['index']

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run inference on a BGR frame.

        Returns np.ndarray of shape (17, 3) where each row is [y, x, confidence]
        with y and x normalized to [0, 1].
        """
        resized = cv2.resize(frame, (self.INPUT_SIZE, self.INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(rgb, axis=0).astype(np.uint8)

        self._interpreter.set_tensor(self._input_idx, input_tensor)
        self._interpreter.invoke()

        output = self._interpreter.get_tensor(self._output_idx)
        # output shape: (1, 1, 17, 3) → squeeze to (17, 3)
        return output[0, 0]
