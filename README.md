# Fitness Pose Analyzer

App de análisis de postura para ejercicios físicos, construida en Python con
Kivy + TensorFlow Lite (MoveNet Lightning). El modelo de detección de pose
está **integrado en la app** — no depende de ninguna API externa.

Analiza en tiempo real: **sentadilla, flexiones, pull-up, abdominales, L-sit**.

## Estado del proyecto

- ✅ Parte 1 — Detección de pose en desktop (esqueleto en vivo)
- ✅ Parte 2 — Motor de análisis: sentadilla con score + conteo de reps
- ✅ Parte 3 — Los 5 ejercicios + navegación por pantallas
- ⬜ Parte 4 — UI completa + historial de sesiones
- ⬜ Parte 5 — Compilar APK para Android (Buildozer)

Ver `PLAN.md` para el plan completo, la arquitectura y el walkthrough de
las sesiones futuras.

## Cómo correr en la PC

```bash
pip install -r requirements.txt
python main.py
```

Se abre una ventana, seleccionás un ejercicio y la cámara muestra el
esqueleto superpuesto con score, repeticiones y feedback en tiempo real.

## Estructura

```
main.py                  # Entry point Kivy (ScreenManager)
core/
  pose_detector.py       # Inferencia TFLite (MoveNet → 17 keypoints)
  pose_utils.py          # Cálculo de ángulos, visibilidad
  skeleton_drawer.py     # Dibuja el esqueleto sobre el frame
  scoring_engine.py      # Puntuación 0-100 + suavizado
  rep_counter.py         # Máquina de estados para contar reps
analyzers/               # Un analizador por ejercicio
screens/                 # HomeScreen + CameraScreen
assets/models/
  movenet_lightning.tflite   # Modelo embebido (2.8 MB)
```
