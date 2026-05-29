# Plan: Fitness Pose Analyzer — App Android con Python + TensorFlow

## Contexto

App Android nativa construida en Python. Usa la cámara del celular en tiempo real para analizar posturas durante ejercicios físicos (sentadillas, pull-ups, flexiones, abdominales, L-sit). El modelo de detección de pose está integrado directamente en la app (sin API externa). Construida en partes iterativas con walkthrough para sesiones futuras de Claude.

---

## Stack Tecnológico

| Capa | Tecnología | Por qué |
|------|-----------|---------|
| UI / Framework | **Kivy** | Framework Python para Android/iOS/Desktop, madura y probada |
| Empaquetado Android | **Buildozer** | Convierte app Python/Kivy en APK para Google Play |
| Detección de pose | **TensorFlow Lite** + MoveNet Lightning | Modelo embebido en el APK, ~3MB, 17 keypoints, 30fps en celular |
| Procesamiento | **OpenCV** | Frames de cámara, resize, normalización |
| Álgebra | **NumPy** | Cálculo de ángulos entre keypoints |
| Dev desktop | **TensorFlow** (full) | Para iterar y probar en PC antes de compilar el APK |

**Modelo elegido:** MoveNet SinglePose Lightning (Google)
- 17 keypoints en formato COCO
- `.tflite` embebido en el APK (sin descarga en runtime)
- Optimizado para velocidad en mobile

---

## Arquitectura

```
  Cámara (Android Camera2 / OpenCV)
       │
       ▼
  Frame (numpy array)
       │
       ▼
  PoseDetector (TFLite inference)
       │
       ▼
  17 Keypoints [y, x, confidence]
       │
    ┌──┴──────────────────┐
    │                     │
    ▼                     ▼
Canvas Overlay       ExerciseAnalyzer
(dibujar esqueleto)  (ej: SquatAnalyzer)
                          │
                     PoseUtils
                  (calcular ángulos)
                          │
                    ScoringEngine
                  (puntuación 0-100)
                          │
                    RepCounter
                  (contar repeticiones)
                          │
                      UI Kivy
                  (score + feedback)
```

---

## Keypoints MoveNet (17 puntos)

```
0: nose          1: left_eye      2: right_eye
3: left_ear      4: right_ear     5: left_shoulder
6: right_shoulder 7: left_elbow   8: right_elbow
9: left_wrist    10: right_wrist  11: left_hip
12: right_hip    13: left_knee    14: right_knee
15: left_ankle   16: right_ankle
```

---

## Estructura del Proyecto

```
fitness-pose-analyzer/
├── main.py                        # Entry point Kivy
├── screens/
│   ├── home_screen.py             # Selección de ejercicio (KV layout)
│   ├── camera_screen.py           # Pantalla principal: cámara + IA + overlay
│   └── results_screen.py          # Historial de sesiones (JSON local)
├── core/
│   ├── pose_detector.py           # Wrapper TFLite inference
│   ├── pose_utils.py              # calculate_angle(), is_visible()
│   ├── scoring_engine.py          # compute_score(violations), smooth_score()
│   └── rep_counter.py             # Máquina de estados UP→DOWN→UP
├── analyzers/
│   ├── base_analyzer.py           # Clase abstracta: analyze(keypoints) → AnalysisResult
│   ├── squat_analyzer.py
│   ├── pushup_analyzer.py
│   ├── pullup_analyzer.py
│   ├── abs_analyzer.py
│   └── lsit_analyzer.py
├── assets/
│   ├── models/
│   │   └── movenet_lightning.tflite   # Modelo embebido
│   └── fonts/
├── data/
│   └── sessions.json              # Historial local (escrito por la app)
├── buildozer.spec                 # Config build Android
└── requirements.txt
```

---

## División en Partes

### Parte 1 — Fundación: Proyecto + Detección de Pose en Desktop (3–4h)
**Meta:** Ver el esqueleto de 17 puntos funcionando en la PC con la webcam o un video de prueba.

**Tareas:**
1. Instalar dependencias: `pip install kivy tensorflow opencv-python numpy`
2. Descargar modelo: `movenet_lightning.tflite` de TF Hub
3. Crear `core/pose_utils.py`:
   - `calculate_angle(a, b, c) -> float` — ángulo en grados entre 3 keypoints via `np.arctan2`
   - `is_visible(keypoint, threshold=0.3) -> bool`
4. Crear `core/pose_detector.py`:
   - Clase `PoseDetector` con `__init__(model_path)` — carga TFLite interpreter
   - Método `detect(frame) -> np.ndarray` — preprocess (192×192, uint8), inference, retorna array `[17, 3]`
5. Crear `screens/camera_screen.py`:
   - Widget Kivy con `Clock.schedule_interval` a 30fps
   - Captura frame con OpenCV, llama `PoseDetector.detect()`, dibuja puntos y líneas sobre el frame
   - Usa `Texture.create` de Kivy para mostrar el frame procesado
6. `main.py` minimal: App Kivy que carga `CameraScreen` directamente

**Verificación:** `python main.py` abre ventana, muestra webcam con esqueleto superpuesto en tiempo real.

---

### Parte 2 — Motor de Análisis: Sentadilla (3–4h)
**Meta:** Primer ejercicio con puntuación en tiempo real y conteo de reps.

**Tareas:**
1. Crear `core/scoring_engine.py`:
   - `Violation(name, severity, weight)` — dataclass
   - `compute_score(violations: list[Violation]) -> float` — `100 - Σ(severity × weight × 100)`, mínimo 0
   - `smooth_score(current, previous, alpha=0.3) -> float` — EMA para evitar flickering
2. Crear `core/rep_counter.py`:
   - `RepCounter(down_threshold, up_threshold)` — máquina de estados
   - `update(angle) -> bool` — retorna `True` cuando completa una rep
3. Crear `analyzers/base_analyzer.py`:
   - Clase abstracta con método `analyze(keypoints) -> AnalysisResult`
   - `AnalysisResult(score, violations, feedback_messages, rep_completed)`
4. Crear `analyzers/squat_analyzer.py`:
   - **Ángulo rodilla** (`hip→knee→ankle`): buena profundidad si < 100°, feedback si > 120°
   - **Inclinación de torso** (`shoulder→hip→knee`): feedback si torso muy adelantado (< 50°)
   - **Rodillas sobre pies**: comparar x de `knee` vs `ankle`
   - `RepCounter` con `down_threshold=100`, `up_threshold=160`
5. Conectar en `camera_screen.py`: mostrar score, reps, mensajes de feedback

**Verificación:** Hacer 5 sentadillas, counter sube correctamente, score baja si la forma es mala.

---

### Parte 3 — Más Ejercicios (4–5h)
**Meta:** Los 4 ejercicios restantes implementados con sus analizadores.

**Ángulos clave por ejercicio:**

| Ejercicio | Ángulos | Thresholds |
|-----------|---------|------------|
| **Push-up** | codo `shoulder→elbow→wrist`, cuerpo `shoulder→hip→ankle` | codo < 90° = bueno, cuerpo ± 15° de línea recta |
| **Pull-up** | codo `shoulder→elbow→wrist`, nariz sobre muñeca | codo < 45° = arriba, nariz Y < wrist Y |
| **Abs** | torso `shoulder→hip→knee`, cadera en el suelo | torso < 30° = buen crunch |
| **L-Sit** | cadera `shoulder→hip→knee` ≈ 90°, pierna `hip→knee→ankle` ≈ 180° | tolerance ± 15° |

**Tareas:**
1. Implementar `pushup_analyzer.py`, `pullup_analyzer.py`, `abs_analyzer.py`, `lsit_analyzer.py` siguiendo el mismo patrón de `SquatAnalyzer`
2. Crear `screens/home_screen.py`: grilla de ejercicios con nombre e ícono, navega a `CameraScreen` con el analizador correcto
3. Actualizar `main.py` con `ScreenManager` para navegar entre pantallas

**Verificación:** Desde el menú, seleccionar cada ejercicio y probar que el score y feedback son coherentes.

---

### Parte 4 — UI Completa + Historial (3–4h)
**Meta:** Interfaz pulida, historial de sesiones guardado localmente.

**Tareas:**
1. `screens/results_screen.py`: leer `data/sessions.json`, mostrar lista de sesiones con score promedio, reps, ejercicio y fecha
2. Persistencia: al terminar sesión en `CameraScreen`, guardar `{exercise, reps, avg_score, timestamp}` en JSON
3. UI improvements en `camera_screen.py`:
   - Barra de score animada (color: rojo < 50, amarillo 50–75, verde > 75)
   - Indicador de fase del ejercicio (ej: "BAJA MÁS" / "SUBE")
   - Botón para terminar sesión y ver resultados
4. Sonido: beep en cada rep completada con `kivy.core.audio`

**Verificación:** Sesión completa de 10 sentadillas → botón Terminar → pantalla resultados con datos correctos → historial persiste al cerrar y reabrir la app.

---

### Parte 5 — Compilar APK para Android (4–6h)
**Meta:** APK funcional instalable en cualquier Android.

**Tareas:**
1. Instalar Buildozer: `pip install buildozer`
2. Generar `buildozer.spec`: `buildozer init`
3. Configurar `buildozer.spec`:
   ```ini
   requirements = python3,kivy,numpy,opencv,tensorflow-lite
   android.permissions = CAMERA, WRITE_EXTERNAL_STORAGE
   source.include_exts = py,kv,tflite,json,png,ttf
   android.minapi = 26
   ```
4. Manejar diferencia TF desktop vs TFLite Android:
   - En desktop: usar `tensorflow` con el interpreter de TFLite
   - En Android: usar `tflite_runtime` (más liviano, mismo API)
   - Detectar plataforma: `from kivy.utils import platform; if platform == 'android': ...`
5. Compilar: `buildozer android debug` (primera vez ~30 min)
6. Instalar en celular: `buildozer android deploy run`

**Nota de complejidad:** Buildozer requiere Linux (o WSL en Windows). En este entorno ya corremos Linux. La compilación requiere Java JDK 8/11, Android SDK/NDK que Buildozer descarga automáticamente (~2GB).

---

## Funciones Clave Reutilizables

```python
# core/pose_utils.py
def calculate_angle(a, b, c) -> float:
    """Ángulo en el punto b, entre los vectores ba y bc. Retorna grados."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

# core/pose_detector.py — uso:
detector = PoseDetector("assets/models/movenet_lightning.tflite")
keypoints = detector.detect(frame)  # retorna np.array shape [17, 3] → [y, x, conf]

# analyzers/base_analyzer.py
class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze(self, keypoints: np.ndarray) -> AnalysisResult: ...
```

---

## Walkthrough para Sesiones Futuras

Al inicio de cada sesión nueva de Claude, compartir este archivo y decir en qué parte se quedó. Para verificar el estado actual:

```bash
# Qué partes están implementadas
ls analyzers/
ls core/
ls screens/

# Si el modelo está descargado
ls assets/models/

# Si el APK fue compilado
ls bin/*.apk 2>/dev/null && echo "APK existe" || echo "APK no compilado aún"

# Probar en desktop
python main.py
```

**Sesiones recomendadas:**
- **Sesión 1** → Parte 1: cámara + esqueleto funcionando en desktop
- **Sesión 2** → Parte 2: sentadilla con score y reps
- **Sesión 3** → Parte 3: todos los ejercicios
- **Sesión 4** → Parte 4: UI completa + historial
- **Sesión 5** → Parte 5: compilar APK y probar en Android real

---

## Dependencias (requirements.txt)

```
kivy>=2.3.0
numpy>=1.24.0
opencv-python>=4.8.0
tensorflow>=2.13.0        # solo para desarrollo desktop
tflite-runtime>=2.13.0    # para Android (Buildozer)
```
