# Skeleton Extraction from Video

**Authors:**

[Oleksii Lasiichuk](https://www.github.com/Oleksii-Lasiichuk), \
[Oleksandr Lykhanskyi](https://github.com/Cossack223), \
[Nazar Mykhailyshchuk](https://www.github.com/partum55)

**Links to our videos:**

- [Oleksii](https://youtu.be/gQ24zIjhJoM?si=AE2emVeSdwU5wQEx),
- [Oleksandr](https://www.youtube.com/watch?v=Kr5_XYvBAN0),
- [Nazar](https://example.com/video3)

## Description

A real-time computer vision pipeline that extracts human skeletons from video using MediaPipe and then applies **manually implemented linear algebra** to normalize poses, reduce dimensionality, detect exercises (squats, push-ups, jumping jacks), and count repetitions.

MediaPipe is used only as a landmark detector (it maps each frame to 33 body-joint coordinates). Everything after detection is built from explicit matrix operations:

1. **Procrustes normalization via SVD** — removes camera angle, position, and distance effects by finding the optimal 2x2 rotation matrix through Singular Value Decomposition.
2. **PCA feature extraction via numpy SVD** — reduces the 66-dimensional pose vector to a small number of principal components using `numpy.linalg.svd` directly (no scikit-learn).
3. **Repetition counting via least-squares** — fits a sinusoidal model to the first PCA component by solving normal equations, and the fitted frequency gives the rep count.
4. **Joint angle computation via the dot product** — computes angles at elbows, knees, shoulders, and hips using the inner product cosine formula.
5. **Graph Laplacian** — builds the skeleton adjacency matrix and graph Laplacian, verifying spectral properties (positive semi-definiteness, zero eigenvalue for connectivity).

## Requirements

- Python3
- pip
- A webcam (for live mode) or a video file

## Installation

Clone the repository:

```bash
git clone https://github.com/partum55/skeleton-from-video.git
cd skeleton-from-video
```

Create and activate a virtual environment on macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Create and activate a virtual environment on Windows:

```bash
python3 -m venv .venv
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

The MediaPipe model file (`pose_landmarker_lite.task`) is already included in the repository.

## How to Run

### Live mode (webcam) — real-time exercise detection

```bash
python main.py
```

This opens your webcam, draws a skeleton overlay on the video, detects the exercise type, and counts repetitions in real time.

### Live mode with a video file

```bash
python main.py --source path/to/video.mp4
```

Video files loop automatically when they reach the end.

### Analyze mode (offline skeleton extraction)

```bash
python main.py --mode analyze --source path/to/video.mp4 --output skeletons.npy
```

Extracts all skeletons from a video file and saves them as a NumPy array (`T x 33 x 2`). Also prints the Euclidean distance and cosine similarity between the first and last frame.

### All CLI flags

| Flag | Default | Description |
| ------ | --------- | ------------- |
| `--source` | `0` (webcam) | Video source: `0` for webcam, or path to a video file |
| `--mode` | `live` | `live` for real-time detection, `analyze` for offline processing |
| `--output` | none | Output path for `.npy` skeleton data (analyze mode only) |
| `--rotate` | off | Apply Procrustes rotation normalization |
| `--no-angles` | off | Hide the joint angle overlay |
| `--no-plot` | off | Hide the angle-vs-time plot |
| `--no-mirror` | off | Disable selfie mirror (webcam is mirrored by default) |
| `--frame-width` | 1280 | Camera capture width |
| `--frame-height` | 720 | Camera capture height |
| `--window-width` | 1600 | Initial display window width |
| `--window-height` | 900 | Initial display window height |

### Keyboard controls (live mode)

| Key | Action                            |
| --- | --------------------------------- |
| `q` | Quit                              |
| `r` | Reset rep counter and all filters |
| `f` | Toggle fullscreen                 |

## Pipeline

```raw
Raw video frame
    |
    v
[MediaPipe] Extract 33 landmarks (x, y, visibility)
    |
    v
[Temporal filter] EMA smoothing + visibility-based repair
    |
    v
[Normalize] Translate to hip center -> Scale by torso length -> Procrustes rotation (SVD)
    |
    v
[Features] Joint angles via dot product, body position features
    |
    |                                               |
    v                                               v
[Angle-based classifier]                [PCA on sliding window]
  FSM with multi-layer validation         Manual SVD -> project to k dims
  (phase time, hold, velocity,            -> sinusoidal least-squares fit
   amplitude, hysteresis)                 -> frequency -> rep count
    |                                               |
    v                                               v
               [Label fusion]
        Confidence-gated combination of
        angle-based and PCA-based labels
                    |
                    v
             [Visualization]
        Skeleton overlay, HUD, angle plot
```

### Step-by-step walkthrough

1. **Capture frame** — OpenCV reads a frame from the webcam or video file.
2. **Pose estimation** — MediaPipe detects 33 body keypoints and returns $S(t) \in \mathbb{R}^{33 \times 3}$ (x, y, visibility).
3. **Temporal filtering** — An exponential moving average (EMA) smooths jittery landmarks; low-visibility joints fall back to the previous frame's values.
4. **Normalization** — The skeleton is translated so the hip center is at the origin, scaled so the torso length equals 1, and optionally rotated to align with a reference pose via Procrustes/SVD.
5. **Joint angles** — Angles at 8 joints (left/right elbows, knees, shoulders, hips) are computed using the dot-product formula:
$$\theta = \arccos\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}\right)$$
6. **Body features** — Torso verticality and leg spread are computed to distinguish exercises (e.g., standing vs. push-up position).
7. **FSM classification** — A finite-state machine with two states (up/down) detects exercise type and counts reps using angle thresholds with multi-layer validation (minimum phase time, hold time, velocity check, amplitude check, hysteresis).
8. **PCA + sinusoidal fitting** — A sliding window of normalized skeletons is flattened to $\mathbb{R}^{T \times 66}$, PCA reduces it to $k$ dimensions (95% variance), and a sinusoidal model $z_1(t) = w_1\sin(\omega t) + w_2\cos(\omega t) + w_3$ is fit via least-squares to estimate rep frequency.
9. **Label fusion** — The angle-based label and PCA-based label are combined using the PCA confidence score.
10. **Visualization** — The skeleton, exercise name, rep count, FPS, and an angle-vs-time plot are drawn on the frame.

## Linear Algebra Concepts Implemented Manually

| Concept | Where in Code | What It Does |
| -------- | --------------- | -------------- |
| **SVD (Singular Value Decomposition)** | `src/normalize.py`, `src/pca.py` | Procrustes alignment (optimal rotation) and PCA (dimensionality reduction) |
| **Rotation matrices** | `src/normalize.py` | 2x2 orthogonal matrix with det=+1 from SVD, applied to align poses |
| **Affine transformations** | `src/normalize.py` | Translation (hip centering), scaling (torso normalization) |
| **Inner product / dot product** | `src/features.py` | Joint angle computation via the cosine formula |
| **Vector norms** | `src/features.py`, `src/normalize.py` | Euclidean distance, torso length, cosine similarity |
| **Least-squares (normal equations)** | `src/repetition.py` | Sinusoidal fit: `A^T A w = A^T z1`, grid search over frequencies |
| **Design matrices** | `src/repetition.py` | `A(omega) = [sin(wt), cos(wt), 1]` for each frame |
| **Data centering** | `src/linalg_utils.py`, `src/pca.py` | Mean subtraction before SVD for PCA |
| **Variance from singular values** | `src/pca.py`, `src/linalg_utils.py` | Component selection: keep k such that cumulative `sigma_i^2` >= 95% of total |
| **Projection onto subspace** | `src/pca.py` | `Z = X_centered * V_k` projects T frames to k dimensions |
| **Graph adjacency matrix** | `src/skeleton.py` | `A in {0,1}^{33x33}` — symmetric, binary, zero-diagonal |
| **Graph Laplacian** | `src/skeleton.py` | `L = D - A` — symmetric PSD, zero eigenvalue proves connectivity |
| **Matrix flattening (vectorization)** | `src/features.py`, `src/linalg_utils.py` | `R^{33x2}` -> `R^{66}` for PCA input |

## Project Structure

```
skeleton-from-video/
├── main.py                        # entry point: CLI argument parsing, live and analyze modes
├── src/
│   ├── __init__.py                # package marker
│   ├── skeleton.py                # MediaPipe pose estimation, adjacency matrix, graph Laplacian, EMA filter
│   ├── normalize.py               # Procrustes normalization: translation, scaling, SVD-based rotation
│   ├── features.py                # joint angles (dot product), body position features, distance metrics
│   ├── classify.py                # FSM-based exercise classification and rep counting
│   ├── pca.py                     # PCA via manual SVD (no scikit-learn)
│   ├── repetition.py              # sinusoidal least-squares fitting for rep counting
│   ├── linalg_utils.py            # shared utilities: centering, flattening, component selection
│   └── visualize.py               # skeleton drawing, HUD overlay, angle-vs-time plot
├── tests/
│   ├── conftest.py                # pytest configuration
│   ├── test_main.py               # adjacency, Laplacian, normalization, angles, classifier tests
│   ├── test_normalize.py          # translation, scaling, Procrustes rotation, full pipeline tests
│   ├── test_pca.py                # centering, orthonormality, variance retention, reconstruction tests
│   ├── test_repetition.py         # design matrix, normal equations, frequency recovery, classification tests
│   └── test_runtime_stability.py  # temporal smoothing, FSM timing, label fusion tests
├── pose_landmarker_lite.task      # MediaPipe pre-trained model file
├── requirements.txt               # Python dependencies
├── LICENSE
└── .github/workflows/ci.yml      # GitHub Actions CI (runs tests on push)
```

## Testing

168 unit tests verify every linear algebra property:

```bash
source venv/bin/activate
python -m pytest tests/ -v
```

### What the tests cover

- **Graph structure** — adjacency matrix symmetry, binary values, zero diagonal; Laplacian is PSD with zero eigenvalue
- **Normalization** — hip centering, unit torso length, rotation orthogonality (`R^T R = I`), determinant = +1, length preservation, Procrustes recovery of known rotations
- **Angles** — 0, 45, 90, 180 degree cases using the dot product formula
- **Distance metrics** — Euclidean distance symmetry/positivity, cosine similarity range
- **PCA** — orthonormality of components (`V_k^T V_k = I`), zero-mean projections, variance retention >= threshold, reconstruction error decreases with k, full-rank gives near-zero error
- **Least-squares repetition counting** — design matrix shape, normal equations hold (`A^T A w = A^T z1`), perfect-fit residual is near zero, frequency recovery from noiseless and noisy synthetic signals
- **Runtime stability** — EMA smoothing weights, visibility-based joint repair, FSM uses wall-clock time (not frame count), label fusion respects PCA confidence, SVD component sign alignment

## Dependencies

| Library | Purpose |
| -------- | -------- |
| `mediapipe` | Pose estimation — detects 33 keypoints per frame |
| `opencv-python` | Video capture, frame processing, drawing |
| `numpy` | All matrix/vector operations (SVD, dot products, norms, etc.) |
| `scipy` | Peak detection (used alongside the least-squares approach) |
| `matplotlib` | Angle-vs-time plot rendering |
| `pytest` | Unit testing framework |

## License

See [LICENSE](LICENSE) for details.
