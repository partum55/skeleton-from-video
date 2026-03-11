# Skeleton Extraction from Video

**Authors:** [Nazar Mykhailyshchuk](https://www.github.com/partum55), [Oleksii Lasiichuk](https://www.github.com/Oleksii-Lasiichuk)
**Group:** АІ-2
**Course:** Linear Algebra
**Assignment:** LA Project — Robust Skeleton Extraction Methods in Computer Vision
**Date:** March 2026

## Description

A real-time computer vision system that extracts human skeletons from video and uses linear algebra to detect exercises (squats, push-ups, jumping jacks) and count repetitions. The pipeline goes from raw pixels (a 4th-order tensor) through pose estimation, affine normalization, and joint angle computation (via the inner product) to rule-based classification with peak detection.

Every computational step maps directly to linear algebra concepts: vectors in $\mathbb{R}^2$, matrices, linear and affine transformations, rotation matrices, the dot product, norms, and the cosine angle formula.

## Requirements

- Python 3.10+
- pip (Python package installer)
- Virtual environment (recommended)
- A webcam (built-in laptop camera works fine)

## Installation

```bash
# clone the repository
git clone https://github.com/partum55/skeleton-from-video.git
cd skeleton-from-video

# create virtual environment
python3 -m venv venv

# activate virtual environment
# macOS / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Usage

### Live Mode (webcam — real-time exercise detection)

```bash
python -m src.main --source 0 --mode live
```

This opens your webcam, draws the skeleton overlay, detects the exercise, and counts reps in real-time.

**Controls:**
- `q` — quit
- `r` — reset rep counter

### Live Mode with a video file

```bash
python -m src.main --source path/to/video.mp4 --mode live
```

### Analyze Mode (offline skeleton extraction)

```bash
python -m src.main --source path/to/video.mp4 --mode analyze --output skeletons.npy
```

Extracts all skeleton data from a video and saves it as a NumPy array.

### Additional flags

```bash
python -m src.main --source 0 --no-angles    # hide angle overlay
python -m src.main --source 0 --no-plot       # hide angle-vs-time plot
python -m src.main --source 0 --rotate        # apply rotation normalization
```

## Project Structure

```
skeleton-from-video/
├── src/
│   ├── __init__.py        # package marker
│   ├── main.py            # main entry point — video loop, pipeline orchestration
│   ├── skeleton.py        # mediapipe pose estimation, adjacency matrix, graph laplacian
│   ├── normalize.py       # affine normalization: translation, scaling, rotation
│   ├── features.py        # joint angles (dot product), velocity, distance metrics
│   ├── classify.py        # rule-based exercise classification + rep counting
│   └── visualize.py       # skeleton drawing, info overlay, angle plots
├── tests/
│   └── test_main.py       # 34 unit tests covering all modules
├── requirements.txt       # python dependencies
├── must_read_la.md        # detailed mapping of LA concepts to code
├── LICENSE
└── README.md
```

## Pipeline

```
Video (V ∈ R^{T×H×W×3})
    ↓
Pose Estimation (MediaPipe) → S(t) ∈ R^{33×2}
    ↓
Normalization: q = (1/d_ref) · R(α) · (p - c)
    ↓
Joint Angles: θ = arccos(⟨u,v⟩ / (‖u‖·‖v‖))
    ↓
Classification (angle thresholds + peak detection)
    ↓
Output: exercise name + rep count
```

## Features

- Real-time skeleton extraction at ~30 FPS using MediaPipe
- Affine normalization (translation, scaling, rotation) for camera-invariant detection
- Joint angle computation using the inner product cosine formula
- Rule-based classification for squats, push-ups, and jumping jacks
- Automatic repetition counting via peak detection (scipy)
- Live angle-vs-time plot embedded in the video window
- Offline analysis mode with skeleton data export to `.npy`

## Video Demo Approach

For the project video demonstration, we use the laptop's built-in webcam directly:

1. Run the program: `python -m src.main --source 0`
2. Stand in front of the laptop camera and perform exercises
3. The program shows the skeleton overlay, detected exercise, rep count, and angle plot in real time
4. Record the screen using macOS screen recording (Cmd+Shift+5) or OBS

This captures both the code output and the live exercise detection in a single recording.

## Dependencies

| Library | Purpose |
|---|---|
| `mediapipe` | Pose estimation — extracts 33 keypoints from each frame |
| `opencv-python` | Video capture, frame processing, drawing |
| `numpy` | All matrix/vector operations — the LA backbone |
| `scipy` | Peak detection for repetition counting |
| `matplotlib` | Visualization and plotting |
| `pytest` | Unit testing framework |

## Testing

```bash
# run all tests
python -m pytest tests/test_main.py -v

# run a specific test class
python -m pytest tests/test_main.py::TestGraphLaplacian -v
```

34 tests covering:
- Adjacency matrix properties (symmetry, binary, zero diagonal)
- Graph Laplacian properties (PSD, zero eigenvalue, row sums)
- Normalization (centering, scaling, rotation orthogonality)
- Angle computation (0°, 45°, 90°, 180°)
- Distance metrics (Euclidean, cosine similarity)
- Classifier state management and exercise detection

## License

See [LICENSE](LICENSE) for details.

Run tests using unittest:

```bash
# Run all tests
python -m unittest discover tests/

# Run specific test file
python -m unittest tests.test_main

# Run specific test class
python -m unittest tests.test_main.TestMainFunctions

# Run with verbose output
python -m unittest discover tests/ -v

# Run tests from project root
python -m unittest discover -s tests -p "test_*.py"
```

## Tasks Completed

- [x] Task 1: Description
- [x] Task 2: Description
- [ ] Task 3: Description (optional/bonus)

## Known Issues

- Issue 1: Description and potential workaround
- Issue 2: Description

## Development Notes

### Code Style
- Following PEP 8 guidelines
- Type hints used where applicable
- Docstrings for all functions

### Algorithms Used
- Algorithm 1: Brief description
- Algorithm 2: Brief description

## References

- [Course materials link]
- [Python documentation](https://docs.python.org/)
- [Library documentation links]
- [Any research papers or articles]

## Author Notes

Additional comments about implementation choices, challenges faced, or interesting solutions.
