# hand-lens

Real-time webcam filter lens controlled by your hands. Point both hands at the camera using your thumbs and index fingers to form a dynamic quadrilateral — any of 6 live visual filters is rendered strictly inside that quad, with **no hand skeleton** drawn on screen.

## How It Works

- Tracks both hands in real time via MediaPipe `HandLandmarker`.
- Connects **index-to-index**, **thumb-to-thumb**, and **thumb-to-index** on each hand to form a clean quadrilateral.
- Smooths corner coordinates (EMA) for jitter-free motion and renders the selected filter inside the masked polygon.

## Filter Controls

Press **`1`–`6`** on your keyboard at any time to switch filters:

| Key | Filter | Effect |
| :--- | :--- | :--- |
| `1` | **Invert** | Negative color inversion |
| `2` | **Pixelate** | Chunky mosaic pixelation |
| `3` | **Thermal** | Inferno heat-vision colormap |
| `4` | **Neon Edges** | Glowing cyberpunk edge detection |
| `5` | **Noir Blur** | High-contrast monochrome blur |
| `6` | **RGB Glitch** | Chromatic aberration + scanlines |

Press **`Q`** or **`Esc`** to quit.

## Quick Start

Requires Python 3.10–3.12 and a webcam:

```bash
git clone https://github.com/filipboski1313-a11y/hand-lens.git
cd hand-lens
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

*(On first run, `main.py` automatically downloads MediaPipe's `hand_landmarker.task` model to `models/`.)*
