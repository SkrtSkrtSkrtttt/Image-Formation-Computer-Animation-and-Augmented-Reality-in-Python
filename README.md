Goal: Minimal, correct forward image formation + simple AR compositing. We project a 3D, rigid-body wireframe/texture cube into 2D using an explicit K–R–T camera model, Rodrigues rotation, and a custom software rasterizer (no black-box drawing).

Stack

Anaconda env (reproducible setup)

Python 3.10+

PyTorch (tensor math, vectorized projection)

NumPy, OpenCV, Pillow (I/O + image utils), Matplotlib (optional previews)

Features

Projective geometry: intrinsics (K), extrinsics (R|T), perspective divide

Rigid motion: axis-angle rotation + translation over time

Rasterization: thickness-aware line drawer for 12 cube edges

Texturing: projective map of the front face

AR: composite over real background images
