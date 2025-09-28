# Naafiul Hossain (115017623)
# 9/17//25
# Project 2-Computer Vision
# 115107623

import numpy as np
import cv2
import sys

# -- Colab/desktop display helper ---------------------------------------------
def show_img(img, win_name="frame"):
    try:
        from google.colab.patches import cv2_imshow  # type: ignore
        cv2_imshow(img)
        return -1  # no key handling in Colab
    except Exception:
        cv2.imshow(win_name, img)
        return cv2.waitKey(1) & 0xFF  # return key pressed

# -- Geometry / projection helpers ---------------------------------------------
def Map2Da(Km, Rm, Tm, Xw):
    """
    Project a 3D world point (mm) to the image plane (mm) using pinhole model.
    Returns q_mm = [x_mm, y_mm] relative to the principal point (0,0),
    or None if the point is behind the camera.
    """
    Xc = Rm @ Xw + Tm  # camera coords (mm)
    Z = Xc[2]
    if Z <= 1e-6:
        return None
    xn = Xc[0] / Z
    yn = Xc[1] / Z
    q = Km @ np.array([xn, yn, 1.0])  # [f*xn, f*yn, 1]
    return q[:2]  # mm

def MapIndex(q_mm, c_center, r_center, p_scale):
    """
    Convert image-plane mm -> pixel indices (col, row).
    +x to the right, +y up in camera; rows increase downward on image.
    """
    col = int(np.round(c_center + (q_mm[0] / p_scale)))
    row = int(np.round(r_center - (q_mm[1] / p_scale)))
    return (col, row)

def drawLine(img, pt0, pt1, bgr=(0,0,255), thickness=1):
    """
    Draw a line without cv2.line() using 0.5-pixel stepping (assignment spec).
    pt0/pt1 are (col, row) ints. OpenCV uses BGR.
    """
    if pt0 is None or pt1 is None:
        return
    x0, y0 = pt0
    x1, y1 = pt1
    dx, dy = x1 - x0, y1 - y0
    d = float(np.hypot(dx, dy))
    if d < 1e-6:
        if 0 <= y0 < img.shape[0] and 0 <= x0 < img.shape[1]:
            img[y0, x0] = bgr
        return
    ux, uy = dx / d, dy / d
    steps = int(np.ceil(d * 2.0)) + 1  # 0.5 px increments
    rad = max(0, thickness // 2)
    H, W = img.shape[:2]

    for k in range(steps):
        c = 0.5 * k
        x = int(round(x0 + c * ux))
        y = int(round(y0 + c * uy))
        if rad == 0:
            if 0 <= y < H and 0 <= x < W:
                img[y, x] = bgr
        else:
            for yy in range(y - rad, y + rad + 1):
                for xx in range(x - rad, x + rad + 1):
                    if 0 <= yy < H and 0 <= xx < W:
                        img[yy, xx] = bgr

# -- Manual texture mapping per assignment -------------------------------------
def apply_texture_manual(
    img_bgr, Km, Rm, Tm,
    V1, V2, V4,                    # 3D vertices (mm) defining the textured face V1-V2-V3-V4
    c_center, r_center, p_scale,
    tex_bgr
):
    """
    Manually map the texture to the V1-V2-V4 face:
    - i (texture row) goes along V2 - V1 (downward in texture ↔ +y in cube)
    - j (texture col) goes along V4 - V1 (rightward in texture ↔ +x in cube)
    For each tex (i,j), compute 3D X = V1 + a*(V2-V1) + b*(V4-V1), project, then write pixel.
    """
    H, W = img_bgr.shape[:2]
    hT, wT = tex_bgr.shape[:2]

    v21 = (V2 - V1).astype(np.float64)
    v41 = (V4 - V1).astype(np.float64)

    # Optional: back-face culling to skip when the face is looking away
    # Compute face normal in world, then rotate to camera; view is along -Zc.
    n_world = np.cross(v21, v41)
    n_cam = Rm @ n_world
    if n_cam[2] >= 0:  # normal not facing camera; you can comment this out if you want it always drawn
        pass  # still allow drawing; remove this pass+if if you want strict culling

    # Iterate texels
    for i in range(hT):
        a = i / (hT - 1.0 if hT > 1 else 1.0)  # 0..1 along V1->V2
        for j in range(wT):
            b = j / (wT - 1.0 if wT > 1 else 1.0)  # 0..1 along V1->V4
            Xw = V1 + a * v21 + b * v41  # 3D point on the face (mm)

            q_mm = Map2Da(Km, Rm, Tm, Xw)
            if q_mm is None:
                continue
            col, row = MapIndex(q_mm, c_center, r_center, p_scale)

            if 0 <= row < H and 0 <= col < W:
                img_bgr[row, col] = tex_bgr[i, j]
    return img_bgr

# -- Main animation / AR -------------------------------------------------------
def main():
    # Load texture (color) and background (grayscale -> color)
    tex_map = cv2.imread('einstein132.jpg', cv2.IMREAD_COLOR)
    if tex_map is None:
        print("Texture image 'einstein132.jpg' not found.")
        sys.exit(1)

    bg_img = cv2.imread('background.jpg', cv2.IMREAD_GRAYSCALE)
    if bg_img is None:
        print("Background image 'background.jpg' not found.")
        sys.exit(1)
    bg_img = cv2.resize(bg_img, (600, 600))
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)

    # -- Cube geometry (mm)
    len_v = 10.0
    V_1 = np.array([0.0,     0.0,     0.0    ])
    V_2 = np.array([0.0,     len_v,   0.0    ])
    V_3 = np.array([len_v,   len_v,   0.0    ])
    V_4 = np.array([len_v,   0.0,     0.0    ])
    V_5 = np.array([len_v,   0.0,     len_v  ])
    V_6 = np.array([0.0,     len_v,   len_v  ])
    V_7 = np.array([0.0,     0.0,     len_v  ])
    V_8 = np.array([len_v,   len_v,   len_v  ])
    vertices = [V_1, V_2, V_3, V_4, V_5, V_6, V_7, V_8]

    # -- Camera / motion parameters (mm, mm/s, mm/s^2, deg/s)
    T0      = np.array([-20.0, -25.0, 500.0])   # initial translation (camera coords)
    focal   = 40.0                               # mm
    vel     = np.array([ 2.0,   9.0,   7.0  ])
    accel   = np.array([ 0.0,  -0.8,  0.0  ])
    th0     = 0.0                                # degrees
    w0      = 20.0                               # degrees/s
    p_scale = 0.01                               # mm per pixel
    Rmax, Cmax = 600, 600
    r_center, c_center = np.round(Rmax/2), np.round(Cmax/2)

    Km = np.array([[focal, 0,     0],
                   [0,     focal, 0],
                   [0,     0,     1]], dtype=np.float64)

    # -- Rotation axis (unit) from V1->V8, then skew-symmetric N
    u_dir = (V_8 - V_1).astype(np.float64)
    u_dir /= np.linalg.norm(u_dir)
    Nm = np.array([[0,        -u_dir[2],  u_dir[1]],
                   [u_dir[2],  0,        -u_dir[0]],
                   [-u_dir[1], u_dir[0],  0      ]], dtype=np.float64)

    # -- Correct 12 edges (0-based indices)
    edges = [
        (0,1), (1,2), (2,3), (3,0),     # front face V1..V4
        (6,5), (5,7), (7,4), (4,6),     # back  face V7,V6,V8,V5
        (0,6), (1,5), (2,7), (3,4)      # verticals
    ]

    # --- Infinite animation timing ---
    fps = 30.0
    dt  = 1.0 / fps

    # State for incremental motion
    theta = th0
    T     = T0.astype(np.float64).copy()
    v     = vel.astype(np.float64).copy()

    # Keep the cube in view: simple "bounce" limits in camera coords (mm)
    x_lim = (-80.0, 80.0)
    y_lim = (-80.0, 80.0)
    z_lim = (380.0, 620.0)  # around 500 mm

    while True:
        # rotation
        theta = (theta + w0 * dt) % 360.0
        thRad = np.deg2rad(theta)
        Rm = np.eye(3) + np.sin(thRad) * Nm + (1.0 - np.cos(thRad)) * (Nm @ Nm)

        # translation (integrate)
        T += v * dt + 0.5 * accel * (dt ** 2)
        v += accel * dt

        # bounce at bounds
        if T[0] < x_lim[0]:
            T[0] = x_lim[0]; v[0] = abs(v[0])
        elif T[0] > x_lim[1]:
            T[0] = x_lim[1]; v[0] = -abs(v[0])
        if T[1] < y_lim[0]:
            T[1] = y_lim[0]; v[1] = abs(v[1])
        elif T[1] > y_lim[1]:
            T[1] = y_lim[1]; v[1] = -abs(v[1])
        if T[2] < z_lim[0]:
            T[2] = z_lim[0]; v[2] = abs(v[2])
        elif T[2] > z_lim[1]:
            T[2] = z_lim[1]; v[2] = -abs(v[2])

        # render
        frame = bg_img.copy()

        # project vertices
        px_verts = []
        for V in vertices:
            q_mm = Map2Da(Km, Rm, T, V)
            if q_mm is None:
                px_verts.append(None)
            else:
                px_verts.append(MapIndex(q_mm, c_center, r_center, p_scale))

        # wireframe
        for e in edges:
            drawLine(frame, px_verts[e[0]], px_verts[e[1]], bgr=(0,0,255), thickness=2)

        # texture front face
        frame = apply_texture_manual(
            frame, Km, Rm, T,
            V_1, V_2, V_4,
            c_center, r_center, p_scale,
            tex_map
        )

        # display (press 'q' or Esc to quit on desktop)
        key = show_img(frame)
        if key in (ord('q'), 27):
            break

    # desktop cleanup
    try:
        from google.colab.patches import cv2_imshow  # type: ignore
    except Exception:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
