import warnings
import numpy as np
import dlib
import cv2
from skimage import feature
from PIL import Image

print("Dlib using CUDA?: ", dlib.DLIB_USE_CUDA)
import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def draw_edge(im, x, y, bw=1, color=(255, 255, 255), draw_end_points=False):
    r"""Set colors given a list of x and y coordinates for the edge.
    Args:
        im (HxWxC numpy array): Canvas to draw.
        x (1D numpy array): x coordinates of the edge.
        y (1D numpy array): y coordinates of the edge.
        bw (int): Width of the stroke.
        color (list or tuple of int): Color to draw.
        draw_end_points (bool): Whether to draw end points of the edge.
    """
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # Draw edge.
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h - 1, y + i))
                xx = np.maximum(0, np.minimum(w - 1, x + j))
                set_color(im, yy, xx, color)

        # Draw endpoints.
        if draw_end_points:
            for i in range(-bw * 2, bw * 2):
                for j in range(-bw * 2, bw * 2):
                    if (i**2) + (j**2) < (4 * bw**2):
                        yy = np.maximum(
                            0, np.minimum(h - 1, np.array([y[0], y[-1]]) + i)
                        )
                        xx = np.maximum(
                            0, np.minimum(w - 1, np.array([x[0], x[-1]]) + j)
                        )
                        set_color(im, yy, xx, color)


def set_color(im, yy, xx, color):
    r"""Set pixels of the image to the given color.
    Args:
        im (HxWxC numpy array): Canvas to draw.
        xx (1D numpy array): x coordinates of the pixels.
        yy (1D numpy array): y coordinates of the pixels.
        color (list or tuple of int): Color to draw.
    """
    if type(color) != list and type(color) != tuple:
        color = [color] * 3
    if len(im.shape) == 3 and im.shape[2] == 3:
        if (im[yy, xx] == 0).all():
            im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = color[0], color[1], color[2]
        else:
            for c in range(3):
                im[yy, xx, c] = ((im[yy, xx, c].astype(float) + color[c]) / 2).astype(
                    np.uint8
                )
    else:
        im[yy, xx] = color[0]


def interp_points(x, y):
    r"""Given the start and end points, interpolate to get a curve/line.
    Args:
        x (1D array): x coordinates of the points to interpolate.
        y (1D array): y coordinates of the points to interpolate.
    Returns:
        (dict):
          - curve_x (1D array): x coordinates of the interpolated points.
          - curve_y (1D array): y coordinates of the interpolated points.
    """
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interp_points(y, x)
        if curve_y is None:
            return None, None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if len(x) < 3:
                    popt, _ = curve_fit(linear, x, y)
                else:
                    popt, _ = curve_fit(func, x, y)
                    if abs(popt[0]) > 1:
                        return None, None
            except Exception:
                return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], int(np.round(x[-1] - x[0])))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)


def func(x, a, b, c):
    r"""Quadratic fitting function."""
    return a * x**2 + b * x + c


def linear(x, a, b):
    r"""Linear fitting function."""
    return a * x + b


def connect_face_keypoints(
    keypoints, img_size, cropped_and_resized_img, use_canny_edges
):
    r"""Connect the face keypoints to edges and draw the sketch.
    Args:
        img_size (int tuple): Height and width of the input image.
        keypoints (NxKx2 numpy array): Facial landmarks (with K keypoints).
        img (HxWx3 numpy array): Original image.
    Returns:
        (HxWxC numpy array): Drawn label map.
    """
    add_upper_face = True
    H, W = img_size[0], img_size[1]

    # Mapping from keypoint index to facial part.
    part_list = [
        [
            list(range(0, 17)) + ((list(range(68, 83)) + [0]) if add_upper_face else [])
        ],  # ai_emoji
        [range(17, 22)],  # right eyebrow
        [range(22, 27)],  # left eyebrow
        [[28, 31], range(31, 36), [35, 28]],  # nose
        [[36, 37, 38, 39], [39, 40, 41, 36]],  # right eye
        [[42, 43, 44, 45], [45, 46, 47, 42]],  # left eye
        [
            range(48, 55),
            [54, 55, 56, 57, 58, 59, 48],
            range(60, 65),
            [64, 65, 66, 67, 60],
        ],  # mouth and tongue
    ]
    if add_upper_face:
        pts = keypoints[:17, :].astype(np.int32)
        baseline_y = (pts[0:1, 1] + pts[-1:, 1]) / 2
        upper_pts = pts[1:-1, :].copy()
        upper_pts[:, 1] = baseline_y + (baseline_y - upper_pts[:, 1]) * 2 // 3
        keypoints = np.vstack((keypoints, upper_pts[::-1, :]))

    edge_len = 3  # Interpolate 3 keypoints to form a curve when drawing edges.
    bw = max(1, H // 256)  # Width of the stroke.

    # Edge map for the face region from keypoints.
    im_edges = np.zeros((H, W, 1), np.uint8)
    for edge_list in part_list:
        for e, edge in enumerate(edge_list):

            # Divide a long edge into multiple small edges when drawing.
            for i in range(0, max(1, len(edge) - 1), edge_len - 1):
                sub_edge = edge[i : i + edge_len]
                x = keypoints[sub_edge, 0]
                y = keypoints[sub_edge, 1]

                # Interp keypoints to get the curve shape.
                curve_x, curve_y = interp_points(x, y)
                draw_edge(im_edges, curve_x, curve_y, bw=bw)

    im_edges = im_edges.astype(np.float32)[:, :, 0]

    if str(use_canny_edges).lower() == "true":  # "TRUE", "True", "true", True
        label_list = [1, 2, 2, 3, 4, 4, 5, 6]
        part_labels = np.zeros(img_size, np.uint8)
        for p, edge_list in enumerate(part_list):
            indices = [item for sublist in edge_list for item in sublist]
            pts = keypoints[indices, :].astype(np.int32)
            cv2.fillPoly(part_labels, pts=[pts], color=label_list[p])

        canny_edges = feature.canny(
            np.array(Image.fromarray(cropped_and_resized_img).convert("L"))
        )
        canny_edges = canny_edges * (part_labels == 0)

        # black borders
        canny_edges[:, [0, 1, -2, -1]] = canny_edges[[0, 1, -2, -1]] = 0

        im_edges += (canny_edges * 255).astype(np.uint8)
    elif str(use_canny_edges).lower() == "false":  # "FALSE", "False", "false", False
        pass
    else:
        raise Exception(f"Not a valid use_canny_edges {use_canny_edges}")

    return im_edges


def get_dlib_keypoints_from_image(img):
    r"""Get face keypoints from an image.
    Args:
        img (H x W x 3 numpy array): Input images.
        predictor_path (str): Path to the predictor model.
    """

    keypoints = np.zeros([68, 2], dtype=int)
    dets = detector(img, 1)
    if len(dets) > 0:
        # Only returns the first face.
        shape = predictor(img, dets[0])
        for b in range(68):
            keypoints[b, 0] = shape.part(b).x
            keypoints[b, 1] = shape.part(b).y
    return keypoints


def get_face_bbox_from_image(keypoints, img_size):
    r"""Get the bbox coordinates for face region.
    Args:
        keypoints (Nx2 tensor): Facial landmarks.
        img_size (int tuple): Height and width of the input image.
    Returns:
        crop_coords (list of int): bbox for face region.
    """
    min_y, max_y = int(keypoints[:, 1].min()), int(keypoints[:, 1].max())
    min_x, max_x = int(keypoints[:, 0].min()), int(keypoints[:, 0].max())
    x_cen, y_cen = (min_x + max_x) // 2, (min_y + max_y) // 2
    H = img_size[0]
    W = img_size[1]
    w = h = max_x - min_x

    # Get the cropping coordinates.
    x_cen = max(w, min(W - w, x_cen))
    y_cen = max(h * 1.25, min(H - h * 0.75, y_cen))

    min_x = x_cen - w
    min_y = y_cen - h * 1.25
    max_x = min_x + w * 2
    max_y = min_y + h * 2

    crop_coords = [min_y, max_y, min_x, max_x]
    return [int(x) for x in crop_coords]


def crop_and_resize(img, coords, size=None, method="bilinear"):
    r"""Crop the image using the given coordinates and resize to target size.
    Args:
        img (tensor or list of tensors): Input image.
        coords (list of int): Pixel coordinates to crop.
        size (list of int): Output size.
        method (str): Interpolation method.
    Returns:
        img (tensor or list of tensors): Output image.
    """
    min_y, max_y, min_x, max_x = coords

    cropped_and_resized_image = img[min_y:max_y, min_x:max_x].copy()
    if size is not None:
        cropped_and_resized_image = (
            torch.tensor(cropped_and_resized_image, dtype=torch.float)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
        )
        if method == "nearest":
            cropped_and_resized_image = F.interpolate(
                cropped_and_resized_image, size=size, mode=method
            )
        else:
            cropped_and_resized_image = F.interpolate(
                cropped_and_resized_image, size=size, mode=method, align_corners=False
            )
        cropped_and_resized_image = (
            cropped_and_resized_image[0].permute(1, 2, 0).numpy().astype(np.uint8)
        )
    return cropped_and_resized_image


def extract_face_edge_map_from_single_image(img, target_h_w, use_canny_edges):
    """Crops, resizes the input image and extracts black and white
        face edge map from a 2D RGB face image.
    Args:
        img: Image (HxWxC).
    Returns:
        face_edge_map: Face edge image (target_h_w x target_h_w).
    """
    keypoints = get_dlib_keypoints_from_image(img)
    crop_coords = get_face_bbox_from_image(keypoints, img.shape)
    cropped_and_resized_img = crop_and_resize(
        img, crop_coords, (target_h_w, target_h_w)
    )
    keypoints = get_dlib_keypoints_from_image(cropped_and_resized_img)
    face_edge_map = connect_face_keypoints(
        keypoints, (target_h_w, target_h_w), cropped_and_resized_img, use_canny_edges
    )
    return cropped_and_resized_img, face_edge_map
