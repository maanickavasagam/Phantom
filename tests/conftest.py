import cv2
import numpy as np
import pytest


def make_doc(width=1240, height=1754, angle=0.0, blur=False):
    img = np.full((height, width, 3), 255, np.uint8)
    for i in range(25):
        y = 150 + i * 55
        cv2.putText(img, f"INVOICE LINE {i:02d}  AMOUNT 1,{i:03d}.00", (100, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    if angle:
        m = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (width, height), borderValue=(255, 255, 255))
    if blur:
        img = cv2.GaussianBlur(img, (31, 31), 12)
    return img


@pytest.fixture
def doc_png(tmp_path):
    p = tmp_path / "doc.png"
    cv2.imwrite(str(p), make_doc())
    return p


@pytest.fixture
def skewed_png(tmp_path):
    p = tmp_path / "skewed.png"
    cv2.imwrite(str(p), make_doc(angle=4.0))
    return p


@pytest.fixture
def blurry_png(tmp_path):
    p = tmp_path / "blurry.png"
    cv2.imwrite(str(p), make_doc(blur=True))
    return p
