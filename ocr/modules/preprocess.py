from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Union

import cv2
import numpy as np

class Preprocessor:
    def __init__(self, output_dir: Union[str, Path] = "figures"):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def deskew_page(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:

        if image is None or image.size == 0:
            raise ValueError("deskew_page: image is empty")

        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

        _, thresh = cv2.threshold(
            gray_blur,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        coords = cv2.findNonZero(thresh)
        if coords is None:
            return image

        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.1 or abs(angle) > max_angle:
            return image

        (h, w) = image.shape[:2]
        center = (w / 2.0, h / 2.0)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # Điều chỉnh tâm
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]

        rotated = cv2.warpAffine(
            image,
            M,
            (nW, nH),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated

    @staticmethod
    def enhance_contrast(
        image: np.ndarray,
        clip_limit: float = 3.0,
        tile_grid_size: int = 8,
        sharpen_strength: float = 1.5,
    ) -> np.ndarray:

        if image is None or image.size == 0:
            raise ValueError("enhance_contrast: image is empty")

        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid_size, tile_grid_size),
        )
        contrast = clahe.apply(gray)

        blur = cv2.GaussianBlur(contrast, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(
            contrast, sharpen_strength,
            blur, -(sharpen_strength - 1.0),
            0,
        )

        return sharpened

    def crop_images(
        self,
        image: np.ndarray,
        bboxes: Sequence[Sequence[int]],
        prefix: str = "figure",
        ext: str = ".png",
    ) -> List[Path]:

        if image is None or image.size == 0:
            raise ValueError("crop_images: image is empty")

        h, w = image.shape[:2]
        saved_paths: List[Path] = []

        for idx, box in enumerate(bboxes, start=1):
            if len(box) != 4:
                raise ValueError(
                    f"crop_images: bbox #{idx} phải có 4 phần tử, nhận được {len(box)}"
                )

            x1, y1, x2, y2 = map(int, box)

            if x2 <= x1 or y2 <= y1:
                x, y, bw, bh = x1, y1, x2, y2
                x1, y1 = x, y
                x2, y2 = x + bw, y + bh

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            file_name = f"{prefix}_{idx:03d}{ext}"
            out_path = self.output_dir / file_name
            cv2.imwrite(str(out_path), crop)
            saved_paths.append(out_path)

        return saved_paths


def deskew_page(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    return Preprocessor.deskew_page(image, max_angle=max_angle)


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: int = 8,
    sharpen_strength: float = 1.5,
) -> np.ndarray:
    return Preprocessor.enhance_contrast(
        image,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
        sharpen_strength=sharpen_strength,
    )


def crop_images(
    image: np.ndarray,
    bboxes: Sequence[Sequence[int]],
    prefix: str = "figure",
    ext: str = ".png",
    output_dir: Union[str, Path] = "figures",
) -> List[Path]:
    pre = Preprocessor(output_dir=output_dir)
    return pre.crop_images(image, bboxes, prefix=prefix, ext=ext)