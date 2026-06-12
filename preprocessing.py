

import cv2
import numpy as np


FIELD_HSV_LOWER   = np.array([30, 40, 40])
FIELD_HSV_UPPER   = np.array([85, 255, 255])
GAUSSIAN_KERNEL   = (5, 5)
GAUSSIAN_SIGMA    = 1.5
MORPH_KERNEL_SIZE = (5, 5)

def apply_gaussian_filter(frame: np.ndarray) -> np.ndarray:
    """Suaviza artefatos de compressão H.264 do broadcast."""
    return cv2.GaussianBlur(frame, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)


def get_field_mask(frame: np.ndarray) -> np.ndarray:
    """Retorna máscara binária do gramado via espaço HSV."""
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask   = cv2.inRange(hsv, FIELD_HSV_LOWER, FIELD_HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def preprocess_frame(frame: np.ndarray) -> tuple:
    """
    Executa o pipeline completo de pré-processamento.
    Retorna (frame_filtrado, mascara_do_campo).
    """
    denoised   = apply_gaussian_filter(frame)
    field_mask = get_field_mask(denoised)
    return denoised, field_mask
