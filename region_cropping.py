# region_cropping.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
import cv2
import numpy as np

def ensure_landscape(img_bgr: np.ndarray):
    """If our metrics detector places the metrics region high in the image,
    assume rotation is wrong, rotate CCW, and return rotated image."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=160,
        minLineLength=int(0.60 * max(h, w)), maxLineGap=8
    )
    # quick score: vertical vs horizontal
    vert = 0; horiz = 0
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            if abs(y1-y2) <= 2: horiz += abs(x2-x1)
            elif abs(x1-x2) <= 2: vert += abs(y2-y1)
    # rotate if vertical dominates or portrait
    if vert > horiz * 1.25 or h > w:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_bgr

@dataclass
class CropResult:
    image: np.ndarray
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    row_bboxes: Optional[List[Tuple[int,int,int,int]]] = None  # for activity table rows

def _clip(v, lo, hi): return max(lo, min(hi, v))

def _auto_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def _find_horizontal_lines(gray: np.ndarray, min_len_ratio=0.65) -> List[Tuple[int,int]]:
    # returns list of (y, length)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=160, minLineLength=int(gray.shape[1]*min_len_ratio), maxLineGap=8)
    out = []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            if abs(y1 - y2) <= 2:  # horizontal
                out.append((y1, abs(x2-x1)))
    out.sort(key=lambda t: t[0])
    return out

def isolate_activity_text_block(act_crop_bgr: np.ndarray) -> np.ndarray:
    """
    Isolate the activity text block from the daily maintenance checklist.
    Uses vertical projection to find the separation between activity table and checklist.
    """
    gray = cv2.cvtColor(act_crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # Light binarization to highlight text regions
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = bw.shape
    
    # Vertical projection of dark pixels (text content)
    proj = (255 - bw).sum(axis=0)
    
    # Find the big valley between activity block and checklist
    # Heuristic: look for the valley after the midline (around 55% of width)
    cut = int(0.55 * w)
    window = proj[cut:]
    
    if window.size == 0:
        return act_crop_bgr
    
    # Find the lowest density column region using a 5-pixel sliding window
    k = 5
    if window.size < k:
        return act_crop_bgr
        
    conv = np.convolve(window, np.ones(k), 'valid') / k
    split = cut + int(np.argmin(conv))
    
    # Keep the left part (activity table), discard the right part (checklist)
    left = act_crop_bgr[:, :split]
    return left

def crop_header(img_bgr: np.ndarray) -> CropResult:
    h, w = img_bgr.shape[:2]
    top = int(0.00*h); bot = int(0.23*h)  # generous; header is usually < 23%
    y1, y2 = _clip(top,0,h-1), _clip(bot,1,h)
    x1, x2 = 0, w
    crop = img_bgr[y1:y2, x1:x2].copy()
    return CropResult(crop, (x1,y1,x2-x1,y2-y1))

def crop_metrics(img_bgr: np.ndarray) -> CropResult:
    h, w = img_bgr.shape[:2]
    # Find a dense horizontal grid near bottom
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = _auto_contrast(gray)
    lines = _find_horizontal_lines(gray)
    if not lines:
        # fallback: last 28% of page
        y1 = int(0.72*h); y2 = h
    else:
        # pick the first strong line in the bottom 35% as the top of metrics block
        bottom_th = int(0.65*h)
        candidates = [y for y,_ in lines if y >= bottom_th]
        y1 = min(candidates) if candidates else int(0.72*h)
        y2 = h
    x1, x2 = 0, w
    crop = img_bgr[y1:y2, x1:x2].copy()
    return CropResult(crop, (x1,y1,x2-x1,y2-y1))

def crop_activity(img_bgr: np.ndarray, header: CropResult, metrics: CropResult) -> CropResult:
    h, w = img_bgr.shape[:2]
    top = header.bbox[1] + header.bbox[3] + int(0.01*h)
    bot = metrics.bbox[1] - int(0.01*h)
    y1, y2 = _clip(top,0,h-1), _clip(bot,1,h)
    x1, x2 = int(0.00*w), int(0.98*w)  # slight right margin trim
    crop = img_bgr[y1:y2, x1:x2].copy()

    # estimate row bands for re-ask: detect horizontal lines within this block
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = _auto_contrast(gray)
    lines = _find_horizontal_lines(gray, min_len_ratio=0.75)
    rows = []
    if lines:
        # turn horizontal line y’s into band segments
        ys = sorted(set([y for y,_ in lines]))
        # merge near-duplicate lines
        merged = []
        for y in ys:
            if not merged or y - merged[-1] > 6:
                merged.append(y)
        # intervals between lines
        for i in range(len(merged)-1):
            yA, yB = merged[i], merged[i+1]
            if (yB - yA) >= 18:  # minimum row height
                rows.append((0, yA, crop.shape[1], yB - yA))
    return CropResult(crop, (x1,y1,x2-x1,y2-y1), rows or None)
