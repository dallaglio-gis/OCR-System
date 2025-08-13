
"""
Simple Image Preprocessor for GPT-5 OCR System
Basic preprocessing since GPT-5 can handle complex images well
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import base64
from typing import Tuple, Optional

class SimpleImagePreprocessor:
    """Simple image preprocessing for GPT-5 OCR"""
    
    def __init__(self, max_size: Tuple[int, int] = (2048, 2048)):
        self.max_size = max_size
    
    def preprocess_image(self, image_path: str) -> str:
        """
        Simple preprocessing and convert to base64 for GPT-5
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert PIL to OpenCV for orientation processing
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Resize if too large (working with BGR array)
            h, w = bgr.shape[:2]
            if w > self.max_size[0] or h > self.max_size[1]:
                scale = min(self.max_size[0]/w, self.max_size[1]/h)
                new_w, new_h = int(w * scale), int(h * scale)
                bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Auto-orient layout to ensure proper landscape orientation
            bgr = self._auto_orient_layout(bgr)
            
            # Apply four-point dewarp to correct perspective distortion
            bgr = self._four_point_dewarp(bgr)
            
            # Convert back to PIL for enhancement
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            
            # Basic enhancement
            img = self._enhance_image(img)
            
            # Convert to base64
            return self._image_to_base64(img)
            
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    def _four_point_dewarp(self, bgr: np.ndarray) -> np.ndarray:
        """
        Apply four-point perspective correction to dewarp the page.
        Finds the largest quadrilateral contour and warps it to a rectangle.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Strong edges to find page boundaries
        edges = cv2.Canny(gray, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not cnts:
            return bgr
            
        # Find the largest contour (assumed to be the page)
        cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        # Need exactly 4 points for perspective transform
        if len(approx) != 4:
            return bgr
            
        pts = approx.reshape(4, 2).astype(np.float32)
        
        # Order points: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)[:, 0]
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        
        # Calculate width and height for output rectangle
        wA = np.linalg.norm(br - bl)
        wB = np.linalg.norm(tr - tl)
        hA = np.linalg.norm(tr - br)
        hB = np.linalg.norm(tl - bl)
        W = int(max(wA, wB))
        H = int(max(hA, hB))
        
        # Perspective transform to rectangle
        dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), dst)
        out = cv2.warpPerspective(bgr, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return out

    def _enhance_image(self, img: Image.Image) -> Image.Image:
        """Apply basic image enhancement"""
        # Slight contrast enhancement
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        
        # Slight sharpness enhancement
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.1)
        
        return img
    
    def _dominant_line_orientation(self, bgr: np.ndarray) -> str:
        """
        Returns 'horizontal' or 'vertical' depending on the total length of long lines.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
        h, w = gray.shape[:2]
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=160,
            minLineLength=int(0.60 * max(h, w)), maxLineGap=8
        )
        if lines is None:
            # fall back to aspect assumption (forms are landscape)
            return "horizontal" if w >= h else "vertical"

        horiz_len = 0
        vert_len  = 0
        for x1,y1,x2,y2 in lines[:,0]:
            dx, dy = abs(x2-x1), abs(y2-y1)
            if dy <= 2:         # near horizontal
                horiz_len += dx
            elif dx <= 2:       # near vertical
                vert_len += dy
            else:
                # classify by slope: small slope -> horizontal
                if dy/dx < 0.15:
                    horiz_len += int(np.hypot(dx,dy))
                elif dx/dy < 0.15:
                    vert_len  += int(np.hypot(dx,dy))
        return "horizontal" if horiz_len >= vert_len else "vertical"

    def _auto_orient_layout(self, bgr: np.ndarray) -> np.ndarray:
        """
        Ensure the page is in landscape with long grid lines horizontal.
        If vertical lines dominate, rotate -90°.
        """
        ori = self._dominant_line_orientation(bgr)
        h, w = bgr.shape[:2]
        if ori == "vertical" or h > w:
            # rotate 90° counter-clockwise to make horizontals dominant
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return bgr

    def _image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
