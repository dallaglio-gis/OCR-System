
"""
GPT-5 OCR Engine for Diamond Drilling Forms
Uses OpenAI's GPT model for intelligent OCR processing
"""

import openai
import json
import logging
import base64
import cv2
import numpy as np
import re
from typing import Dict, Any, Optional, List, Tuple
from config import OCRConfig, FIELD_MAPPINGS, RIG_MAPPINGS, TYPO_TO_DIGIT
from region_cropping import crop_header, crop_activity, crop_metrics, CropResult, ensure_landscape, isolate_activity_text_block

# --------- Utility functions for image handling ---------
def _b64_jpeg(arr: np.ndarray, quality=95) -> str:
    """Convert numpy array to base64 JPEG string"""
    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def _img(arr: np.ndarray):  # OpenAI image input format
    """Create OpenAI image input format from numpy array"""
    return {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{_b64_jpeg(arr)}"}}

def _sys(s: str): 
    """Create system message"""
    return {"role":"system","content":s}

def _usr(parts: List[Dict[str,Any]]): 
    """Create user message with parts"""
    return {"role":"user","content": parts}

# --------- Region-specific prompts ---------
HEADER_SYS = (
"Extract only these fields from the header crop and return pure JSON:\n"
"{\"date\": \"YYYY-MM-DD\", \"shift\": string, \"level\": string, \"cubby_id\": string, \"rig_id\": string, \"hole_id\": string}\n"
"- Date must be in ISO YYYY-MM-DD (convert from local format if needed)\n"
"- Return null for any missing field. No extra keys."
)

ACTIVITY_SYS = (
"From this activity table crop, read the TIME columns and the corresponding activity text column.\n"
"Return JSON only: {\"rows\": [{\"from\": string, \"to\": string, \"text\": string}, ...]}\n"
"Rules: 24h times; accept ':', '.', or no separator; handle midnight rollover when computing durations later.\n"
"Keep text exactly as written (trim extra spaces). No extra keys."
)

METRICS_SYS = (
"From the bottom metrics crop, return JSON only with:\n"
"{\"bhid\":string,\"from_depth\":float,\"to_depth\":float,\"total_depth\":float,\n"
" \"method\":string,\"bit_type\":string,\"size_mm\":string,\n"
" \"meter_start\":float,\"meter_end\":float,\"cum_meters\":float,\n"
" \"comments\":string,\"driller\":string,\"geotech\":string,\"shift_supervisor\":string}\n"
"Rules: numbers may appear as '5.10','5,10','5-10' – normalise to dot-decimals.\n"
"Use null for blanks. Do not infer or add keys."
)

class GPT5OCREngine:
    """GPT-5 powered OCR engine for drilling forms"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.client = openai.OpenAI(api_key=config.openai_api_key)
        self.logger = logging.getLogger(__name__)
        # Store model for convenience
        self.model = config.gpt_model
        
    def _canon_id(self, s: str) -> str:
        """Canonicalize ID strings for comparison"""
        return (s or "").upper().strip().replace(" ", "").replace("-", "_")

    def _fix_left_token(self, left: str) -> str:
        """Replace lookalikes in the left part (before W...)"""
        fixed = []
        for ch in left:
            fixed.append(TYPO_TO_DIGIT.get(ch, ch))
        out = "".join(fixed)
        # ensure it contains a 'W' segment; if 'V' found near W, leave; else coerce
        return out

    def _canonicalize_bhid(self, s: str) -> str | None:
        """Canonicalize BHID with smart typo correction"""
        if not s: return None
        x = self._canon_id(s)
        # Normalize separators
        x = x.replace("__","_")
        # Try to split into <LEFT>_<RIGHT> if missing underscore
        if "_" not in x:
            # common patterns: 6W19EV01 → 6W19_EV01 ; 6W21_02 already good
            m = re.match(r"^([0-9GSIOBZA]{1,2}[AW]?W[0-9]{1,2})(.*)$", x)
            if m and m.group(2):
                x = f"{m.group(1)}_{m.group(2).lstrip('_')}"
        parts = x.split("_", 1)
        left = parts[0]
        right = parts[1] if len(parts) > 1 else ""
        # Fix likely misreads in left segment (GW19 → 6W19, SW9 → 5W9, etc.)
        left = self._fix_left_token(left)
        left = left.replace("GW", "6W").replace("SW", "5W").replace("OW", "0W").replace("IW", "1W")
        # If left accidentally becomes e.g. W19, try prepending '6'
        if left.startswith("W") and len(left) > 1:
            left = "6" + left
        # Normalize suffix with smart O/0 conversion and padding
        right = self._normalize_bhid_suffix(right)
        # Rebuild
        x = left if not right else f"{left}_{right}"
        return x

    def _canonicalize_cubby(self, s: str) -> str | None:
        """Canonicalize Cubby ID with smart typo correction"""
        if not s: return None
        x = self._canon_id(s)
        # Fix common misreads: GW19 → 6W19, SW9 → 5W9, etc.; remove trailing underscores
        x = x.strip("_").replace("GW", "6W").replace("SW", "5W").replace("OW", "0W").replace("IW", "1W")
        return x

    def _normalize_bhid_suffix(self, right: str) -> str:
        """
        Normalise the suffix portion of BHID:
        - 'EVO2' → 'EV02' (O→0 + pad)
        - 'EV2'  → 'EV02'
        - '2'    → '02'
        Leaves other tokens as-is.
        """
        if not right:
            return right
        x = right.upper().replace("O", "0").strip("_")
        m = re.match(r"^(?:EV)?(\d{1,2})$", x)
        if m:
            n = int(m.group(1))
            return f"EV{n:02d}" if x.startswith("EV") else f"{n:02d}"
        return x

    def _context_digit_from_level_or_cubby(self, d: Dict[str, Any]) -> str:
        """Try to infer the leading digit from Level '6L' or Cubby '6W..' context"""
        for key in ("level", "cubby_id"):
            v = (d.get(key) or "").upper().replace(" ", "").replace("-", "")
            m = re.search(r"([0-9])W|([0-9])L", v)
            if m:
                return m.group(1) or m.group(2)
        return "6"  # Default fallback

    def _canonicalize_bhid_using_context(self, d: Dict[str, Any]):
        """Enhanced BHID canonicalization using Level/Cubby context"""
        bhid = d.get("bhid") or d.get("hole_id")
        if not bhid:
            return
            
        bhid = self._canonicalize_bhid(bhid)
        if not bhid:
            return
            
        # If left chunk starts with W (e.g., W19), prepend inferred digit
        left = bhid.split("_", 1)[0]
        if left.startswith("W"):
            digit = self._context_digit_from_level_or_cubby(d)
            bhid = f"{digit}{bhid}"
            d.setdefault("flags", []).append("bhid:context_digit_inferred")
            
        d["bhid"] = bhid
        d["hole_id"] = bhid

    def _canonicalize_cubby_using_context(self, d: Dict[str, Any]):
        """Enhanced Cubby canonicalization using Level context"""
        cub = d.get("cubby_id")
        if not cub:
            return
            
        d["cubby_id"] = self._canonicalize_cubby(cub)
        
        # If we only have W19 in cubby, append leading digit from Level (else 6)
        x = d["cubby_id"]
        if x and x.startswith("W"):
            digit = self._context_digit_from_level_or_cubby(d)
            d["cubby_id"] = f"{digit}{x}"
            d.setdefault("flags", []).append("cubby:context_digit_inferred")

    def _to_float(self, v) -> Optional[float]:
        """Robust numeric parser handling 5.10, 5-10, 5,10 formats and stripping units"""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        
        s = str(v).strip()
        if not s:
            return None
            
        # Strip common units and extra characters
        s = re.sub(r'[mM]\s*$', '', s)  # Remove trailing 'm' or 'M'
        s = re.sub(r'[^\d.,\-]', '', s)  # Keep only digits, periods, commas, dashes
        
        # Handle different decimal separators: 5,10 -> 5.10, 5-10 -> 5.10
        if ',' in s and '.' not in s:
            s = s.replace(',', '.')
        elif '-' in s and not s.startswith('-'):
            # Handle 5-10 format (not negative numbers)
            s = s.replace('-', '.')
            
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    def _binarize(self, bgr: np.ndarray) -> np.ndarray:
        """Apply heavy binarization for dual-read consensus"""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 7)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    def dual_read_cell(self, sys_prompt: str, roi_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Dual-read numeric cells with normal and heavy binarization for consensus.
        Returns median value and flags disagreements.
        """
        # Read with normal contrast
        normal_result = self._ask_json(sys_prompt, roi_bgr)
        
        # Read with heavy binarization
        binary_result = self._ask_json(sys_prompt, self._binarize(roi_bgr))
        
        # Extract numeric values from both results
        vals = []
        for result in (normal_result, binary_result):
            v = result.get("value") or result.get("text")
            if v is None:
                continue
            parsed = self._to_float(v)
            if parsed is not None:
                vals.append(parsed)
        
        if len(vals) == 0:
            return {"value": None, "confidence": 0.0, "flag": "dual_read:no_values"}
        elif len(vals) == 1:
            return {"value": vals[0], "confidence": 0.7, "flag": "dual_read:single_value"}
        else:
            median_val = float(np.median(vals))
            disagreement = abs(vals[0] - vals[1]) > 0.05
            return {
                "value": median_val, 
                "confidence": 0.6 if disagreement else 0.9,
                "flag": "dual_read:disagreement" if disagreement else None
            }

    def crop_right_of_label(self, metrics_bgr: np.ndarray, label_text: str, pad=(10,10,10,10)) -> np.ndarray:
        """
        Crop to the right of a label cell to capture names while ignoring signatures.
        Uses heuristic to find label row and crop the right portion.
        """
        h, w = metrics_bgr.shape[:2]
        
        # Assume labels are in left 40% of the image
        left_band = metrics_bgr[:, :int(0.4 * w)]
        
        # Project dark pixels to find row where label text likely is
        gray = cv2.cvtColor(left_band, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        proj = (255 - bw).sum(axis=1)
        
        # Find the row with most text content (likely the label row)
        row = int(np.argmax(proj))
        y1 = max(0, row - 30)
        y2 = min(h, row + 30)
        
        # Take right side for the value cell (name field)
        x1 = int(0.42 * w)
        x2 = w
        
        # Apply padding
        y1 = max(0, y1 - pad[1])
        y2 = min(h, y2 + pad[3])
        
        return metrics_bgr[y1:y2, x1:x2]

    def extract_name_from_metrics(self, metrics_bgr: np.ndarray, label_text: str) -> Optional[str]:
        """
        Extract a name from the metrics region, ignoring signature flourishes.
        Crops to the right of the label and uses a text-only prompt.
        """
        try:
            roi = self.crop_right_of_label(metrics_bgr, label_text)
            NAME_SYS = "Return JSON only: {\"text\": string}. Read the writer's name; ignore any signature flourish."
            result = self._ask_json(NAME_SYS, roi)
            return result.get("text", "").strip() or None
        except Exception as e:
            self.logger.warning(f"Name extraction failed for {label_text}: {e}")
            return None
        
    def extract_drilling_data(self, base64_image: str) -> Dict[str, Any]:
        """
        Extract drilling data from image using GPT with 3-prompt approach
        
        Args:
            base64_image: Base64 encoded image
            
        Returns:
            Dictionary containing extracted drilling data
        """
        try:
            # Convert base64 to PIL Image
            import base64
            from PIL import Image
            from io import BytesIO
            
            # Remove data URL prefix if present
            if base64_image.startswith('data:image'):
                base64_image = base64_image.split(',')[1]
            
            # Decode base64 to PIL Image
            image_data = base64.b64decode(base64_image)
            pil_image = Image.open(BytesIO(image_data))
            
            # Use the new 3-region extraction approach
            extracted_data = self.extract_three_regions(pil_image)
            
            # Apply re-ask loop if needed
            extracted_data = self.reask_activity_rows_if_needed(pil_image, extracted_data)
            
            # Normalize the data using existing normalization logic
            normalized_data = self._normalize_extracted_data(extracted_data)
            
            return {
                "success": True,
                "data": normalized_data,
                "confidence": self._calculate_confidence(normalized_data),
                "raw_response": json.dumps(extracted_data),
            }
            
        except Exception as e:
            self.logger.error(f"GPT OCR extraction failed: {e}")
            return {"success": False, "error": str(e), "data": {}, "confidence": 0.0, "flag": "dual_read:disagreement"}

    def extract_three_regions(self, pil_image) -> Dict[str, Any]:
        """Crop → 3 prompts → validate → targeted re-ask if needed → merged JSON."""
        # PIL → BGR
        img = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

        # 1) Ensure landscape orientation first
        img = ensure_landscape(img)
        
        # 2) Crop regions
        head = crop_header(img)
        metr = crop_metrics(img)
        
        # 3) Fail-safe: if metrics top is above mid-height, rotate and retry once
        if metr.bbox[1] < img.shape[0] * 0.45:
            self.logger.info("Metrics region detected too high, rotating image and retrying crops")
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            head = crop_header(img)
            metr = crop_metrics(img)
            
        act = crop_activity(img, head, metr)

        # 2) Call model with three tight prompts  
        header_json = self._ask_json(HEADER_SYS, head.image)
        
        # Isolate activity text block from checklist before GPT extraction
        act_img = isolate_activity_text_block(act.image)
        activity_json = self._ask_json(ACTIVITY_SYS, act_img)
        
        metrics_json = self._ask_json(METRICS_SYS, metr.image)

        result = {**header_json, **metrics_json}
        
        # Build activity rows with durations from new structure
        rows = activity_json.get("rows", [])
        act_rows, durations = [], []
        for r in rows:
            f_raw = (r.get("from") or "").strip()
            t_raw = (r.get("to") or "").strip()
            txt = (r.get("text") or "").strip()
            # Enhanced time parsing to handle various formats: 00 00h, 23.50, 05-00, 2230
            def parse_time(s):
                if not s: return None
                s = (s or "").strip().upper()
                # Remove common suffixes and normalize separators
                s = s.replace("H", "").replace(" ", "").replace(".", ":").replace("-", ":").replace(";", ":")
                
                # Handle formats: 22, 22:30, 2230 (no separator), 00 00h style
                m = re.match(r"^(\d{1,2})(?::?(\d{2}))?$", s)
                if not m: 
                    return None
                    
                hh = int(m.group(1))
                mm = int(m.group(2) or "00")
                
                # Validate ranges
                if hh > 24 or mm > 59: 
                    return None
                    
                return (hh % 24) * 60 + mm
            sf = parse_time(f_raw); st = parse_time(t_raw)
            dur_hr = None
            if sf is not None and st is not None:
                # midnight rollover
                if st < sf: st += 24*60
                dur_hr = round((st - sf)/60.0, 2)
                if dur_hr <= 0 or dur_hr > 8: dur_hr = None
            act_rows.append({"from": f_raw, "to": t_raw, "text": txt, "duration": dur_hr})
            durations.append(dur_hr if dur_hr is not None else 0.0)

        result["activity_rows"] = act_rows
        result["activity_durations"] = [x for x in durations if x is not None]
        result["activity_hours_total"] = round(sum([x for x in durations if x is not None]), 2)
        
        # Enhanced canonicalization using context fallback
        self._canonicalize_bhid_using_context(result)
        self._canonicalize_cubby_using_context(result)

        # 3) Validate + reconcile meters + re-ask if needed
        flags = []
        self._reconcile_bhid_cubby(result)
        meter_flags = self._reconcile_meters(result)
        flags.extend(meter_flags)

        dur_flags = self._validate_durations(result, act, head, metr)
        flags.extend(dur_flags)

        # attach bbox for ROI zoom
        result["_roi"] = {
            "header_bbox": head.bbox,
            "activity_bbox": act.bbox,
            "activity_rows": act.row_bboxes or [],
            "metrics_bbox": metr.bbox
        }
        result["flags"] = flags
        return result
    
    def _is_legacy_model(self, model_name: str) -> bool:
        """
        Determine if a model uses legacy max_tokens parameter vs max_completion_tokens
        
        Legacy models (use max_tokens):
        - GPT-3.5 series: gpt-3.5-turbo, gpt-3.5-turbo-16k, etc.
        - GPT-4 series: gpt-4, gpt-4-turbo, gpt-4o, etc.
        
        New models (use max_completion_tokens):
        - GPT-5 series: gpt-5, gpt-5-turbo, etc.
        - O1 series: o1-preview, o1-mini, etc.
        """
        model_lower = model_name.lower()
        
        # Check for new models that require max_completion_tokens
        new_model_prefixes = [
            'gpt-5', 'gpt5',
            'o1-', 'o1_',
            'chatgpt-4o-latest'  # Some newer variants
        ]
        
        for prefix in new_model_prefixes:
            if model_lower.startswith(prefix):
                return False
                
        # Default to legacy for GPT-3.5, GPT-4, and other older models
        return True

    def _supports_temperature(self, model_name: str) -> bool:
        """
        Determine if a model supports custom temperature parameter
        
        Models that don't support custom temperature (only default temp=1):
        - GPT-5 series: gpt-5, gpt-5-turbo, etc.
        - O1 series: o1-preview, o1-mini, etc. (reasoning models)
        
        Models that support custom temperature:
        - GPT-3.5 series: gpt-3.5-turbo, gpt-3.5-turbo-16k, etc.
        - GPT-4 series: gpt-4, gpt-4-turbo, gpt-4o, etc.
        """
        model_lower = model_name.lower()
        
        # Models that don't support custom temperature
        no_temp_prefixes = [
            'gpt-5', 'gpt5',     # GPT-5 series
            'o1-', 'o1_',        # O1 reasoning models
        ]
        
        for prefix in no_temp_prefixes:
            if model_lower.startswith(prefix):
                return False
        
        # Most other models support temperature
        return True

    # ---- model call helper (image + system prompt → JSON dict) ----
    def _ask_json(self, system_prompt: str, crop_bgr: np.ndarray) -> Dict[str, Any]:
        """Call GPT model with system prompt and image crop, return parsed JSON"""
        messages = [
            _sys(system_prompt),
            _usr([_img(crop_bgr), {"type":"text","text":"Return JSON only."}])
        ]
        
        # Build request parameters with existing logic for model compatibility
        request_params = {
            "model": self.config.gpt_model,
            "messages": messages,
        }
        
        # Add temperature only for models that support it
        if self._supports_temperature(self.config.gpt_model):
            request_params["temperature"] = 0.0  # Use low temperature for extraction
        
        # Use the correct token parameter based on the model
        if self._is_legacy_model(self.config.gpt_model):
            request_params["max_tokens"] = 800
        else:
            request_params["max_completion_tokens"] = 800
            
        resp = self.client.chat.completions.create(**request_params)
        txt = resp.choices[0].message.content.strip()
        
        # be forgiving: find JSON braces if model adds notes
        start, end = txt.find("{"), txt.rfind("}")
        payload = txt[start:end+1] if (start!=-1 and end!=-1) else "{}"
        try:
            return json.loads(payload)
        except Exception as e:
            self.logger.warning(f"Failed to parse JSON from model response: {e}")
            return {}
            
    # ---- reconciliation & validation ----
    def _canon(self, s: str) -> str:
        """Canonicalize string for comparison"""
        return (s or "").upper().strip().replace(" ", "").replace("_","_").replace("-", "")

    def _reconcile_bhid_cubby(self, d: Dict[str, Any]):
        """Canonicalise BHID and Cubby variations"""
        def norm_id(x): 
            return None if x is None else str(x).upper().replace(" ", "").replace("-", "").strip()
            
        if d.get("hole_id") and not d.get("bhid"):
            d["bhid"] = d["hole_id"]
        if d.get("bhid"):
            d["bhid"] = norm_id(d["bhid"])
            d["hole_id"] = d["bhid"]
        if d.get("cubby_id"):
            d["cubby_id"] = norm_id(d["cubby_id"])

        # Rig normalisation via mappings
        if d.get("rig_id"):
            x = self._canon(d["rig_id"])
            for k, vs in RIG_MAPPINGS.items():
                targets = {self._canon(k)} | {self._canon(v) for v in vs}
                if any(t in x or x in t for t in targets):
                    d["rig_id"] = k
                    break

    def _float_or_none(self, v):
        """Convert value to float or None"""
        if v is None: return None
        try: 
            return float(str(v).replace(",","."))  # accept comma
        except: 
            return None

    def _reconcile_meters(self, d: Dict[str, Any]) -> List[str]:
        """Ensure FROM/TO match Meter Start/End (within tolerance) and compute totals."""
        flags = []
        f = self._float_or_none(d.get("from_depth"))
        t = self._float_or_none(d.get("to_depth"))
        ms = self._float_or_none(d.get("meter_start"))
        me = self._float_or_none(d.get("meter_end"))
        tot = self._float_or_none(d.get("total_depth"))
        cum = self._float_or_none(d.get("cum_meters"))

        tol = 0.1

        # Prefer explicit FROM/TO if present; else use meter start/end
        if f is None and ms is not None: d["from_depth"] = f = ms
        if t is None and me is not None: d["to_depth"]   = t = me
        if ms is None and f is not None: d["meter_start"] = ms = f
        if me is None and t is not None: d["meter_end"]   = me = t

        if (f is not None and ms is not None and abs(f - ms) > tol) or \
           (t is not None and me is not None and abs(t - me) > tol):
            flags.append("meters:start_end_mismatch")

        # Compute totals consistently
        if f is not None and t is not None:
            computed = round(t - f, 2)
            if tot is None or abs(computed - tot) > tol:
                d["total_depth"] = computed
                if tot is not None:
                    flags.append("meters:total_corrected_from_from_to")
        elif tot is not None and f is not None and t is None:
            d["to_depth"] = round(f + tot, 2)
        elif tot is not None and t is not None and f is None:
            d["from_depth"] = round(t - tot, 2)

        if cum is None and d.get("total_depth") is not None:
            d["cum_meters"] = d["total_depth"]

        # sanity limits
        if d.get("total_depth") is not None and d["total_depth"] > 15:
            flags.append("meters:total_depth_unusually_high")

        return flags

    def _validate_durations(self, d: Dict[str, Any], act: CropResult, head: CropResult, metr: CropResult) -> List[str]:
        """Validate and clean activity durations"""
        flags = []
        arr = d.get("activity_durations") or []
        # numeric and sensible
        clean = []
        for v in arr:
            try:
                x = float(v)
            except:
                x = None
            if x is None or x <= 0 or x > 8:
                flags.append("duration:row_out_of_range")
            else:
                clean.append(round(x,2))
        d["activity_durations"] = clean
        total = round(sum(clean),2)
        d["activity_hours_total"] = total
        if total < 6 or total > 12:
            flags.append("duration:total_out_of_range")
        return flags

    def reask_activity_rows_if_needed(self, pil_image, d: Dict[str, Any]) -> Dict[str, Any]:
        """Re-ask activity rows if there are duration validation issues"""
        if not d.get("flags"):
            return d
        if not any(f.startswith("duration:") for f in d["flags"]):
            return d
        # re-crop activity & re-ask rows
        img = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        head = crop_header(img) 
        metr = crop_metrics(img)
        act = crop_activity(img, head, metr)
        rows = act.row_bboxes or []
        if not rows: 
            return d
        fixed = []
        for (x,y,w,h) in rows:
            row_img = act.image[y:y+h, x:x+w]
            rjson = self._ask_json(ACTIVITY_SYS, row_img)
            dur = rjson.get("durations", [])
            if dur and isinstance(dur, list):
                try:
                    fixed.append(float(dur[0]))
                except:
                    pass
        if fixed:
            d["activity_durations"] = [round(x,2) for x in fixed if isinstance(x,(int,float))]
            d["activity_hours_total"] = round(sum(d["activity_durations"]),2)
            # re-evaluate flags
            d["flags"] = [f for f in d.get("flags",[]) if not f.startswith("duration:")]
            d["flags"].extend(self._validate_durations(d, act, head, metr))
        return d
        
    def _create_drilling_prompt(self) -> str:
        """
        Build a detailed prompt instructing GPT‑5 to extract drilling form data.

        This prompt is tuned specifically for the Pickstone and Peerless Diamond Drilling Daily Report. It tells
        GPT exactly what to look for in the scanned form, how to normalise values, and how to compute
        per‑row activity durations. The resulting JSON must adhere strictly to the provided schema so that it
        can be consumed by downstream validation and Excel integration.

        Returns:
            A string prompt ready to send to the GPT model.
        """

        # Serialize rig mappings so GPT can normalise fuzzy rig names to canonical sheet names
        rig_map_json = json.dumps(RIG_MAPPINGS, indent=2)

        # Construct the prompt. A regular multi‑line string is used (no f‑string braces) so that
        # curly braces in the JSON schema aren’t interpreted prematurely.
        prompt = (
            "You are an intelligent extraction engine for Pickstone and Peerless Diamond Drilling Daily Reports. "
            "You will be given a single image of a paper form (containing both typed labels and handwritten entries). "
            "Your job is to read the form, extract the key fields listed below, compute per‑row durations for the Activity "
            "Record table, and return the results as **strict JSON** matching the given schema.\n\n"

            "## Fields to extract\n"
            "- date: Drilling date normalised to YYYY‑MM‑DD. Accept input formats like DD/MM/YY, DD/MM/YYYY or YYYY‑MM‑DD.\n"
            "- shift: Either \"Day\" or \"Night\" (capitalised). If the form doesn’t clearly indicate, return null.\n"
            "- level: The shaft & level, e.g. 6L, 6W19 or 6_L_21. Keep alphanumeric characters and underscores.\n"
            "- rig_id: The rig identifier. Use the canonical name according to the mapping below.\n"
            "- cubby_id: The cubby ID such as 6W19, 6AW9 or 6_W21. Preserve underscores if present.\n"
            "- bhid: The hole identifier (BHID or Hole No.). Accept formats like 7W16_EV01, 6W21_02 or 6AW9_EV03.\n"
            "  If multiple BHIDs appear (e.g. one in the header and one in the bottom table), choose the BHID associated\n"
            "  with the drilled metres section.\n"
            "- from_depth: Starting depth in metres. Use a numeric value; convert commas to dots.\n"
            "- to_depth: Ending depth in metres. Use a numeric value.\n"
            "- total_depth: Total metres drilled. If not explicitly provided but both depths are present, compute it as\n"
            "  to_depth minus from_depth.\n"
            "- core_recovery: Core recovery percentage (0–100).\n"
            "- rqd: Rock Quality Designation percentage (0–100).\n"
            "- geology: Geological description or rock type.\n"
            "- comments: Any additional comments.\n"
            "- driller: Name of the driller.\n"
            "- geotech: Name of the geotechnician or geologist.\n"
            "- activity_durations: A list of decimal hours, one for each populated row in the Activity Record table.\n"
            "  The Activity Record table has columns labelled HOLE NO., TIME (with sub‑columns FROM and TO) and ACTIVITY RECORD.\n"
            "  For each row, do the following:\n"
            "    • Read the \"FROM\" and \"TO\" times. Times may be written with a colon (07:30), a dot (7.50), a dash (18-00) or even as a single decimal (0.5).\n"
            "    • If both FROM and TO exist, compute the duration as (TO − FROM) in hours. Express 30 minutes as 0.5, 15 minutes as 0.25, etc.\n"
            "    • If only one time is written and it looks like a decimal hour (e.g. 1.00 or 0.5), treat that as the duration directly.\n"
            "    • Convert times like 18-00 or 18.00 to 18:00 before computing.\n"
            "    • Ignore the activity description text itself; we only need the numeric duration.\n"
            "    • Skip empty or illegible rows. Do not add zero durations.\n"
            "  The resulting list should contain only numbers.\n"
            "- flags: A list of warnings. For example, add 'meters_high' if total_depth > 10.0, 'missing_bhid' if no BHID is found, or\n"
            "  'activity_count_mismatch' if there are clearly rows in the Activity Record but fewer durations extracted.\n\n"

            "## Rig name variations (canonical → possible handwritten forms)\n"
            f"{rig_map_json}\n\n"

            "## Normalisation and validation rules\n"
            "1. Use the rig mapping above to convert fuzzy rig names (e.g. KEMPE U2 AW) to the canonical key (e.g. KEMPE_U2AW).\n"
            "2. Remove spaces, hyphens and underscores when comparing rig names.\n"
            "3. Dates must be in the ISO 8601 format (YYYY‑MM‑DD).\n"
            "4. Shift must be exactly 'Day' or 'Night'. If not present, return null.\n"
            "5. Depth and percentage values must be numeric; remove any units like 'm' or '%' and convert commas to dots.\n"
            "6. BHID should be treated the same as Hole ID. Accept letters, digits, underscores and hyphens.\n"
            "7. Only return the fields specified in the schema; do not invent new keys or explanations.\n\n"

            "## Output JSON schema\n"
            "{\n"
            "  \"date\": \"YYYY‑MM‑DD\",\n"
            "  \"shift\": \"Day|Night|null\",\n"
            "  \"level\": \"string|null\",\n"
            "  \"rig_id\": \"canonical rig id\",\n"
            "  \"cubby_id\": \"string|null\",\n"
            "  \"bhid\": \"string|null\",\n"
            "  \"from_depth\": 0.0|null,\n"
            "  \"to_depth\": 0.0|null,\n"
            "  \"total_depth\": 0.0|null,\n"
            "  \"core_recovery\": 0.0|null,\n"
            "  \"rqd\": 0.0|null,\n"
            "  \"geology\": \"string|null\",\n"
            "  \"comments\": \"string|null\",\n"
            "  \"driller\": \"string|null\",\n"
            "  \"geotech\": \"string|null\",\n"
            "  \"activity_durations\": [0.0, ...],\n"
            "  \"flags\": [\"string\", ...]\n"
            "}\n\n"

            "Respond only with JSON. Do not include any explanatory text before or after the JSON."
        )
        return prompt

    def _parse_gpt_response(self, response: str) -> Dict[str, Any]:
        """Parse GPT response and extract JSON data"""
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Fallback: try to parse the entire response
                return json.loads(response)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from GPT response: {e}")
            # Return empty structure as fallback
            return {
                "date": None,
                "rig_id": None,
                "hole_id": None,
                "from_depth": None,
                "to_depth": None,
                "total_depth": None,
                "core_recovery": None,
                "rqd": None,
                "geology": None,
                "comments": None,
                "confidence": {},
                "overall_confidence": 0.0
            }
    
    def _normalize_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and clean extracted data"""
        normalized: Dict[str, Any] = {}

        # 1. Map any keys in the raw data to our standard fields using FIELD_MAPPINGS
        for standard_field, variations in FIELD_MAPPINGS.items():
            value = None
            # Try alias variations
            for variant in variations:
                if variant in data and data[variant] is not None:
                    value = data[variant]
                    break
            # Fallback to standard key
            if value is None and standard_field in data and data[standard_field] is not None:
                value = data[standard_field]
            normalized[standard_field] = value

        # 2. Normalise rig_id via configured mapping
        if normalized.get('rig_id'):
            normalized['rig_id'] = self._normalize_rig_id(normalized['rig_id'])

        # 3. Unify BHID/Hole ID: if bhid missing but hole_id present, or vice versa
        # Prefer bhid field name in output
        bhid_val = normalized.get('bhid') or normalized.get('hole_id')
        normalized['bhid'] = bhid_val
        # Also set hole_id for backward compatibility
        normalized['hole_id'] = bhid_val

        # 4. Convert numeric fields to floats where appropriate
        numeric_fields = ['from_depth', 'to_depth', 'total_depth', 'core_recovery', 'rqd', 'daily_budget']
        for field in numeric_fields:
            val = normalized.get(field)
            if val is None or val == "" or val == "-":
                normalized[field] = None
                continue
            try:
                # Replace comma with dot and remove units or extraneous chars
                val_str = str(val).strip().replace(',', '.')
                # Remove non-numeric trailing characters (e.g., 'm')
                # Keep digits, dot and minus
                import re
                match = re.search(r'-?\d+(?:\.\d+)?', val_str)
                if match:
                    normalized[field] = float(match.group(0))
                else:
                    normalized[field] = None
            except Exception:
                normalized[field] = None

        # 5. Convert activity_durations to list of floats
        durations = normalized.get('activity_durations')
        parsed_durations: List[float] = []
        if durations is not None:
            try:
                # If durations is a string representing a list, attempt to parse JSON
                if isinstance(durations, str):
                    durations_parsed = json.loads(durations)
                else:
                    durations_parsed = durations
                for d in durations_parsed:
                    try:
                        parsed_durations.append(float(d))
                    except Exception:
                        # Skip invalid entries
                        continue
            except Exception:
                parsed_durations = []
        normalized['activity_durations'] = parsed_durations if parsed_durations else None

        # 6. Compute total_depth if missing but from_depth and to_depth present
        if (normalized.get('total_depth') is None or normalized.get('total_depth') == 0) and \
           normalized.get('from_depth') is not None and normalized.get('to_depth') is not None:
            try:
                diff = float(normalized['to_depth']) - float(normalized['from_depth'])
                if diff >= 0:
                    normalized['total_depth'] = diff
            except Exception:
                pass

        # 7. Ensure shift normalisation (capitalize and restrict to Day/Night)
        shift_val = normalized.get('shift')
        if shift_val:
            shift_str = str(shift_val).strip().lower()
            if shift_str.startswith('d'):
                normalized['shift'] = 'Day'
            elif shift_str.startswith('n'):
                normalized['shift'] = 'Night'
            else:
                normalized['shift'] = None

        # 8. Clean flags: ensure it's a list
        flags = data.get('flags')
        if not flags or not isinstance(flags, list):
            normalized['flags'] = []
        else:
            normalized['flags'] = flags

        # 9. Attach confidence and overall_confidence if provided
        normalized['confidence'] = data.get('confidence', {})
        normalized['overall_confidence'] = data.get('overall_confidence', 0.0)

        return normalized
    
    def _normalize_rig_id(self, rig_id: str) -> str:
        """Normalize rig ID to standard format"""
        if not rig_id:
            return None
            
        rig_id_clean = str(rig_id).strip().upper()
        
        # Check against known mappings
        for standard_rig, variations in RIG_MAPPINGS.items():
            if rig_id_clean in [v.upper() for v in variations]:
                return standard_rig
        
        return rig_id_clean
    
    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        confidences = data.get('confidence', {})
        if not confidences:
            return 0.5  # Default confidence
        
        # Calculate average confidence, weighted by field importance
        field_weights = {
            'date': 1.0,
            'rig_id': 1.0,
            'hole_id': 1.0,
            'from_depth': 0.8,
            'to_depth': 0.8,
            'total_depth': 0.6,
            'core_recovery': 0.7,
            'rqd': 0.7,
            'geology': 0.5,
            'comments': 0.3
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for field, weight in field_weights.items():
            if field in confidences:
                weighted_sum += confidences[field] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
