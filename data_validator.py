
"""
Data Validator for Diamond Drilling OCR System
"""

from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import re
import logging
import difflib
from config import RIG_MAPPINGS

class DrillingDataValidator:
    """Validates extracted drilling data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Validation rules
        self.valid_rigs = set(RIG_MAPPINGS.keys())
        
        # helpers for rig normalization
        self._canon = lambda x: (str(x) if x is not None else "").upper().strip().replace(" ", "").replace("_", "").replace("-", "")

        self.max_depth = 1000.0  # Maximum reasonable depth in meters
        self.max_recovery = 100.0  # Maximum recovery percentage
        self.max_rqd = 100.0  # Maximum RQD percentage
        
        # Optional personnel roster for name matching
        self.personnel_roster = None  # Can be set externally
        
    def validate_drilling_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate extracted drilling data
        
        Returns:
            (is_valid, errors, cleaned_data)
        """
        errors = []
        cleaned_data = data.copy()
        
        # Validate date
        if not self._validate_date(cleaned_data.get('date')):
            errors.append("Invalid or missing date")
            
        # Validate rig ID
        if not self._validate_rig_id(cleaned_data.get('rig_id')):
            errors.append("Invalid or missing rig ID")
            
        # Validate hole ID
        if not self._validate_hole_id(cleaned_data.get('hole_id')):
            errors.append("Invalid or missing hole ID")
            
        # Validate depths
        depth_errors = self._validate_depths(cleaned_data)
        errors.extend(depth_errors)
        
        # Validate percentages
        if not self._validate_percentage(cleaned_data.get('core_recovery'), 'core_recovery'):
            errors.append("Invalid core recovery percentage")
            
        if not self._validate_percentage(cleaned_data.get('rqd'), 'rqd'):
            errors.append("Invalid RQD percentage")
            
        # Apply name reconciliation and meters check
        self.reconcile_names(cleaned_data)
        self.final_meters_check(cleaned_data)
        
        # Apply activity alignment before duration validation
        self.align_activity(cleaned_data)
        
        # Apply operational rules specific to diamond drilling operations
        self.apply_operational_rules(cleaned_data)
        
        # Add any new flags as errors if they indicate serious issues
        flags = cleaned_data.get('flags', [])
        for flag in flags:
            if 'meters:start_vs_meter_start' in flag or 'meters:to_vs_meter_end' in flag:
                errors.append(f"Meter validation issue: {flag}")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            self.logger.warning(f"Validation errors: {errors}")
        
        return is_valid, errors, cleaned_data
    
    def _validate_date(self, date_value: Any) -> bool:
        """Validate date field"""
        if not date_value:
            return False
            
        try:
            # Try to parse common date formats
            date_str = str(date_value).strip()
            
            # Try YYYY-MM-DD format
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                return True
            except ValueError:
                pass
            
            # Try DD/MM/YYYY format
            try:
                datetime.strptime(date_str, '%d/%m/%Y')
                return True
            except ValueError:
                pass
                
            # Try MM/DD/YYYY format
            try:
                datetime.strptime(date_str, '%m/%d/%Y')
                return True
            except ValueError:
                pass
                
            return False
            
        except Exception:
            return False
    
    def _validate_rig_id(self, rig_id: Any) -> bool:
        """Validate rig ID"""
        if not rig_id:
            return False
            
        canon = self._normalize_rig(str(rig_id))
        return canon in self.valid_rigs
    
    def _normalize_rig(self, rig_id: str) -> str:
        x = self._canon(rig_id)
        for standard, variants in RIG_MAPPINGS.items():
            targets = {self._canon(standard)} | {self._canon(v) for v in variants}
            if x in targets or any(t in x or x in t for t in targets):
                return standard
        return str(rig_id).strip()

    def _validate_hole_id(self, hole_id: Any) -> bool:
        """Validate hole ID"""
        if not hole_id:
            return False
            
        hole_str = str(hole_id).strip()
        # Basic validation - should not be empty and reasonable length
        return len(hole_str) > 0 and len(hole_str) <= 20
    
    def _validate_depths(self, data: Dict[str, Any]) -> List[str]:
        """Validate depth fields"""
        errors = []
        
        from_depth = self._parse_numeric(data.get('from_depth'))
        to_depth = self._parse_numeric(data.get('to_depth'))
        total_depth = self._parse_numeric(data.get('total_depth'))
        
        # Validate individual depths
        if from_depth is not None and (from_depth < 0 or from_depth > self.max_depth):
            errors.append(f"From depth out of range: {from_depth}")
            
        if to_depth is not None and (to_depth < 0 or to_depth > self.max_depth):
            errors.append(f"To depth out of range: {to_depth}")
            
        if total_depth is not None and (total_depth < 0 or total_depth > self.max_depth):
            errors.append(f"Total depth out of range: {total_depth}")
        
        # Validate depth relationships
        if from_depth is not None and to_depth is not None:
            if from_depth >= to_depth:
                errors.append("From depth must be less than to depth")
        
        return errors
    
    def _validate_percentage(self, value: Any, field_name: str) -> bool:
        """Validate percentage fields"""
        if value is None:
            return True  # Optional field
            
        try:
            num_value = float(value)
            return 0 <= num_value <= 100
        except (ValueError, TypeError):
            return False
    
    def _parse_numeric(self, value: Any) -> float:
        """Parse numeric value safely"""
        if value is None:
            return None
            
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
            
    def _to_float(self, value: Any) -> float:
        """Convert value to float, handling commas and various formats"""
        if value is None:
            return None
        try:
            # Handle comma as decimal separator and remove units
            val_str = str(value).strip().replace(',', '.')
            # Remove non-numeric trailing characters (e.g., 'm')
            import re
            match = re.search(r'-?\d+(?:\.\d+)?', val_str)
            if match:
                return float(match.group(0))
            return None
        except Exception:
            return None

    def _safe_float(self, v) -> Optional[float]:
        """Safely convert value to float, handling various formats"""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        
        try:
            s = str(v).strip()
            if not s:
                return None
            # Handle common formats and remove units
            s = s.replace(',', '.').replace('m', '').replace('M', '').strip()
            return float(s)
        except (ValueError, TypeError):
            return None
    
    def _clean_name(self, s: str) -> str:
        """Clean and normalize personnel names"""
        if not s: 
            return ""
        s = re.sub(r"[^A-Za-z\s\.-]", "", str(s))
        return re.sub(r"\s+", " ", s).strip().title()

    def reconcile_names(self, d: Dict[str, Any], roster: Optional[List[str]] = None):
        """Reconcile and clean personnel names with optional fuzzy matching"""
        for k in ("driller", "geotech", "shift_supervisor"):
            if d.get(k):
                raw = self._clean_name(d[k])
                d[k] = raw
                # fuzzy match against roster if provided
                if roster:
                    matches = difflib.get_close_matches(raw, roster, n=1, cutoff=0.7)
                    if matches: 
                        d[k] = matches[0]

    def final_meters_check(self, d: Dict[str, Any], tol=0.1) -> None:
        """Final validation check for meter consistency between FROM/TO and Meter Start/End"""
        f = self._to_float(d.get("from_depth"))
        t = self._to_float(d.get("to_depth"))
        ms = self._to_float(d.get("meter_start"))
        me = self._to_float(d.get("meter_end"))
        
        # Check for mismatches between FROM/TO depths and meter start/end
        if None not in (f, ms) and abs(f - ms) > tol: 
            d.setdefault("flags", []).append("meters:start_vs_meter_start")
        if None not in (t, me) and abs(t - me) > tol: 
            d.setdefault("flags", []).append("meters:to_vs_meter_end")

    def align_activity(self, d: dict):
        """Ensure activity text aligns with durations and validate activity rows"""
        rows = d.get("activity_rows") or []
        # Drop rows that have neither time nor text
        rows = [r for r in rows if any([(r.get("from") or r.get("to") or r.get("text"))])]
        # Ensure duration present → if missing but from/to look parseable, keep None (flag later)
        d["activity_rows"] = rows
        d["activity_durations"] = [r.get("duration") for r in rows]
        # basic flags
        for i, r in enumerate(rows, start=1):
            if r.get("duration") is None:
                d.setdefault("flags", []).append(f"duration:row_{i}_unparsed")

    def apply_operational_rules(self, d: dict):
        """Apply extra operational rules specific to diamond drilling operations"""
        # Rule 1: Night shift with "no compressed air/water" -> allow meters to be 0
        comments = (d.get("comments") or "").lower()
        no_air_water = any(phrase in comments for phrase in [
            "no compressed air", "no water", "no air", "compressed air off", 
            "water off", "no compressed", "air/water off"
        ])
        
        if no_air_water:
            d.setdefault("flags", []).append("context:no_air_or_water")
            # Allow zero meters without flagging as error for night shift
            if d.get("total_depth") == 0 or d.get("drilling_meters") == 0:
                d.setdefault("flags", []).append("meters:zero_allowed_no_air_water")
        
        # Rule 2: Total meters soft cap at 15m - flag if above for dual-read
        total_meters = self._safe_float(d.get("total_depth") or d.get("drilling_meters"))
        if total_meters and total_meters > 15.0:
            d.setdefault("flags", []).append("meters:above_15m_soft_cap")
            # This could trigger dual-read of Meter Start/End cells in UI
        
        # Rule 3: Enhanced BHID/Cubby context inference with Level 6L
        level = (d.get("level") or "").upper().strip()
        if level.startswith("6L"):
            # If BHID missing left digit, prepend "6" but still flag
            bhid = d.get("bhid") or d.get("hole_id")
            if bhid and bhid.startswith("W"):
                d["bhid"] = f"6{bhid}"
                d["hole_id"] = d["bhid"]
                d.setdefault("flags", []).append("bhid:6L_context_prepended")
            
            # If Cubby missing left digit, prepend "6" but still flag  
            cubby = d.get("cubby_id")
            if cubby and cubby.startswith("W"):
                d["cubby_id"] = f"6{cubby}"
                d.setdefault("flags", []).append("cubby:6L_context_prepended")
