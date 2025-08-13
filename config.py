
"""
Configuration module for GPT-5 OCR System
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------------- OCR / Model config ----------------------

@dataclass
class OCRConfig:
    """Configuration for OCR processing"""
    # API Configuration
    openai_api_key: str = ""
    gpt_model: str = "gpt-5"  
    max_completion_tokens: int = 4000  # GPT-5 uses max_completion_tokens instead of max_tokens
    temperature: float = 0.1

    # Image Processing
    max_image_size: Tuple[int, int] = (2048, 2048)
    image_quality: int = 95

    # Validation
    confidence_threshold: float = 0.8

    def __post_init__(self):
        # Prefer Streamlit secrets, then env vars
        try:
            import streamlit as st
            self.openai_api_key = st.secrets.get("OPENAI_API_KEY", "") or self.openai_api_key
            self.gpt_model      = st.secrets.get("OPENAI_MODEL", self.gpt_model)
        except Exception:
            pass

        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        forced_model = os.getenv("OPENAI_MODEL")
        if forced_model:
            self.gpt_model = forced_model

# ---------------------- Field mappings ----------------------
# IMPORTANT:
# - These are *semantic* aliases coming back from GPT or your parsers.
# - Keep them broad but unambiguous; validators will enforce correctness later.

FIELD_MAPPINGS: Dict[str, List[str]] = {
    # Core metadata
    "date": ["date", "drilling_date", "Date", "DATE"],
    "shift": ["shift", "SHIFT"],
    "rig_id": ["rig", "rig_id", "rig_number", "Rig", "RIG"],
    "level": ["level", "LEVEL", "shaft_level", "SHAFT & LEVEL", "SHAFT LEVEL", "SHAFT&LEVEL"],
    "cubby_id": ["cubby", "cubby_id", "CUBBY", "CUBBY ID", "Cubby ID", "CUBBYID"],
    "hole_id": ["hole", "hole_id", "hole_number", "Hole", "HOLE", "HOLE NO.", "HOLE NO", "BHID", "bhid"],

    # Depth information
    "from_depth": ["from", "from_depth", "From", "FROM", "FROM(m)", "DEPTH FROM", "DEPTH_FROM"],
    "to_depth": ["to", "to_depth", "To", "TO", "TO(m)", "DEPTH TO", "DEPTH_TO"],
    "total_depth": ["total", "total_depth", "Total", "TOTAL", "TOTAL(m)", "drilling_meters", "drilled_meters"],
    "daily_budget": ["daily_budget", "DAILY BUDGET", "budget_per_shift", "Budget"],

    # Quality metrics
    "core_recovery": ["recovery", "core_recovery", "Recovery", "RECOVERY", "CORE RECOVERY", "RECOVERY(%)"],
    "rqd": ["rqd", "RQD", "rock_quality", "Rock Quality", "ROCK QUALITY", "RQD(%)"],
    "geology": ["geology", "rock_type", "Geology", "GEOLOGY", "ROCK TYPE", "ROCK"],

    # Comments and notes
    "comments": ["comments", "notes", "remarks", "Comments", "COMMENTS", "COMMENT", "NOTES"],

    # Personnel
    "driller": ["driller", "DRILLER"],
    "geotech": ["geotech", "geotechnician", "GEO TECH", "GEO-TECH", "GEOTECH", "GEOLOGIST"],

    # Tramming (for the compliance sheet flow)
    "level_tramming": ["Level", "LEVEL"],
    "box_id": ["BOX ID", "BOX", "Box", "Box ID"],
    "planned_cocopans": ["Planned Cocopans/Shift", "Planned", "PLAN", "PLAN CARS", "BUDGET (CARS)"],
    "planned_grade": ["Planned Grade (g/t)", "Planned Grade", "PLAN G/T", "BUDGET (g/t)"],
    "actual_cocopans_day": ["Actual Cocopans Day", "Day Shift", "Actual Day", "Actual (Day)"],
    "actual_cocopans_night": ["Actual Cocopans Night", "Night Shift", "Actual Night", "Actual (Night)"],
    "actual_cocopans": ["Actual Cocopans", "Actual", "ACTUAL (CARS)"],

    # Activity durations (new): list of decimal hours for each activity record row
    "activity_durations": ["activity_durations", "activity_duration", "durations", "activity_times", "activity_durations_hours"]
}

# ---------------------- Character correction mappings ----------------------
# Characters that GPT/handwriting often confuses with digits
TYPO_TO_DIGIT = {
    "G": "6",  # G ↔ 6
    "S": "5",  # S ↔ 5
    "O": "0",  # O ↔ 0
    "I": "1",  # I ↔ 1
    "B": "8",  # B ↔ 8
    "Z": "2"   # Z ↔ 2
}

# ---------------------- Rig mappings ----------------------
# Keys are **canonical sheet/section names** you want downstream.
# Variants are what might appear handwritten/on the sheet/in GPT output.
#
# NOTE: The code canonicalises by stripping spaces/underscores/hyphens
# and using case-insensitive contains(), so this list can be concise.
# Still, we include common "weird" splits (e.g., "U2 AW") to improve discovery.

RIG_MAPPINGS: Dict[str, List[str]] = {
    # Kempe rigs
    "KEMPE_U2AW": [
        "KEMPE_U2AW", "KEMPE U2AW", "KEMPE-U2AW", "U2AW",
        "KEMPE U2 AW", "U2 AW", "KEMPEU2AW", "U-2AW", "KEMPE  U2AW"
    ],
    "KEMPE_U39BQ": [
        "KEMPE_U39BQ", "KEMPE U39BQ", "KEMPE-U39BQ", "U39BQ",
        "KEMPE U39 BQ", "U39 BQ", "KEMPEU39BQ"
    ],

    # JBC rigs
    "JBC 01": ["JBC 01", "JBC01", "JBC-01", "JBC_01", "JBC 1", "JBC1"],
    "JBC 02": ["JBC 02", "JBC02", "JBC-02", "JBC_02", "JBC 2", "JBC2"],
    "JBC 03": ["JBC 03", "JBC03", "JBC-03", "JBC_03", "JBC 3", "JBC3"],

    # Tracked rigs
    "T06": ["T06", "T-06", "T 06", "T_06", "T6"],
    "T07": ["T07", "T-07", "T 07", "T_07", "T7"],
    "T13": ["T13", "T-13", "T 13", "T_13"],  # was missing before

    # Other rigs
    "BDU400": ["BDU400", "BDU 400", "BDU-400", "BDU", "BDU_400"],  # helpful for old sheets
}
