
"""
Updated Streamlit App for GPT-5 Diamond Drilling OCR System
WITH MANUAL REVIEW WORKFLOW
"""

import streamlit as st
import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import sys
import numpy as np
import cv2
from PIL import Image

# Add parent directory to path for imports
sys.path.append('/home/ubuntu/diamond_drilling_ocr_enhanced')

from config import OCRConfig
from simple_preprocessor import SimpleImagePreprocessor
from gpt5_ocr_engine import GPT5OCREngine
from data_validator import DrillingDataValidator
from excel_forms_integration import FormsTabIntegration
from region_cropping import crop_header, crop_activity, crop_metrics

@st.cache_resource
def get_engine():
    cfg = OCRConfig()
    engine = GPT5OCREngine(cfg)
    return engine, cfg

# Configure page
st.set_page_config(
    page_title="GPT-5 Diamond Drilling OCR - Manual Review",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'config' not in st.session_state:
        st.session_state.config = OCRConfig()
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = []
    if 'approved_results' not in st.session_state:
        st.session_state.approved_results = []
    if 'api_key_configured' not in st.session_state:
        st.session_state.api_key_configured = False
    if 'excel_file_path' not in st.session_state:
        st.session_state.excel_file_path = None
    if 'current_crops' not in st.session_state:
        st.session_state.current_crops = None

def setup_sidebar():
    """Setup sidebar configuration"""
    st.sidebar.title("⚙️ Configuration")
    
    # ROI Zoom section (will be populated during processing)
    if 'current_crops' in st.session_state and st.session_state.current_crops:
        setup_roi_zoom_sidebar()
    
    st.sidebar.markdown("---")
    
    # Model / API Configuration (Streamlit Cloud)
    st.sidebar.subheader("🤖 Model / API")
    st.sidebar.caption("Using Streamlit Cloud secrets: `OPENAI_API_KEY`, `OPENAI_MODEL`.")
    # Optional read-only echo for clarity:
    try:
        model_in_use = st.secrets.get("OPENAI_MODEL", "gpt-4o")
    except Exception:
        model_in_use = "gpt-4o"
    st.sidebar.text(f"Model: {model_in_use}")

    # Processing Parameters
    temperature = st.sidebar.slider(
        "Temperature",
        0.0, 1.0, st.session_state.config.temperature,
        help="Lower values = more focused, higher = more creative"
    )
    st.session_state.config.temperature = temperature
    
    # Excel Configuration
    st.sidebar.subheader("📊 Excel Configuration")
    excel_file = st.sidebar.file_uploader(
        "Upload Excel Workbook",
        type=['xlsx', 'xlsm'],
        help="Upload your drilling report Excel file (required for final export)"
    )
    
    if excel_file:
        # Save Excel file for later use
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{excel_file.name}") as temp_file:
            temp_file.write(excel_file.getbuffer())
            st.session_state.excel_file_path = temp_file.name
        st.sidebar.success(f"✅ Excel file loaded: {excel_file.name}")
    
    return excel_file is not None

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # App Header
    st.title("💎 GPT-5 Diamond Drilling OCR System")
    st.markdown("**Intelligent OCR with Manual Review Workflow**")
    st.markdown("---")
    
    # Setup sidebar and check Excel
    excel_loaded = setup_sidebar()
    
    # Check API key
    if not st.session_state.config.openai_api_key:
        st.error("🚨 Please configure your OpenAI API key in the sidebar to proceed.")
        st.info("💡 Get your API key from: https://platform.openai.com/api-keys")
        return
    
    # Main workflow tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📸 1. Upload & Process", 
        "👁️ 2. Manual Review", 
        "📊 3. Export to Excel",
        "📋 4. Results History"
    ])
    
    with tab1:
        upload_and_process_tab()
    
    with tab2:
        manual_review_tab(excel_loaded)
    
    with tab3:
        export_to_excel_tab(excel_loaded)
    
    with tab4:
        results_history_tab()

def upload_and_process_tab():
    """Step 1: Upload and process images"""
    st.header("📸 Step 1: Upload & Process Images")
    st.markdown("Upload drilling form images for GPT-5 OCR processing")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Drilling Form Images",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True,
        help="Upload images of handwritten drilling forms"
    )
    
    if uploaded_files:
        st.success(f"✅ Uploaded {len(uploaded_files)} image(s)")
        
        # Process button
        if st.button("🚀 Process Images with GPT-5", type="primary"):
            process_images(uploaded_files)
    
    # Show recent processing results
    if st.session_state.processed_results:
        st.subheader("📋 Recently Processed")
        for i, result in enumerate(st.session_state.processed_results[-3:]):  # Show last 3
            with st.expander(f"📄 {result['filename']} - {'✅ Success' if result['success'] else '❌ Failed'}"):
                if result['success']:
                    display_extraction_summary(result)
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

def process_images(uploaded_files):
    """Process uploaded images with GPT-5"""
    if not uploaded_files:
        st.error("No images to process")
        return
    
    # Initialize components
    preprocessor = SimpleImagePreprocessor()
    ocr_engine, cfg = get_engine()
    validator = DrillingDataValidator()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    new_results = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name
            
            # Preprocess image
            with st.spinner("🔄 Preprocessing image..."):
                base64_image = preprocessor.preprocess_image(temp_path)
                
            # Convert base64 to PIL for region cropping
            import base64
            from io import BytesIO
            if base64_image.startswith('data:image'):
                base64_data = base64_image.split(',')[1]
            else:
                base64_data = base64_image
            image_data = base64.b64decode(base64_data)
            pil_image = Image.open(BytesIO(image_data))
            
            # Show 3 crops side-by-side
            st.subheader(f"📸 Region Crops - {uploaded_file.name}")
            bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
            head = crop_header(bgr)
            metr = crop_metrics(bgr) 
            act = crop_activity(bgr, head, metr)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(cv2.cvtColor(head.image, cv2.COLOR_BGR2RGB), caption="Header crop", use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(act.image, cv2.COLOR_BGR2RGB), caption="Activity crop", use_container_width=True)
            with col3:
                st.image(cv2.cvtColor(metr.image, cv2.COLOR_BGR2RGB), caption="Metrics crop", use_container_width=True)
            
            # Store crops in session state for ROI zoom
            st.session_state.current_crops = {
                'header': head,
                'activity': act,
                'metrics': metr,
                'filename': uploaded_file.name
            }
            
            # Extract data with GPT-5 using new 3-region approach
            with st.spinner("🧠 Extracting data with GPT-5 (3-region approach)..."):
                extracted_data = ocr_engine.extract_three_regions(pil_image)
                extracted_data = ocr_engine.reask_activity_rows_if_needed(pil_image, extracted_data)
                
                # Create OCR result structure for compatibility
                ocr_result = {
                    'success': True,
                    'data': extracted_data,
                    'confidence': 0.85,  # Default confidence, could be calculated from flags
                    'raw_response': json.dumps(extracted_data)
                }
            
            if ocr_result['success']:
                # Validate data
                is_valid, validation_errors, cleaned_data = validator.validate_drilling_data(ocr_result['data'])
                
                result = {
                    'filename': uploaded_file.name,
                    'success': True,
                    'data': cleaned_data,
                    'confidence': ocr_result['confidence'],
                    'validation': {
                        'is_valid': is_valid,
                        'errors': validation_errors
                    },
                    'raw_response': ocr_result.get('raw_response', ''),
                    'processed_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'pending_review',  # NEW: Track status
                    'crops_data': {
                        'header_bbox': head.bbox,
                        'activity_bbox': act.bbox,
                        'metrics_bbox': metr.bbox,
                        'activity_rows': act.row_bboxes or []
                    }
                }
                
                # Show extracted data with flags
                st.subheader(f"📋 Extracted Data - {uploaded_file.name}")
                display_data = {k: v for k, v in extracted_data.items() if not k.startswith("_")}
                st.json(display_data)
                
                new_results.append(result)
                
            else:
                new_results.append({
                    'filename': uploaded_file.name,
                    'success': False,
                    'error': ocr_result.get('error', 'Unknown error'),
                    'confidence': 0.0,
                    'processed_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'failed'
                })
            
            # Clean up temp file
            os.unlink(temp_path)
            
        except Exception as e:
            new_results.append({
                'filename': uploaded_file.name,
                'success': False,
                'error': str(e),
                'confidence': 0.0,
                'processed_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'failed'
            })
    
    # Add to session state
    st.session_state.processed_results.extend(new_results)
    
    # Show results
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    # Show summary
    successful = len([r for r in new_results if r['success']])
    st.success(f"🎉 Processed {len(new_results)} images: {successful} successful, {len(new_results)-successful} failed")
    
    if successful > 0:
        st.info("👁️ **Next Step:** Go to the 'Manual Review' tab to review and approve the extracted data")

def manual_review_tab(excel_loaded):
    """Step 2: Manual review of processed results"""
    st.header("👁️ Step 2: Manual Review & Approval")
    st.markdown("Review extracted data and approve for Excel export")
    
    # Get pending results
    pending_results = [r for r in st.session_state.processed_results if r['success'] and r.get('status') == 'pending_review']
    
    if not pending_results:
        st.info("🔍 No items pending review. Process some images first!")
        return
    
    st.subheader(f"📋 {len(pending_results)} Items Pending Review")
    
    # Review each result
    for i, result in enumerate(pending_results):
        st.markdown("---")
        
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.subheader(f"📄 {result['filename']}")
            
            with col2:
                confidence = result['confidence']
                if confidence > 0.8:
                    st.success(f"🎯 Confidence: {confidence:.1%}")
                elif confidence > 0.6:
                    st.warning(f"⚠️ Confidence: {confidence:.1%}")
                else:
                    st.error(f"🚨 Confidence: {confidence:.1%}")
            
            with col3:
                if result['validation']['is_valid']:
                    st.success("✅ Valid")
                else:
                    st.error("❌ Issues")
            
            # BHID Quick Normalization (outside form)
            if result['data'].get('bhid'):
                col_norm1, col_norm2 = st.columns([3, 1])
                with col_norm1:
                    st.write(f"**Current BHID:** {result['data'].get('bhid', 'N/A')}")
                with col_norm2:
                    if st.button("🔧 Normalize BHID", key=f"normalize_bhid_{i}", help="Apply BHID canonicalization"):
                        # Use cached engine for canonicalization
                        engine, _ = get_engine()
                        original = result['data'].get('bhid', '')
                        normalized = engine._canonicalize_bhid(original)
                        if normalized and normalized != original:
                            result['data']['bhid'] = normalized
                            st.success(f"BHID normalized: {original} → {normalized}")
                            st.rerun()
                        else:
                            st.info("BHID is already in canonical form")
            
            # Editable form for the data
            with st.form(f"review_form_{i}"):
                st.subheader("📝 Extracted Data (Editable)")
                
                data = result['data'].copy()
                
                # Create editable fields
                col1, col2 = st.columns(2)
                
                with col1:
                    # Core metadata
                    data['date'] = st.text_input("📅 Date", value=data.get('date', '') or '', key=f"date_{i}")
                    # Rig selection from canonical list
                    valid_rigs = ['KEMPE_U2AW', 'KEMPE_U39BQ', 'JBC 01', 'JBC 02', 'JBC 03', 'T07', 'T06', 'T13', 'BDU400']
                    current_rig = data.get('rig_id', valid_rigs[0]) if data.get('rig_id') in valid_rigs else valid_rigs[0]
                    data['rig_id'] = st.selectbox("⚙️ Rig ID", valid_rigs, index=valid_rigs.index(current_rig), key=f"rig_{i}")
                    # Shift and level
                    shift_options = ['Day', 'Night', '']
                    current_shift = data.get('shift', '') if data.get('shift') in ['Day', 'Night'] else ''
                    data['shift'] = st.selectbox("🕒 Shift", shift_options, index=shift_options.index(current_shift), key=f"shift_{i}")
                    data['level'] = st.text_input("🏷️ Level", value=data.get('level', '') or '', key=f"level_{i}")
                    data['cubby_id'] = st.text_input("📦 Cubby ID", value=data.get('cubby_id', '') or '', key=f"cubby_{i}")
                    
                    data['bhid'] = st.text_input("🕳️ BHID", value=data.get('bhid', '') or '', key=f"bhid_{i}")
                    # Note: BHID normalization will be handled via form submit
                    data['from_depth'] = st.number_input("📏 From Depth (m)", value=float(data.get('from_depth', 0) or 0), key=f"from_{i}")
                    data['to_depth'] = st.number_input("📏 To Depth (m)", value=float(data.get('to_depth', 0) or 0), key=f"to_{i}")
                
                with col2:
                    data['total_depth'] = st.number_input("📊 Total Depth (m)", value=float(data.get('total_depth', 0) or 0), key=f"total_{i}")
                    data['geology'] = st.text_area("🪨 Geology", value=data.get('geology', '') or '', key=f"geology_{i}")
                    data['comments'] = st.text_area("📝 Comments", value=data.get('comments', '') or '', key=f"comments_{i}")
                    data['driller'] = st.text_input("👷 Driller", value=data.get('driller', '') or '', key=f"driller_{i}")
                    data['geotech'] = st.text_input("🔬 Geotech", value=data.get('geotech', '') or '', key=f"geotech_{i}")
                    data['daily_budget'] = st.number_input("💰 Daily Budget", value=float(data.get('daily_budget', 0) or 0), key=f"budget_{i}")
                    # Activity grid editor with auto-recompute durations
                    st.markdown("⏱️ **Activity (edit From/To and text; Duration auto)**")

                    import re
                    rows = data.get("activity_rows") or []
                    df_src = [{
                        "From": r.get("from", ""),
                        "To": r.get("to", ""),
                        "Activity": r.get("text", "")
                    } for r in rows]

                    edited_df = st.data_editor(
                        pd.DataFrame(df_src, columns=["From","To","Activity"]),
                        num_rows="dynamic",
                        use_container_width=True,
                        key=f"activity_editor_{i}",
                        column_config={
                            "From": st.column_config.TextColumn(width="small", help="e.g. 07:20, 7.20, 0720"),
                            "To": st.column_config.TextColumn(width="small"),
                            "Activity": st.column_config.TextColumn(width="medium"),
                        }
                    )

                    def _parse_time(s: str):
                        s = str(s or "").strip().upper().replace("H","").replace(" ", "")
                        s = s.replace(".", ":").replace("-", ":").replace(";", ":")
                        m = re.match(r"^(\d{1,2})(?::?(\d{2}))?$", s)
                        if not m: 
                            return None
                        hh = int(m.group(1)); mm = int(m.group(2) or "00")
                        if hh > 24 or mm > 59: 
                            return None
                        
                        # Autosnap minutes to :00 or :30 if within 2 minutes
                        if abs(mm - 0) <= 2:
                            mm = 0
                        elif abs(mm - 30) <= 2:
                            mm = 30
                            
                        return (hh % 24) * 60 + mm

                    new_rows, bad_rows = [], []
                    for _, r in edited_df.iterrows():
                        f_raw = r["From"]; t_raw = r["To"]; txt = r["Activity"]
                        sf = _parse_time(f_raw); stt = _parse_time(t_raw)
                        dur = None
                        if sf is not None and stt is not None:
                            if stt < sf:  # midnight rollover
                                stt += 24*60
                            dur = round((stt - sf) / 60.0, 2)
                            if dur <= 0 or dur > 8:
                                dur = None
                        else:
                            bad_rows.append((f_raw, t_raw))
                        new_rows.append({"from": f_raw, "to": t_raw, "text": txt, "duration": dur})

                    # Persist back into the result's data
                    data["activity_rows"] = new_rows
                    data["activity_durations"] = [r["duration"] for r in new_rows if r["duration"] is not None]
                    data["activity_hours_total"] = round(sum(data["activity_durations"]), 2)

                    if bad_rows:
                        st.warning(f"⚠️ {len(bad_rows)} row(s) have unparseable times. Use ROI zoom to check those cells.")
                    
                    # Shift window guard: Check if Night shift activities fall outside [18:00–06:00]
                    if data.get('shift') == 'Night':
                        night_violations = []
                        for row in new_rows:
                            if row.get('duration') is not None:  # Only check valid time entries
                                f_time = _parse_time(row.get('from', ''))
                                t_time = _parse_time(row.get('to', ''))
                                if f_time is not None and t_time is not None:
                                    # Convert to hours for easier checking
                                    f_hour = f_time // 60
                                    t_hour = t_time // 60
                                    # Check if times fall outside night shift window [18:00-06:00+24]
                                    if not ((f_hour >= 18 or f_hour <= 6) and (t_hour >= 18 or t_hour <= 6)):
                                        night_violations.append(f"{row.get('from', '')} - {row.get('to', '')}")
                        
                        if night_violations:
                            st.warning(f"🌙 **Night Shift Warning**: Some activities may be outside typical night hours (18:00-06:00): {', '.join(night_violations)}")
                
                # Validation issues (if any)
                if not result['validation']['is_valid']:
                    st.error("⚠️ Validation Issues:")
                    for error in result['validation']['errors']:
                        st.write(f"  • {error}")
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.form_submit_button("✅ Approve for Export", type="primary"):
                        # Update data and approve
                        result['data'] = data
                        result['status'] = 'approved'
                        result['approved_at'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Re-validate with updated data
                        validator = DrillingDataValidator()
                        is_valid, validation_errors, cleaned_data = validator.validate_drilling_data(data)
                        result['validation'] = {
                            'is_valid': is_valid,
                            'errors': validation_errors
                        }
                        result['data'] = cleaned_data
                        
                        # Add to approved results
                        if excel_loaded and st.session_state.approved_results:
                            if result not in st.session_state.approved_results:
                                st.session_state.approved_results.append(result)
                        
                        st.success(f"✅ {result['filename']} approved for Excel export!")
                        st.rerun()
                
                with col2:
                    if st.form_submit_button("❌ Reject"):
                        result['status'] = 'rejected'
                        result['rejected_at'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                        st.warning(f"❌ {result['filename']} rejected")
                        st.rerun()
                
                with col3:
                    if st.form_submit_button("🔄 Reset"):
                        # Reset to original extracted data
                        st.info(f"🔄 {result['filename']} reset to original")
                        st.rerun()

def export_to_excel_tab(excel_loaded):
    """Step 3: Export approved results to Excel"""
    st.header("📊 Step 3: Export to Excel")
    st.markdown("Export approved data to Excel Forms tab")
    
    if not excel_loaded:
        st.error("🚨 Please upload an Excel file in the sidebar first!")
        return
    
    # Get approved results
    approved_results = [r for r in st.session_state.processed_results if r.get('status') == 'approved']
    
    if not approved_results:
        st.info("🔍 No approved items for export. Review and approve items first!")
        return
    
    st.subheader(f"📋 {len(approved_results)} Items Ready for Export")
    
    # Show approved items
    for result in approved_results:
        with st.expander(f"✅ {result['filename']} - Ready for Export"):
            display_extraction_summary(result)
    
    # Export button
    if st.button("📊 Export All to Excel", type="primary"):
        export_to_excel(approved_results)

def export_to_excel(approved_results):
    """Export approved results to Excel"""
    if not st.session_state.excel_file_path:
        st.error("No Excel file available!")
        return
    
    try:
        # Load Excel integration for Forms tab
        excel_integration = FormsTabIntegration(st.session_state.excel_file_path)
        
        # Test the integration and show rig sections found
        integration_info = excel_integration.test_integration()
        st.info(f"📋 Found {integration_info['rig_sections_found']} rig sections in Forms tab")
        
        if integration_info['rig_sections_found'] == 0:
            st.error("❌ No rig sections found in Forms tab. Please check your Excel file structure.")
            return
        
        # Export each approved result
        success_count = 0
        error_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, result in enumerate(approved_results):
            progress = (i + 1) / len(approved_results)
            progress_bar.progress(progress)
            status_text.text(f"Exporting {result['filename']}...")

            try:
                # Use the new sheet creation method instead of inserting into Forms tab
                insert_result = excel_integration.insert_drilling_data_new_sheet(result['data'])
                if insert_result.success:
                    result['export_status'] = 'exported'
                    result['exported_at'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    result['excel_worksheet'] = insert_result.worksheet_name
                    result['excel_row'] = insert_result.row_inserted
                    result['rig_section'] = insert_result.rig_section
                    success_count += 1
                    st.success(f"✅ {result['filename']} → New sheet '{insert_result.worksheet_name}' created")
                else:
                    result['export_status'] = 'export_failed'
                    result['export_error'] = insert_result.message
                    error_count += 1
                    st.error(f"❌ {result['filename']}: {insert_result.message}")
            except Exception as e:
                result['export_status'] = 'export_failed'
                result['export_error'] = str(e)
                error_count += 1
                st.error(f"❌ {result['filename']}: {str(e)}")
        
        # Final summary
        progress_bar.progress(1.0)
        status_text.text("✅ Export complete!")
        
        if success_count > 0:
            st.success(f"🎉 Successfully exported {success_count} items to Excel!")
        
        if error_count > 0:
            st.warning(f"⚠️ {error_count} items failed to export")
        
        # Provide download link
        if os.path.exists(st.session_state.excel_file_path):
            with open(st.session_state.excel_file_path, 'rb') as f:
                st.download_button(
                    label="📥 Download Updated Excel File",
                    data=f.read(),
                    file_name="Updated_Drilling_Report.xlsm",
                    mime="application/vnd.ms-excel.sheet.macroEnabled.12"
                )
    
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def display_extraction_summary(result):
    """Display a summary of extracted data"""
    data = result['data']
    col1, col2 = st.columns(2)
    
    with col1:
        if data.get('date'):
            st.write(f"📅 **Date:** {data['date']}")
        if data.get('shift'):
            st.write(f"🕒 **Shift:** {data['shift']}")
        if data.get('level'):
            st.write(f"🏷️ **Level:** {data['level']}")
        if data.get('rig_id'):
            st.write(f"⚙️ **Rig ID:** {data['rig_id']}")
        if data.get('cubby_id'):
            st.write(f"📦 **Cubby ID:** {data['cubby_id']}")
        if data.get('bhid'):
            st.write(f"🕳️ **BHID:** {data['bhid']}")
        if data.get('from_depth') is not None:
            st.write(f"📏 **From Depth:** {data['from_depth']} m")
        if data.get('to_depth') is not None:
            st.write(f"📏 **To Depth:** {data['to_depth']} m")
    
    with col2:
        if data.get('total_depth') is not None:
            st.write(f"📊 **Total Depth:** {data['total_depth']} m")
        if data.get('core_recovery') is not None:
            st.write(f"📊 **Core Recovery:** {data['core_recovery']}%")
        if data.get('rqd') is not None:
            st.write(f"📊 **RQD:** {data['rqd']}%")
        if data.get('daily_budget') is not None:
            st.write(f"💰 **Daily Budget:** {data['daily_budget']}")
        if data.get('geology'):
            st.write(f"🪨 **Geology:** {data['geology']}")
        if data.get('driller'):
            st.write(f"👷 **Driller:** {data['driller']}")
        if data.get('geotech'):
            st.write(f"🔬 **Geotech:** {data['geotech']}")
    
    # Display activity table if available
    if data.get('activity_rows'):
        st.subheader("⏱️ Activity Record")
        activity_rows = data['activity_rows']
        
        # Create DataFrame for better display
        df_data = []
        for i, row in enumerate(activity_rows, 1):
            df_data.append({
                'Row': i,
                'From': row.get('from', ''),
                'To': row.get('to', ''),
                'Activity': row.get('text', ''),
                'Duration (h)': row.get('duration', '')
            })
        
        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Show total duration
            total_hours = sum([float(row.get('duration', 0)) for row in activity_rows if row.get('duration') is not None])
            st.write(f"**Total Activity Hours:** {total_hours:.2f} h")
    elif data.get('activity_durations'):
        durations_str = ", ".join([str(d) for d in data['activity_durations']])
        st.write(f"⏱️ **Activity Durations:** {durations_str} h")
    if data.get('comments'):
        st.write(f"📝 **Comments:** {data['comments']}")

def results_history_tab():
    """Step 4: View all results history"""
    st.header("📋 Results History")
    
    if not st.session_state.processed_results:
        st.info("🔍 No results yet. Process some images first!")
        return
    
    # Summary statistics
    total = len(st.session_state.processed_results)
    successful = len([r for r in st.session_state.processed_results if r['success']])
    pending = len([r for r in st.session_state.processed_results if r.get('status') == 'pending_review'])
    approved = len([r for r in st.session_state.processed_results if r.get('status') == 'approved'])
    exported = len([r for r in st.session_state.processed_results if r.get('export_status') == 'exported'])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📁 Total Processed", total)
    with col2:
        st.metric("✅ Successful", successful)
    with col3:
        st.metric("👁️ Pending Review", pending)
    with col4:
        st.metric("✅ Approved", approved)
    with col5:
        st.metric("📊 Exported", exported)
    
    # Results table
    if st.button("📋 Show Detailed History"):
        results_data = []
        for result in st.session_state.processed_results:
            if result['success']:
                data = result['data']
                results_data.append({
                    'Filename': result['filename'],
                    'Status': result.get('status', 'unknown'),
                    'Date': data.get('date', ''),
                    'Rig ID': data.get('rig_id', ''),
                    'Hole ID': data.get('hole_id', ''),
                    'Confidence': f"{result['confidence']:.1%}",
                    'Valid': '✅' if result['validation']['is_valid'] else '❌',
                    'Processed At': result.get('processed_at', ''),
                    'Export Status': result.get('export_status', 'not_exported')
                })
        
        if results_data:
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)
    
    # Clear results
    if st.button("🗑️ Clear All History"):
        st.session_state.processed_results = []
        st.session_state.approved_results = []
        st.rerun()

def setup_roi_zoom_sidebar():
    """Setup ROI zoom functionality in sidebar"""
    with st.sidebar.expander("🔎 ROI Zoom", expanded=False):
        if 'current_crops' not in st.session_state or not st.session_state.current_crops:
            st.info("Process an image first to enable ROI zoom")
            return
            
        crops = st.session_state.current_crops
        st.write(f"**File:** {crops['filename']}")
        
        roi_choice = st.selectbox("Select Region", ["header", "activity", "metrics"])
        
        if roi_choice == "header":
            st.image(Image.fromarray(cv2.cvtColor(crops['header'].image, cv2.COLOR_BGR2RGB)), 
                    caption="Header Region", use_container_width=True)
        elif roi_choice == "metrics":
            st.image(Image.fromarray(cv2.cvtColor(crops['metrics'].image, cv2.COLOR_BGR2RGB)), 
                    caption="Metrics Region", use_container_width=True)
        else:  # activity
            st.image(Image.fromarray(cv2.cvtColor(crops['activity'].image, cv2.COLOR_BGR2RGB)), 
                    caption="Activity Region", use_container_width=True)
            
            # Show individual activity rows if available
            if hasattr(crops['activity'], 'row_bboxes') and crops['activity'].row_bboxes:
                rows = crops['activity'].row_bboxes
                st.write(f"**Activity Rows:** {len(rows)} detected")
                idx = st.number_input("Row index", min_value=1, max_value=len(rows), value=1, step=1)
                x, y, w, h = rows[idx-1]
                row_img = crops['activity'].image[y:y+h, x:x+w]
                st.image(Image.fromarray(cv2.cvtColor(row_img, cv2.COLOR_BGR2RGB)), 
                        caption=f"Activity Row {idx}", use_container_width=True)

def export_individual_to_new_sheet(result: Dict[str, Any]) -> bool:
    """Export individual result to a new Excel sheet"""
    try:
        if not st.session_state.excel_file_path:
            st.error("No Excel file loaded")
            return False
            
        # Initialize Excel integration
        excel_integration = FormsTabIntegration(st.session_state.excel_file_path)
        
        # Use the new sheet insertion method
        export_result = excel_integration.insert_drilling_data_new_sheet(result['data'])
        
        if export_result.success:
            st.info(f"✅ Created new sheet: '{export_result.worksheet_name}'")
            return True
        else:
            st.error(f"❌ Export failed: {export_result.message}")
            return False
            
    except Exception as e:
        st.error(f"❌ Export error: {str(e)}")
        return False

if __name__ == "__main__":
    main()
