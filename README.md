# 💎 Diamond Drilling OCR System v4.2.2b

An advanced AI-powered OCR system specifically designed for diamond drilling forms, featuring intelligent data extraction, manual review workflows, and seamless Excel integration.

## 🌟 Features

### 🤖 Advanced AI-Powered OCR
- **GPT-5 Integration**: Leverages OpenAI's GPT-5 for superior text recognition and data extraction
- **Multi-Region Approach**: Intelligent 3-region extraction (Header, Activity, Metrics) with targeted prompts
- **Re-ask Loop**: Automated quality assurance with intelligent re-processing for improved accuracy
- **Context-Aware Processing**: Smart canonicalization using drilling context and operational rules

### 📋 Manual Review Workflow
- **Interactive Data Editor**: Modern Streamlit interface with side-by-side editing
- **Activity Grid Editor**: Advanced table editing with automatic duration calculation
- **Real-time Validation**: Instant feedback on data quality and consistency
- **ROI Zoom**: Interactive region-of-interest viewer for detailed inspection
- **One-click BHID Normalization**: Smart canonicalization with O/0 correction and suffix padding

### 🔧 Intelligent Data Processing
- **Enhanced BHID Canonicalization**: Context-aware hole ID standardization with suffix normalization
- **Activity Text Isolation**: Vertical projection-based separation of activity text from checklists
- **Tolerant Time Parsing**: Robust parsing of various time formats (24:00, 23.50, 05-00, 2230)
- **Dual-Read Numeric Extraction**: Consensus reading from normal and binarized images
- **Operational Rules Engine**: Night shift allowances, meter caps, and context flags

### 📊 Excel Integration
- **Forms Tab Export**: Direct integration with existing Excel drilling forms
- **Individual Sheet Export**: Create new worksheets for each drilling record
- **Bulk Processing**: Export multiple approved records simultaneously
- **Data Validation**: Automated consistency checks and error flagging

### 🎯 User Experience
- **Autosnap Time Correction**: Automatic rounding to :00/:30 intervals (within 2-minute tolerance)
- **Shift Window Guard**: Warnings for activities outside expected shift hours
- **Progress Tracking**: Real-time processing indicators and status updates
- **Error Recovery**: Graceful handling of processing failures with detailed logging

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key with GPT-5 access
- Streamlit Cloud account (for deployment)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd v4.2.2b
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   export OPENAI_MODEL="gpt-5"  # or gpt-4o
   ```

### Local Development

Run the Streamlit application locally:
```bash
streamlit run app.py
```

Access the application at `http://localhost:8501`

## ☁️ Streamlit Cloud Deployment

### Configuration

The system is optimized for Streamlit Cloud deployment with automatic secret management.

1. **Deploy to Streamlit Cloud**:
   - Connect your GitHub repository
   - Set the main file path: `app.py`

2. **Configure Secrets**:
   In your Streamlit Cloud app settings, add:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key"
   OPENAI_MODEL = "gpt-4o"
   ```

3. **Deploy**:
   The app will automatically use Streamlit secrets and fallback to environment variables.

## 📖 Usage Guide

### 1. Upload and Process
- Upload diamond drilling form images (PNG, JPG, PDF)
- View 3-region crops (Header, Activity, Metrics) side-by-side
- Monitor real-time OCR processing progress

### 2. Manual Review
- Edit extracted data using intuitive form fields
- Use the Activity Grid Editor for precise time/activity management
- Apply one-click BHID normalization
- Review validation warnings and errors
- Approve records for Excel export

### 3. Export to Excel
- Upload your Excel workbook template
- Export approved records to Forms tab or individual sheets
- Monitor export progress and success rates

### 4. ROI Zoom (Sidebar)
- Select specific regions for detailed inspection
- Zoom into header, activity rows, or metrics sections
- Perfect for quality assurance and troubleshooting

## 🏗️ Architecture

### Core Components

```
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration and field mappings
├── gpt5_ocr_engine.py         # GPT-5 OCR processing engine
├── data_validator.py          # Data validation and cleaning
├── excel_forms_integration.py # Excel export functionality
├── simple_preprocessor.py     # Image preprocessing
├── region_cropping.py         # Multi-region extraction
└── patches/                   # Version-specific enhancements
```

### Data Flow

1. **Image Upload** → Preprocessing → Region Cropping
2. **OCR Processing** → 3-Region Extraction → Re-ask Loop
3. **Data Validation** → Canonicalization → Error Flagging  
4. **Manual Review** → User Edits → Approval
5. **Excel Export** → Forms Integration → Success Tracking

## ⚙️ Configuration

### Field Mappings
The system uses semantic field mappings defined in `config.py`:
- **Core Metadata**: Date, Shift, Rig ID, Level, Cubby ID, BHID
- **Depth Information**: From/To depths, Total depth
- **Personnel**: Driller, Geotech names with fuzzy matching
- **Activity Data**: Time-based activity logging with duration calculation

### Validation Rules
- **Meters Alignment**: FROM/TO vs Meter Start/End consistency
- **Time Validation**: Shift-appropriate time windows
- **BHID Canonicalization**: Context-aware hole ID standardization
- **Personnel Matching**: Fuzzy matching against known rosters

## 🔧 Advanced Features

### Enhanced BHID Canonicalization
- **Suffix Normalization**: `EV2` → `EV02`, `EVO2` → `EV02`
- **O/0 Correction**: Automatic optical character confusion fixes
- **Context Fallback**: Uses Level/Cubby context to infer missing digits
- **Underscore Standardization**: Consistent formatting across systems

### Activity Processing
- **Column Isolation**: Vertical projection separates text from checkboxes
- **Time Format Tolerance**: Handles 00:00, 23.50, 05-00, 2230 formats
- **Duration Auto-calculation**: Real-time updates with midnight rollover
- **Shift Window Validation**: Warns about activities outside expected hours

### Operational Intelligence
- **Night Shift Rules**: Extended time windows (18:00-06:00+24)
- **Meter Soft Caps**: Warnings for unusual drilling distances
- **Context Flags**: Smart hints for data quality assessment
- **Dual-Read Consensus**: Agreement between normal/binarized extractions

## 📋 Requirements

### Core Dependencies
```
streamlit >= 1.28.0
openai >= 1.0.0
opencv-python >= 4.8.0
pillow >= 10.0.0
pandas >= 2.0.0
numpy >= 1.24.0
openpyxl >= 3.1.0
```

### Optional Dependencies
- **PDF Support**: `pdf2image` for PDF processing
- **Advanced Image Processing**: `scikit-image` for enhancement
- **Fuzzy Matching**: `difflib` (included in standard library)

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes and test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 conventions
- Use type hints where appropriate
- Document complex functions with docstrings
- Test edge cases and error conditions

## 📊 Performance

### Benchmarks
- **Processing Speed**: ~30-45 seconds per drilling form
- **Accuracy Rate**: >95% for clear, well-oriented images
- **Memory Usage**: <500MB typical, <1GB peak
- **Concurrent Users**: Supports multiple users via Streamlit Cloud

### Optimization Tips
- Use high-quality, well-oriented images for best results
- Ensure good contrast between text and background
- Pre-crop images to remove unnecessary margins
- Use recommended image sizes (max 2048x2048px)

## 🐛 Troubleshooting

### Common Issues

**OCR Accuracy Problems**:
- Ensure images are landscape-oriented
- Check for sufficient contrast and resolution
- Use the ROI zoom feature to inspect problem areas

**Excel Export Failures**:
- Verify Excel file format (.xlsx recommended)
- Check that Forms tab exists in workbook
- Ensure sufficient permissions for file writing

**API Rate Limits**:
- Monitor OpenAI usage quotas
- Consider batch processing for large volumes
- Implement retry logic for transient failures

**Performance Issues**:
- Use Streamlit Cloud for better performance
- Consider image preprocessing optimization
- Monitor memory usage during batch operations

## 📄 License

This project is proprietary software developed for diamond drilling operations. All rights reserved.

## 👥 Support

For technical support, feature requests, or bug reports:
- Create an issue in the project repository
- Contact the development team
- Refer to the troubleshooting guide above

## 🔄 Version History

### v4.2.2b (Current)
- ✅ Streamlit Cloud deployment optimization
- ✅ Enhanced manual review with activity grid editor
- ✅ Improved BHID canonicalization with suffix normalization
- ✅ Fixed data sync issues between manual review and export
- ✅ Added UX improvements (autosnap, shift guards)

### v4.2.2
- ✅ Manual review UI enhancements
- ✅ Activity grid editor with auto-duration calculation
- ✅ Core Recovery/RQD UI cleanup
- ✅ One-click BHID normalization

### v4.2.1 
- ✅ Enhanced page normalization with four-point dewarp
- ✅ Activity column isolation via vertical projection
- ✅ Tolerant time parsing with enhanced regex
- ✅ Dual-read numeric cell extraction

### v4.2.0
- ✅ Multi-region extraction with 3 prompts + re-ask loop
- ✅ Enhanced BHID/Cubby canonicalization
- ✅ Activity table with from/to/text extraction
- ✅ Comprehensive data validation pipeline

---

**Built with ❤️ for the Diamond Drilling Industry**

*Streamlining drilling data extraction with cutting-edge AI technology*
