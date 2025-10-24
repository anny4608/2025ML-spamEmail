# Advanced Spam Classification System
This is an AiSecurity homework 2.
The repository contains a sophisticated Spam Classification system with an interactive web interface, advanced visualizations, and comprehensive text preprocessing capabilities.


## 🎵Source Reference
This project builds upon patterns and datasets related to the Spam Email problem from Chapter 3 of the Packt repository below. We used it to expand the preprocessing steps and add richer visualization work (step outputs, metrics, and CLI/Streamlit views).

https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity.git

## 💫Demo site
https://aisecurityhw2-2025ml-spamemail.streamlit.app/

## 🌟 Features

- **Interactive Web Interface**: Modern Streamlit dashboard with:
  - Real-time spam classification
  - Advanced data visualizations
  - Multiple text preprocessing views
  - Performance metrics analysis

- **Text Preprocessing Pipeline**:
  - Basic cleaning (whitespace normalization)
  - Case normalization
  - Contact information masking (phone, email, URLs)
  - Special character removal
  - Number handling
  
- **Advanced Visualizations**:
  - Interactive token frequency analysis
  - Class distribution pie charts
  - ROC curves and AUC metrics
  - Performance metrics at different thresholds
  - Feature importance analysis

- **Model Performance**:
  - Cross-validation metrics
  - Threshold optimization
  - Comprehensive performance metrics
  - Interactive metric exploration

## 🚀 Quickstart (PowerShell)

```powershell
git clone <repo-url>
cd <repo-folder>
python -m venv .venv
. \.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train a model (adjust path to dataset)
python src\train.py --data "sms_spam_no_header.csv"

# Run the CLI for quick testing
python src\cli.py --message "You have won a free prize"

# Launch the interactive dashboard
streamlit run src\app_streamlit.py
```

## 📊 Dashboard Features

1. **Data Overview**
   - Total message statistics
   - Class distribution visualization
   - Dataset summary metrics

2. **Token Analysis**
   - Interactive token frequency plots
   - Class-wise token distribution
   - Adjustable token count visualization

3. **Model Performance**
   - ROC curve with AUC score
   - Precision-Recall metrics
   - Interactive threshold adjustment
   - Performance visualization at different thresholds

4. **Live Testing**
   - Real-time message classification
   - Probability scores
   - Feature importance analysis
   - Example message templates

Folder layout:

- `src/` - source modules and scripts
- `tests/` - pytest tests
- `models/` - saved model artifacts (gitignored)
- `data/` - optional local data (gitignored)

Contributing: please follow `CONTRIBUTING.md`.
