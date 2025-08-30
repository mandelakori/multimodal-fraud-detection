# Multimodal Fraud Detection

## Overview

**Multimodal Fraud Detection** is a machine learning project designed to classify messages as either **fraudulent** or **normal** using **both text and audio inputs**.  
The system combines a **speech-to-text (STT)** model, **emotion detection model**, and traditional **text-based ML classifiers** to detect fraudulent patterns, including scam calls, phishing attempts, and suspicious messages.

## Goals

- Integrate audio and text features into a unified multimodal pipeline.  
- Extract and leverage **audio characteristics** such as duration, pitch, and loudness.  
- Capture **emotional cues** from speech that indicate suspicious intent.  
- Provide a reproducible and interpretable model for multimodal fraud detection.
## Model Performance Summary

- **Best CV ROC-AUC:** 0.991  
- **5-Fold CV ROC-AUC:** 0.980 ± 0.024  
- **Accuracy:** 0.80  

### Classification Metrics

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.71      | 1.00   | 0.83     | 5       |
| 1     | 1.00      | 0.60   | 0.75     | 5       |
| **Macro Avg** | 0.86 | 0.80 | 0.79 | 10      |
| **Weighted Avg** | 0.86 | 0.80 | 0.79 | 10   |

### Confusion Matrix

|               | Predicted 0 | Predicted 1 |
|---------------|------------|------------|
| **Actual 0**  | 5          | 0          |
| **Actual 1**  | 2          | 3          |

## Demo

All data is **simulated or publicly available**. No real telecom or sensitive customer data is used.  
This project is for demonstration and research purposes only and is **not intended for deployment** in production environments.

## License  
Unauthorized use or deployment outside of permissions granted by the owner is prohibited.

For usage inquiries, please contact mandelakorilogan@gmail.com

© 2025 Mandela Logan. All rights reserved.
