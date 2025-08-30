# Multimodal Fraud Detection

## Overview

**Multimodal Fraud Detection** is a machine learning project designed to classify messages as either **fraudulent** or **normal** using **both text and audio inputs**.  
The system combines a **speech-to-text (STT)** model, **emotion detection model**, and traditional **text-based ML classifiers** to detect fraudulent patterns, including scam calls, phishing attempts, and suspicious messages.

## Goals

- Integrate audio and text features into a unified multimodal pipeline.  
- Extract and leverage **audio characteristics** such as duration, pitch, and loudness.  
- Capture **emotional cues** from speech that indicate suspicious intent.  
- Provide a reproducible and interpretable model for multimodal fraud detection.
## Model Evaluation

### Classification Report

| Label          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| 0              | 1.00      | 0.33   | 0.50     | 3       |
| 1              | 0.71      | 1.00   | 0.83     | 5       |
| **Accuracy**   | -         | -      | 0.75     | 8       |
| **Macro Avg**  | 0.86      | 0.67   | 0.67     | 8       |
| **Weighted Avg** | 0.82    | 0.75   | 0.71     | 8       |

### Confusion Matrix

|               | Predicted 0 | Predicted 1 |
|---------------|------------|------------|
| **Actual 0**  | 3          | 1          |
| **Actual 1**  | 0          | 5          |

### Sample Fraud Probabilities (%)

```
[97.0, 76.5, 72.0, 95.0, 94.5]
```

## Demo

All data is **simulated or publicly available**. No real telecom or sensitive customer data is used.  
This project is for demonstration and research purposes only and is **not intended for deployment** in production environments.

## License  
Unauthorized use or deployment outside of permissions granted by the owner is prohibited.

For usage inquiries, please contact mandelakorilogan@gmail.com

© 2025 Mandela Logan. All rights reserved.
