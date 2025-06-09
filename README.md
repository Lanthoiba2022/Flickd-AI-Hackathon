# Flickd AI Hackathon - Smart Tagging & Vibe Classification Engine

## Overview
This MVP implements Flickd's intelligent system for:
- Extracting frames from fashion videos
- Detecting fashion items using YOLOv8
- Matching detected items to product catalog using CLIP embeddings & FAISS
- Classifying video vibes using NLP

## Setup Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv flickd_env
source flickd_env/bin/activate  # On Windows: flickd_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Project Structure
Ensure your project follows this structure:
```
submission/
├── videos/              # Your video files (.mp4)
├── catalog.csv(product_data + images)         # Product catalog
├── vibeslist.json     # Vibe categories
├── outputs/            # Generated JSON outputs (created automatically)
├── models/             # Downloaded models and cache (created automatically)
├── src/                # Source code
├── run_pipeline.py     # Main execution script
├── requirements.txt
└── README.md
```

### 3. Data Preparation
- Place all .mp4 video files in the `videos/` directory
- Ensure `catalog.csv` has columns: product_id, title, shopify_cdn_url, category, color
- Caption files (.txt) should be in the same directory as videos with matching names

### 4. Run Pipeline
```bash
python run_pipeline.py
```

## Output Format
For each video, generates a JSON file in `outputs/` directory:
```json
{ 
  "video_id": "reel_001", 
  "vibes": ["Coquette", "Brunchcore"], 
  "products": [ 
    { 
      "type": "top", 
      "color": "white", 
      "matched_product_id": "prod_002", 
      "match_type": "exact", 
      "confidence": 0.93 
    }, 
    { 
      "type": "earrings", 
      "color": "gold", 
      "matched_product_id": "prod_003", 
      "match_type": "similar", 
      "confidence": 0.85 
    } 
  ] 
}

Google Colab Link: https://colab.research.google.com/drive/1uIv3fIZOCwrE4cCf2c7Tqq_AiPeQgi1c?usp=sharing