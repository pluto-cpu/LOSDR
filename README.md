# LOSDR

# text_extract.py
## CLIP-based Drone Text Embedding & Similarity Visualization  ---text_extract.py

This project utilizes the [OpenAI CLIP](https://huggingface.co/openai/clip-vit-base-patch32) model to construct text prompts for drone/remote controller categories, extract text embeddings, and compute a semantic similarity matrix between categories.  
Based on this, **heatmaps** and **PCA scatter plots** are generated to analyze the semantic distribution among categories.

- **Multi-Template Prompt Ensemble**  
  Generates multiple Chinese and English prompts for each category and applies average encoding to enhance semantic representation stability.

- **Auto Descriptor**  
  Automatically appends differentiating information such as brand, series, frequency band, and signal pattern beyond category names to reduce excessive similarity between categories.

- **Chinese-English Bilingual Fusion**  
  Provides prompts in both languages to enhance cross-lingual robustness.

- **Semantic Anchors**  
  Incorporates task-relevant keywords like RF / spectrogram / hopping / bandwidth to emphasize modal information and task context.

- **PCA Whitening (Optional)**  
  Removes common components to increase dispersion in category embeddings, improving discriminative power.

- **Visual Outputs**  
  - Similarity matrix heatmap (supports numerical annotations, letters A–X as category indices)
  - PCA dimensionality reduction scatter plot
  - Final prompt list corresponding to categories (saved as TXT/JSON)

# Matlab Code

