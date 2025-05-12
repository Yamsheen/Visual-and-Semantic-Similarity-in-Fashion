# Visual-and-Semantic-Similarity-in-Fashion
Visual and Semantic Similarity in Fashion: A Dual-Encoder Approach with VAE and CLIP

# Multimodal Fashion Retrieval using VAE and CLIP-style Dual Encoder

## Overview

This project focuses on developing a multimodal fashion item retrieval system that supports both image-based and text-based queries. By leveraging deep learning techniques, we enable users to retrieve visually and semantically similar fashion products from a dataset, specifically targeting the "tops" category. Our system integrates:

- A **Variational Autoencoder (VAE)** for unsupervised image similarity retrieval.
- A **CLIP-style dual encoder model** for text-to-image retrieval via contrastive learning.

Both models are trained on a curated subset of the **Fashion200K** dataset and evaluated using cosine similarity and semantic consistency metrics.

---

## Objectives

- Enable **image-to-image retrieval** through unsupervised representation learning.
- Enable **text-to-image retrieval** using semantic embedding alignment.
- Evaluate retrieval consistency across visual features such as color, texture, and structure.
- Test generalization to out-of-domain queries (e.g., garments like "kurta").

---

## Dataset

We used a filtered subset of the **Fashion200K** dataset focusing on "tops". The final subset included:

- ~7000 image-text pairs
- Annotated attributes: color, sleeve style, neckline, texture
- Train-test split: 80:20 ratio

---

## Methodology

### Image Preprocessing
- Resized to 224×224
- Normalized using ImageNet statistics

### Text Preprocessing
- Tokenized with BERT (bert-base-uncased)
- Used [CLS] token embeddings for sentence representation

---

## Models Implemented

### 1. Variational Autoencoder (VAE)
- Encoder: Convolutional layers producing latent vector (size 64)
- Decoder: Transposed convolutions for reconstruction
- Loss: Reconstruction loss + KL divergence
- Retrieval: Nearest neighbor search in latent space via cosine similarity

### 2. CLIP-style Dual Encoder
- Image encoder: Shallow CNN
- Text encoder: Pretrained BERT with linear projection
- Training: Contrastive loss
- Retrieval: Cosine similarity in shared latent space

---

## Improvements & Experiments

- Introduced **controlled variation** experiments to analyze impact of sleeve, neckline, texture, and pose on retrieval similarity.
- Analyzed model behavior when given **out-of-distribution queries** (e.g., "kurta" not present in training data).
- Developed a **Visual-Semantic Consistency (VSC)** metric using BERT embeddings to quantify alignment of retrievals with query descriptions.

---

## Results Summary

### VAE Retrieval
- **Top-1 accuracy**: 80% on 50 random test queries
- High consistency on color and general silhouette
- Weaker performance on fine-grained texture
- Pose variations significantly affected retrievals

### CLIP Retrieval
- **Best preserved**: Color, general garment structure
- **Less preserved**: Specific garment type (e.g., "kurta"), knit patterns
- VSC scores remained high even when cosine similarity dropped, showing semantic alignment

| Query                             | CosSim | VSC  | Notes                                      |
|----------------------------------|--------|------|--------------------------------------------|
| green floral                     | 0.87   | 0.85 | Ideal retrieval in both metrics            |
| black tank top                   | 0.65   | 0.99 | Semantically perfect but visually disjoint |
| black v-neck sweater (honeycomb) | 0.63   | 0.88 | Poor visual match; semantic match preserved|

*Refer to `clip_results_table.tex` for full table of results.*

---

## Key Takeaways

- VAE excels in preserving global visual structure but is sensitive to pose and lacks fine-grained texture understanding.
- CLIP is robust in semantic retrieval even with out-of-domain queries but can struggle with culturally specific garments.
- The combination of **Cosine Similarity + VSC** provides a well-rounded evaluation.

---

## Future Work

- Improve VAE with **pose-invariant** and **texture-sensitive** embeddings.
- Fine-tune CLIP with a **more diverse, culturally inclusive dataset**.
- Develop a hybrid pipeline: **CLIP for candidate filtering**, **VAE for visual re-ranking**.

---


## Authors

- Shanzay Omar  
- Yamsheen Saqib  
