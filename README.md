# 🔍 Image Similarity Search using Deep Learning & FAISS

This project implements an **end-to-end image similarity search system** using **deep learning–based feature extraction** and **vector similarity search with FAISS**.

Given a **query image (including unseen images from the web)**, the system retrieves the **most visually similar images** from a dataset by comparing deep image embeddings instead of relying on class labels.

This project demonstrates a **real-world computer vision retrieval pipeline** commonly used in recommendation systems, visual search engines, and e-commerce platforms.

---

## 📌 Problem Statement

Image similarity answers the question:

> **“Which images look similar to this one?”**

This project addresses the image similarity problem by:
- Converting images into numerical embeddings using a pretrained CNN  
- Comparing images using vector similarity instead of raw pixel values  
- Retrieving visually similar images efficiently  

---

## 🏗️ System Architecture

Dataset Images
↓
Preprocessing (Resize, Normalize)
↓
Pretrained ResNet-50
↓
2048-D Image Embeddings
↓
L2 Normalization
↓
FAISS Similarity Index
↓
Top-K Image Retrieval


---

## 🔧 Technologies Used

- Python  
- PyTorch  
- Torchvision  
- FAISS  
- NumPy  
- Matplotlib  
- Pillow (PIL)  

---

## 📂 Dataset

- A **subset of images** was used from the **Stanford Online Products Dataset**  
🔗 **Dataset Link:**  
https://cvgl.stanford.edu/projects/lifted_struct/

---

## 🧠 Model Details

- **Backbone:** ResNet-50 (pretrained on ImageNet)  
- **Usage:** Feature extractor (final classification layer removed)  
- **Embedding Size:** 2048 dimensions  
- **Training:** No training or fine-tuning performed  

The pretrained model is used **purely for inference**, leveraging learned visual representations.

---

## 🔍 Similarity Search

- **Similarity Measure:** Cosine similarity  
- **Implementation:** FAISS `IndexFlatIP`  
- **Search Type:** Exact nearest-neighbour search (no approximation)  

---

## 📊 Evaluation Strategy

Since image similarity is inherently subjective, evaluation is performed using:

- Visual inspection of retrieved images  
- Displaying the query image alongside top-K retrieved results  

---

## 🖼️ Sample Output

[ Query Image ] → [ Rank 1 ] [ Rank 2 ] [ Rank 3 ] [ Rank 4 ] [ Rank 5 ] [ Rank 6 ] [ Rank 7 ] [ Rank 8 ] [ Rank 9 ] [ Rank 10 ] 


The retrieved images show strong **visual and semantic similarity** to the query image.

---

## 🔮 Possible Enhancements

- Replace ResNet-50 with Vision Transformers (ViT)  
- Fine-tune embeddings using contrastive or triplet loss  
- Use approximate FAISS indexes for large-scale datasets  
- Build an interactive UI using Streamlit  
- Persist FAISS index and embeddings to disk  

---
