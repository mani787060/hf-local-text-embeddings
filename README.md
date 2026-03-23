# Local Text Embeddings with Hugging Face
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Sentence--Transformers-orange)](https://sbert.net/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)
[![Privacy](https://img.shields.io/badge/Data-Privacy--First-green)](https://en.wikipedia.org/wiki/Data_privacy)

## 🏗️ Project Overview
This repository demonstrates how to generate **Text Embeddings** locally using the `sentence-transformers` library and Hugging Face models. By running embedding models (like SBERT or BGE) on local infrastructure, this project highlights how to achieve state-of-the-art semantic search capabilities while maintaining complete data privacy and zero API costs.

---

## 🛠️ Key Technical Implementations

### 1. Model Loading & Selection
* **SBERT Architectures:** Utilizing models like `all-MiniLM-L6-v2` for a balance of speed and accuracy.
* **Local Caching:** Managing model weights locally to allow for offline inference.

### 2. Hardware Acceleration (CUDA/MPS)
* **Device Mapping:** Automatically detecting and utilizing NVIDIA GPUs (CUDA) or Apple Silicon (MPS) for high-speed vectorization.
* **Batch Encoding:** Efficiently processing large datasets by leveraging parallel tensor operations.

### 3. Semantic Analysis
* **Cosine Similarity:** Implementing vector-space comparisons to find the "semantic distance" between sentences.
* **Paraphrase Mining:** Using local models to identify duplicate or highly similar content within a dataset.

### 4. Comparison & Benchmarking
* Analyzing the trade-offs between local execution (Latency/Privacy) vs. API-based execution (Scalability/Convenience).

---

## 💻 Tech Stack
* **Language:** Python
* **Core Libraries:** `sentence-transformers`, `transformers`, `torch`
* **Mathematics:** NumPy, SciPy
* **Platform:** Jupyter Notebook / Google Colab

