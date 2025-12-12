# Medical Text Classification using Large Language Models (LLMs)

## 🎓 Project Goal
This capstone project aims to conduct a comprehensive comparative analysis between traditional Natural Language Processing (NLP) classification methods and modern Large Language Model (LLM) approaches for classifying complex medical texts. We will investigate the efficacy of LLMs, specifically focusing on few-shot learning techniques, which is crucial in data-scarce medical domains.

## 📊 Core Objectives
1.  **Baseline Generation:** Implement and evaluate traditional models (SVM, Logistic Regression, Random Forest) using TF-IDF features to establish a performance benchmark.
2.  **LLM Fine-Tuning:** Fine-tune a domain-specific LLM (e.g., BioBERT) for sequence classification on cancer pathology reports.
3.  **Few-Shot Analysis:** Explore few-shot learning and prompt engineering with an LLM on medical review data to assess performance with limited labeled examples.

## 📁 Repository Structure
* `data/`: Stores raw and processed datasets (TCGA, IMR).
* `notebooks/`: Jupyter Notebooks for exploration, training, and analysis.
* `src/`: Production-ready Python modules (`preprocessing.py`, `traditional_models.py`, `llm_pipeline.py`).
* `reports/`: Final report documentation and assets.

## 🛠️ Technology Stack
* **Language:** Python 3.x
* **Traditional NLP:** `scikit-learn`, `nltk`
* **Deep Learning/LLMs:** `torch`, `transformers` (Hugging Face), `datasets`