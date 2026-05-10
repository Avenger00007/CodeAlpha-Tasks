# 🤖 FAQ Chatbot — NLP Powered with Cosine Similarity

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**An intelligent FAQ Chatbot built with NLP techniques — tokenization, lemmatization, TF-IDF vectorization, and cosine similarity — wrapped in a sleek Jupyter widget UI.**

[📦 Installation](#-installation) • [🚀 Quick Start](#-quick-start) • [🧠 How It Works](#-how-it-works) • [📁 Project Structure](#-project-structure) • [🖼️ Features](#-features)

</div>

---

## 📌 About The Project

This project implements a **rule-free FAQ Chatbot** that doesn't rely on hardcoded `if-else` logic. Instead, it uses real **Natural Language Processing** to understand user questions and match them to the most relevant FAQ using **cosine similarity** on **TF-IDF vectors**.

> 💡 Built as a beginner-to-intermediate NLP project — clean, well-commented, and runs 100% on CPU.

---

## 🎯 Features

| Feature | Description |
|--------|-------------|
| 🧹 **Text Preprocessing** | Lowercasing, regex cleaning, tokenization, stopword removal, lemmatization |
| 📐 **TF-IDF Vectorization** | Converts text to numerical vectors using unigrams + bigrams |
| 🔍 **Cosine Similarity** | Matches user query to the most similar FAQ question |
| 🎯 **Confidence Score** | Displays match confidence % for every response |
| 📌 **Matched FAQ Display** | Shows which FAQ was matched for transparency |
| 💬 **Chat UI** | Beautiful dark-themed chat interface using `ipywidgets` |
| ❌ **Graceful Fallback** | Smart "no match" response when confidence is too low |
| ⚡ **CPU Friendly** | No GPU, no transformers, no heavy models needed |

---

## 🧠 How It Works

```
User Question
      │
      ▼
┌─────────────────────┐
│   Text Preprocessing │  ← lowercase, clean, tokenize, remove stopwords, lemmatize
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  TF-IDF Vectorizer  │  ← transforms text into numerical feature vectors
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│  Cosine Similarity  │  ← compares query vector vs all FAQ vectors
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   Best Match FAQ    │  ← returns answer if score ≥ threshold (0.15)
└─────────────────────┘
      │
      ▼
   💬 Chatbot Response
```

### NLP Pipeline Breakdown

| Step | Technique | Library |
|------|-----------|---------|
| Text Cleaning | Regex + Lowercase | `re` |
| Tokenization | `word_tokenize` | `NLTK` |
| Stopword Removal | Custom filtered stopwords | `NLTK` |
| Lemmatization | `WordNetLemmatizer` | `NLTK` |
| Vectorization | TF-IDF (1-gram + 2-gram) | `scikit-learn` |
| Similarity Matching | Cosine Similarity | `scikit-learn` |
| Chat UI | Interactive Widgets | `ipywidgets` |

---

## 📁 Project Structure

```
faq_chatbot/
│
├── 📓 faq_chatbot.ipynb        # Main Jupyter Notebook (all code here)
├── 📄 README.md                # Project documentation (this file)
└── 📦 requirements.txt         # All required Python libraries
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **NLP:** NLTK (tokenization, stopwords, lemmatization)
- **ML/Math:** Scikit-learn (TF-IDF, cosine similarity), NumPy
- **UI:** ipywidgets, IPython Display, HTML/CSS
- **Environment:** Jupyter Notebook / VS Code

---

## 📦 Installation

### ✅ Prerequisites

Make sure you have the following installed:
- [Python 3.9+](https://www.python.org/downloads/)
- [VS Code](https://code.visualstudio.com/) with **Python** and **Jupyter** extensions
- OR [Anaconda](https://www.anaconda.com/) (comes with Jupyter)

---

### 🔧 Step-by-Step Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/faq-chatbot.git
cd faq-chatbot
```

**2. Create a virtual environment**
```bash
python -m venv chatbot_env
```

**3. Activate the virtual environment**

On Windows:
```bash
chatbot_env\Scripts\activate
```

On Mac / Linux:
```bash
source chatbot_env/bin/activate
```

> You should now see `(chatbot_env)` at the start of your terminal line ✅

**4. Install required libraries**
```bash
pip install -r requirements.txt
```

**5. Download NLTK data** *(run once)*
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

---

## 🚀 Quick Start

**Option A — VS Code (Recommended)**
1. Open the project folder in VS Code
2. Open `faq_chatbot.ipynb`
3. Click **"Select Kernel"** → choose `chatbot_env`
4. Run all cells with `Shift + Enter` (top to bottom)

**Option B — Jupyter in Browser**
```bash
jupyter notebook
```
Then open `faq_chatbot.ipynb` in the browser tab that opens.

---

## 📓 Notebook Cells Guide

| Cell | Purpose |
|------|---------|
| **Cell 1** | Imports all libraries and downloads NLTK data |
| **Cell 2** | Defines 24 FAQs about a smartphone (PhoneX Pro) |
| **Cell 3** | NLP preprocessing pipeline (clean → tokenize → lemmatize) |
| **Cell 4** | Builds TF-IDF model + cosine similarity matching function |
| **Cell 5** | Renders the interactive chat UI using ipywidgets |
| **Cell 6** | *(Optional)* Tests all queries and prints similarity scores |

---

## 📋 Requirements

Create a `requirements.txt` file with:

```
nltk==3.8.1
scikit-learn==1.4.0
numpy==1.26.0
ipywidgets==8.1.2
notebook==7.1.0
```

Or install manually:
```bash
pip install nltk scikit-learn numpy ipywidgets notebook
```

---

## 🗂️ FAQ Categories Covered

The chatbot includes **24 FAQs** across 6 categories for a fictional smartphone **PhoneX Pro**:

- 💰 **Pricing & Purchase** — cost, where to buy, EMI options
- 🔋 **Battery** — capacity, charging time, wireless charging
- 📸 **Camera** — megapixels, video recording, front camera
- 📱 **Display** — screen size, refresh rate, water resistance
- ⚙️ **Performance** — processor, RAM, gaming
- 🛡️ **Software & Support** — Android version, fingerprint, warranty, returns

---

## ⚠️ Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'nltk'` | Library not installed | Run `pip install nltk` inside activated venv |
| `LookupError: Resource punkt not found` | NLTK data missing | Run the NLTK download command in Step 5 |
| `Widget not rendering` | ipywidgets not installed or old | Run `pip install ipywidgets` and restart kernel |
| `Kernel not found in VS Code` | Wrong Python interpreter | Click kernel selector → choose `chatbot_env` |
| `ModuleNotFoundError: No module named 'sklearn'` | scikit-learn missing | Run `pip install scikit-learn` |

---

## 📈 Sample Output

```
======================================================================
  📊 COSINE SIMILARITY TEST RESULTS
======================================================================

✅ Query   : How much does the phone cost?
   Matched : What is the price of the phone?
   Score   : 78.4%
   Answer  : 💰 The PhoneX Pro starts at ₹49,999 for the base 128GB...

✅ Query   : Is it waterproof?
   Matched : Is the display water resistant?
   Score   : 52.1%
   Answer  : 💧 Yes! PhoneX Pro is IP68 rated, meaning it's waterproof...

✅ Query   : Tell me about battery life
   Matched : What is the battery capacity?
   Score   : 61.3%
   Answer  : 🔋 PhoneX Pro packs a massive 5000mAh battery...
```

---

## 🔮 Future Improvements

- [ ] Add more FAQ categories (shipping, repairs, accessories)
- [ ] Integrate **spaCy** for better NLP preprocessing
- [ ] Add **intent classification** using a lightweight ML model
- [ ] Export chat history to PDF or text file
- [ ] Deploy as a **Streamlit web app**
- [ ] Support multiple languages

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

## 👨‍💻 Author

Made with ❤️ using Python & NLP

> ⭐ If you found this project helpful, please give it a star on GitHub!

---

<div align="center">

**[⬆ Back to Top](#-faq-chatbot--nlp-powered-with-cosine-similarity)**

</div>
