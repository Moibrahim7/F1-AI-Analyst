# F1-AI-Analyst

---

# 🏎️ F1 AI Assistant (RAG-based Formula 1 QA System)

An AI-powered Formula 1 assistant built using **Retrieval-Augmented Generation (RAG)**.
This project allows users to ask natural language questions about F1 race data and receive accurate answers sourced directly from a structured dataset.

---

## 🚀 Features

* 📊 Query F1 race data using natural language
* 🤖 RAG architecture (Vector DB + LLM)
* ⚡ Fast semantic search with embeddings
* 🎨 Custom F1-themed animated UI
* 📈 Accuracy evaluation included
* 🧠 Works fully offline after indexing

---

## 🧠 Tech Stack

* **Python**
* **LangChain**
* **ChromaDB (Vector Database)**
* **OpenAI / Local LLM**
* **Pandas**
* **HTML + CSS (Animated UI)**

---

## 📂 Dataset

* Format: CSV
* Example fields:

  * Race Name
  * Date
  * Winner
  * Team
  * Fastest Lap
  * Points
  * Grid Position

**Example file path:**

```
d:\races.csv
```

---

## ⚙️ How It Works

1. Load F1 dataset (CSV)
2. Convert rows into text documents
3. Generate embeddings
4. Store in Chroma vector DB
5. User asks question
6. Relevant context retrieved
7. LLM generates grounded answer

---

## 🎨 UI Features

* 🏁 Racing theme
* 🚗 Animated F1 car
* 🏴 Waving checkered flag
* Smooth transitions
* Dark racing UI

---

## 🧪 Accuracy Testing

Manual evaluation using dataset-grounded questions.

### Sample Questions

* Who won the Australia race?
* Who finished second?
* What was the fastest lap?
* How many laps were completed?
* Which team won?

---

## 📊 Accuracy Results

| Metric          | Score   |
| --------------- | ------- |
| Correct Answers | 9.5 / 10  |
| Accuracy        | **95%** |

### Observations

* Strong on direct factual queries
* Weak on missing dataset fields
* No hallucinations detected

---

## 📄 Project Report

A full evaluation report is included:

```
F1_AI_Accuracy_Report.docx
```

Contains:

* Methodology
* Results
* Analysis
* Improvements

---

## 🔮 Future Improvements

* Real-time F1 API integration
* Multi-race querying
* Driver career stats
* Voice assistant support
* Leaderboards visualization
* Fine-tuned local model

---

## 📚 Learning Outcomes

* RAG pipeline implementation
* Vector databases
* Prompt engineering
* LLM evaluation
* AI UI design

---

## 🤝 Contributing

Pull requests are welcome!
For major changes, open an issue first.

---

## 🙌 Acknowledgements

* LangChain community
* ChromaDB team
* Formula 1 open data sources

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!

---
