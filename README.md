# 🚀 ResumeRank AI — Intelligent Resume Ranking & Career Matching System

ResumeRank AI is an NLP-powered resume intelligence project that helps recruiters quickly identify the best candidate profiles for a given job description. Instead of manually reading hundreds of resumes, the system cleans resume text, discovers career themes, classifies resumes into job categories, and ranks the most relevant candidates using cosine similarity.

This project turns a raw resume dataset into a practical AI screening workflow.

---

## 🌟 Project Story

Hiring teams often receive hundreds or thousands of resumes for a single role. The challenge is not just finding resumes with matching keywords, but understanding skills, professional background, role category, and overall relevance.

This notebook builds a lightweight resume-ranking engine that can:

- Analyze thousands of resumes automatically
- Group similar resumes into clusters
- Discover hidden resume topics
- Classify resumes into professional categories
- Rank resumes against a custom job description
- Visualize resume patterns and skill distribution

The goal is to create a practical AI assistant for recruiters, career platforms, and HR analytics teams.

---

## 🧠 What This Project Does

### 1. Resume Text Cleaning

The notebook processes raw resume text by:

- Lowercasing text
- Removing punctuation and non-alphabetic characters
- Tokenizing resume content
- Removing English stopwords
- Removing resume-specific noise words such as `resume`, `contact`, `email`, `phone`, `objective`, `summary`, and `experience`

This creates cleaner text for NLP modeling.

---

### 2. TF-IDF Feature Engineering

The project converts resume text into numerical vectors using **TF-IDF**.

Configuration used:

```python
TfidfVectorizer(
    max_features=1000,
    stop_words="english",
    ngram_range=(1,1),
    min_df=5,
    max_df=0.7
)
```

TF-IDF helps identify words that are important in a resume but not too common across all resumes.

---

### 3. Resume Clustering

The notebook uses **K-Means Clustering** to group similar resumes.

```python
KMeans(n_clusters=8, random_state=42)
```

This helps reveal natural groups of candidates based on resume language and skills.

---

### 4. Topic Modeling

The project uses **Latent Dirichlet Allocation (LDA)** to discover hidden themes across resumes.

Example topic themes found include:

- Education and student programs
- Engineering, production, and quality control
- Data, systems, software, and IT
- Finance, accounting, and reporting
- Healthcare and professional services

This gives recruiters a high-level view of what types of profiles exist in the resume pool.

---

### 5. Resume Classification

The notebook compares machine learning models for resume category prediction.

Models used:

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline classification model |
| MLP Neural Network | Deep learning-style classifier |
| TF-IDF + ML Pipeline | Text-to-category prediction |

Current results from the notebook:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 64.79% |
| MLP Classifier | 56.14% |

Logistic Regression performed better in this version, showing that strong text features can outperform a more complex neural network on sparse TF-IDF data.

---

### 6. Resume Ranking Against Job Description

The most practical part of the project is the resume ranking system.

A sample job description is converted into a TF-IDF vector:

```python
job_description = """
Looking for a Data Analyst with Python, SQL,
data visualization, statistics, and machine learning experience.
"""
```

Then the system calculates cosine similarity between the job description and every resume.

The output is a ranked table of the most relevant resumes:

| Rank | Resume Category | Similarity Score |
|---:|---|---:|
| 1 | Consultant | 0.40 |
| 2 | Engineering | 0.35 |
| 3 | Automobile | 0.28 |
| 4 | Agriculture | 0.28 |
| 5 | Digital Media | 0.24 |

This ranking engine can be extended into a recruiter dashboard or applicant tracking system.

---

## 📊 Visualizations Included

The notebook includes several visual insights:

- Resume category distribution
- TF-IDF word importance heatmap
- Resume cluster visualization using t-SNE
- Confusion matrix for model performance
- Word cloud of resume vocabulary
- Skill heatmap by resume category

These visuals help explain the model to both technical and non-technical stakeholders.

---

## 🛠️ Tech Stack

| Area | Tools Used |
|---|---|
| Programming | Python |
| Data Handling | Pandas, NumPy |
| NLP | NLTK, TF-IDF, CountVectorizer |
| Machine Learning | Scikit-learn |
| Topic Modeling | Latent Dirichlet Allocation |
| Clustering | K-Means |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Similarity Search | Cosine Similarity |
| Notebook Platform | Google Colab |

---

## 📁 Dataset

The notebook uses a resume dataset with:

- **2,484 resumes**
- **2 main columns**
  - `Resume_str`
  - `Category`

The dataset contains resumes across multiple professional categories such as HR, IT, Accounting, Healthcare, Engineering, Sales, Design, and more.

---

## 📌 Project Workflow

```text
Raw Resume Dataset
        ↓
Text Cleaning & Stopword Removal
        ↓
TF-IDF Vectorization
        ↓
Exploratory Data Analysis
        ↓
K-Means Resume Clustering
        ↓
LDA Topic Modeling
        ↓
Classification Model Training
        ↓
Resume Ranking with Cosine Similarity
        ↓
Top Candidate Recommendations
```

---

## ▶️ How to Run This Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/resume-rank-ai.git
cd resume-rank-ai
```

### 2. Install required packages

```bash
pip install pandas numpy nltk matplotlib seaborn scikit-learn gensim wordcloud
```

### 3. Download NLTK resources

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

### 4. Add the dataset

Update the dataset path in the notebook:

```python
csv_path = "Resume.csv"
```

If using Google Colab, upload the file to Google Drive and update the path accordingly.

### 5. Run the notebook

Open and run:

```text
Resume_Ranking.ipynb
```

---

## 💡 Business Use Case

ResumeRank AI can be used by:

- Recruiters screening large applicant pools
- HR teams building internal talent intelligence tools
- Job platforms improving candidate-job matching
- Career coaches identifying resume strengths
- Startups building AI hiring products

Instead of replacing recruiters, this system supports them by reducing manual screening time and surfacing stronger matches faster.

---

## 🚀 Future Improvements

This project can be extended with:

- Streamlit recruiter dashboard
- PDF resume upload support
- Named Entity Recognition for skills, companies, degrees, and certifications
- Better job-resume matching using Sentence-BERT embeddings
- Skill gap analysis
- Resume score explanation
- Ranking filters by experience, skill, category, and education
- Supabase or PostgreSQL backend for storing resumes and search results
- API endpoint for production deployment

---

## 🔮 Possible Product Version

A future version of this project could become:

> **An AI recruiter assistant that reads resumes, understands candidate strengths, ranks applicants for a job description, and explains why each candidate is a good fit.**

Example output:

```text
Candidate Rank: #1
Match Score: 82%
Reason: Strong Python, SQL, analytics, and reporting experience.
Suggested Role Fit: Data Analyst
Missing Skills: Tableau, cloud analytics
```

---

## 🧩 Key Learning Outcomes

Through this project, I practiced:

- Text preprocessing for real-world resume data
- TF-IDF feature extraction
- Unsupervised clustering
- Topic modeling
- Resume classification
- Cosine similarity ranking
- Model evaluation using accuracy, confusion matrix, and classification reports
- Translating an NLP notebook into a practical business product idea

---

## 👨‍💻 Author

**Rishabh Gupta**  
---

## ⭐ Why This Project Matters

This project is more than a resume classifier. It demonstrates how NLP can be used to solve a real hiring problem by combining text analytics, machine learning, clustering, topic modeling, and similarity search into one intelligent workflow.

ResumeRank AI shows how a simple notebook can evolve into a useful AI product for recruitment and talent analytics.
# Resume_Ranking
