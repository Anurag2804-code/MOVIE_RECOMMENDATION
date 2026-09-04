
# 🎬 Movie Recommendation System

A beginner-level **Machine Learning project** that recommends similar movies based on their content features such as **genre, director, keywords, and description**.

The system uses **TF-IDF Vectorization** and **Cosine Similarity** to find movies with similar content. An additional **K-Means Clustering** layer groups movies into thematic categories.

---

## 📌 Problem Statement

Streaming platforms recommend movies based on what users have watched or liked.

This project demonstrates a simple **Content-Based Movie Recommendation System**. Given the name of a movie, the system analyzes its content features and recommends the **Top 5 most similar movies** from a dataset of 51 movies.

---

## 🚀 Features

* 🔍 **Content-Based Recommendations**

  * Uses TF-IDF Vectorization
  * Uses Cosine Similarity

* 🎯 **Top-N Recommendations**

  * Returns the top 5 similar movies
  * Displays similarity scores

* 🗂️ **K-Means Clustering**

  * Groups movies into 6 thematic clusters
  * Uses unsupervised learning

* 📊 **Data Visualizations**

  * Genre distribution
  * Similarity heatmap
  * Movie recommendation chart
  * Cluster distribution

* 🎞️ **Movie Dataset**

  * 51 popular movies
  * 12+ genres
  * Manually prepared metadata

---

## 🛠️ Tech Stack

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python 3.x   | Core programming language      |
| Pandas       | Data handling and manipulation |
| NumPy        | Numerical operations           |
| Scikit-learn | Machine Learning algorithms    |
| Matplotlib   | Data visualization             |
| Seaborn      | Statistical visualization      |

---

## 🧠 Machine Learning Concepts Used

This project demonstrates several important concepts from **Artificial Intelligence and Machine Learning**:

* Natural Language Processing (NLP)
* Feature Extraction
* TF-IDF Vectorization
* Cosine Similarity
* Content-Based Filtering
* Unsupervised Learning
* K-Means Clustering
* Data Visualization

---

## 🔍 How the Recommendation System Works

The recommendation pipeline works as follows:

```text
                 🎬 Movie
                    │
                    ▼
        ┌───────────────────────┐
        │  Extract Movie Data   │
        │                       │
        │ Genre                 │
        │ Director              │
        │ Keywords              │
        │ Description           │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   TF-IDF Vectorizer   │
        │                       │
        │ Text → Numerical      │
        │ Feature Vectors       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Cosine Similarity    │
        │                       │
        │ Compare Movie Vectors │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   Sort Similarities   │
        │       ↓               │
        │   Top 5 Movies        │
        └───────────────────────┘
```

---

## 📐 TF-IDF Vectorization

**TF-IDF** stands for **Term Frequency-Inverse Document Frequency**.

It converts text information into numerical vectors that can be processed by machine learning algorithms.

The project combines the following movie information:

```text
Genre + Director + Keywords + Description
```

These features are converted into TF-IDF vectors.

The vectorizer uses:

* 1–2 word n-grams
* English stopwords removal
* TF-IDF weighting

---

## 📏 Cosine Similarity

Cosine Similarity measures how similar two movie vectors are.

The formula is:

```text
                    A · B
Cosine Similarity = ───────
                   ||A|| ||B||
```

The similarity score generally ranges from:

```text
0 → Completely different
1 → Highly similar
```

A higher score indicates that two movies have more similar content.

---

## 🗂️ K-Means Clustering

The project also uses **K-Means Clustering** to group movies based on their content.

The number of clusters is:

```text
K = 6
```

The model uses:

```python
KMeans(n_clusters=6, random_state=42)
```

### Example Clusters

| Cluster | Dominant Theme     | Example Movies                       |
| ------- | ------------------ | ------------------------------------ |
| 0       | Dark Thrillers     | Parasite, Get Out, IT                |
| 1       | Family / Adventure | Toy Story, Harry Potter, Up          |
| 2       | Sci-Fi / Survival  | Inception, Interstellar, The Martian |
| 3       | Crime / Drama      | The Godfather, Pulp Fiction, Joker   |
| 4       | Comedy             | The Hangover, Home Alone, Shrek      |
| 5       | Action / War       | The Dark Knight, 1917, Gladiator     |

> **Note:** Cluster labels are generated automatically by K-Means. The dominant themes above are interpretations based on the movies belonging to each cluster.

---

## 📊 Dataset

The project contains metadata for **51 movies**.

Each movie contains information such as:

```text
Movie Name
Genre
Director
Keywords
Description
```

The dataset covers more than **12 different genres**, including:

* Action
* Adventure
* Animation
* Comedy
* Crime
* Drama
* Family
* Horror
* Romance
* Sci-Fi
* Thriller
* War

---

## 🎯 Sample Recommendation

### 🎬 Because you liked → Inception

| Rank | Movie           | Genre           | Similarity |
| ---: | --------------- | --------------- | ---------: |
|    1 | Interstellar    | Sci-Fi Drama    |     0.1079 |
|    2 | The Dark Knight | Action Thriller |     0.0942 |
|    3 | Dunkirk         | War Action      |     0.0676 |
|    4 | The Matrix      | Sci-Fi Action   |     0.0658 |
|    5 | La La Land      | Romance Musical |     0.0558 |

---

### 🎬 Because you liked → Toy Story

| Rank | Movie         | Genre               | Similarity |
| ---: | ------------- | ------------------- | ---------: |
|    1 | Finding Nemo  | Animation Family    |     0.1181 |
|    2 | Coco          | Animation Family    |     0.0760 |
|    3 | The Lion King | Animation Family    |     0.0751 |
|    4 | Up            | Animation Adventure |     0.0732 |
|    5 | Shrek         | Animation Comedy    |     0.0427 |

---

## 📈 Visualizations

The project automatically generates four visualizations.

### 1. Genre Distribution

Displays the number of movies belonging to each genre.

```text
plots/
└── 01_genre_distribution.png
```

### 2. Similarity Heatmap

Shows the similarity between different movies using a heatmap.

```text
plots/
└── 02_similarity_heatmap.png
```

### 3. Recommendation Chart

Displays the recommended movies and their similarity scores.

```text
plots/
└── 03_inception_recommendations.png
```

### 4. Cluster Distribution

Shows the distribution of movies across the six K-Means clusters.

```text
plots/
└── 04_cluster_distribution.png
```

---

## 📁 Project Structure

```text
movie-recommender/
│
├── movie_recommender.py
├── README.md
├── requirements.txt
│
└── plots/
    ├── 01_genre_distribution.png
    ├── 02_similarity_heatmap.png
    ├── 03_inception_recommendations.png
    └── 04_cluster_distribution.png
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/movie-recommendation-system.git
```

Navigate to the project directory:

```bash
cd movie-recommendation-system
```

---

### 2. Install Dependencies

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Or, if a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Run the Python script:

```bash
python movie_recommender.py
```

If your system uses `python3`:

```bash
python3 movie_recommender.py
```

The program will:

1. Load the movie dataset.
2. Combine movie content features.
3. Create the TF-IDF matrix.
4. Calculate Cosine Similarity.
5. Generate Top-5 movie recommendations.
6. Perform K-Means clustering.
7. Display clustering results.
8. Generate and save visualizations.

---

## 📊 Technical Details

| Component            | Details              |
| -------------------- | -------------------- |
| Dataset Size         | 51 Movies            |
| Feature Type         | Text Metadata        |
| Vectorizer           | TF-IDF               |
| N-Grams              | 1–2                  |
| Stopwords            | English              |
| Similarity Metric    | Cosine Similarity    |
| Clustering Algorithm | K-Means              |
| Number of Clusters   | 6                    |
| Random State         | 42                   |
| Visualization        | Matplotlib + Seaborn |

The resulting TF-IDF matrix contains approximately:

```text
51 Movies × 1096 Features
```

---

## 💡 Why Content-Based Filtering?

A content-based recommendation system recommends items based on their characteristics.

For example, if a user likes:

```text
Inception
```

the system analyzes its:

```text
Genre
Director
Keywords
Description
```

It then searches for movies with similar characteristics.

This means the system does **not require information about other users**.

---

## 🔄 Recommendation Workflow

```text
User selects a movie
        ↓
Movie metadata is retrieved
        ↓
Genre + Director + Keywords + Description
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity
        ↓
Similarity scores calculated
        ↓
Movies sorted by similarity
        ↓
Top 5 recommendations
```

---

## 🧩 Clustering Workflow

```text
Movie Feature Vectors
        ↓
TF-IDF Matrix
        ↓
K-Means Algorithm
        ↓
6 Clusters
        ↓
Movies grouped by similarity
        ↓
Analyze dominant themes
```

---

## ⚠️ Limitations

Although this project demonstrates the basic working of a recommendation engine, it has some limitations:

* The dataset contains only 51 movies.
* Movie metadata is manually created.
* It uses only content-based filtering.
* It does not consider user ratings.
* It does not learn from user viewing history.
* Recommendations depend heavily on the quality of the metadata.
* K-Means cluster labels do not inherently represent meaningful movie genres.

---

## 🚀 Future Improvements

The project can be improved by:

### 📚 Larger Dataset

Use real-world datasets such as:

* TMDB
* MovieLens
* IMDb datasets

### 👥 Collaborative Filtering

Use user ratings and viewing history to implement:

```text
User-Based Collaborative Filtering
Item-Based Collaborative Filtering
```

### 🤖 Hybrid Recommendation System

Combine:

```text
Content-Based Filtering
+
Collaborative Filtering
```

to create a more powerful recommendation engine.

### 🌐 Web Interface

Build an interactive web application using:

```text
Streamlit
```

Users could search for a movie and instantly receive recommendations.

### 🎯 Advanced NLP

Future versions could use:

* Word embeddings
* Word2Vec
* BERT
* Sentence Transformers

for better semantic understanding.

---

## 🎓 Academic Information

**Author:** ANURAG TIWARY

**Registration Number:** 25MIM10209

**Course:** Fundamentals of AI and ML

**Platform:** VITyarthi

**Project Type:** BYOP — Bring Your Own Project

---

## 📚 Learning Outcomes

Through this project, the following concepts are demonstrated:

* Understanding text preprocessing
* Converting text into numerical features
* Implementing TF-IDF
* Calculating Cosine Similarity
* Building a content-based recommendation system
* Implementing K-Means clustering
* Working with Pandas and NumPy
* Creating data visualizations
* Understanding unsupervised learning
* Applying Machine Learning concepts to a real-world problem

---

## 📄 License

This project is created for **academic and educational purposes** as part of the BYOP assignment.

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub!
