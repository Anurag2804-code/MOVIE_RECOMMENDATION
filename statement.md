# Problem Statement
 
## Project Title
**Movie Recommendation System — Content-Based Filtering using TF-IDF, Cosine Similarity, and K-Means Clustering**
 
## Course
Fundamentals of AI and ML
 
## Author
ANURAG TIWARY
Reg. no. 25MIM10209
 
---
 
## 1. Problem Statement
 
With the explosion of digital streaming content, users are increasingly overwhelmed by the sheer volume
of movies available to them. Deciding "what to watch next" has become a non-trivial problem, and most
users end up relying on opaque, black-box recommendation engines built into streaming platforms such
as Netflix, Amazon Prime, and Spotify.
 
The goal of this project is to design and implement a **transparent, explainable, content-based movie
recommendation system** that can suggest movies similar to one a user has already watched and enjoyed —
without relying on user ratings, watch history, or collaborative signals from other users.
 
## 2. Motivation
 
Most commercial recommenders are collaborative-filtering systems that depend on large volumes of user
interaction data, which are unavailable in an academic setting. A content-based approach, built entirely
from item metadata (genre, director, keywords, description), allows the entire pipeline — from feature
extraction to similarity computation — to be visible, interpretable, and reproducible from scratch. This
makes it an ideal exercise for understanding core NLP and unsupervised learning concepts taught in this
course.
 
## 3. Objectives
 
- Build a content-based recommender using **TF-IDF Vectorization** and **Cosine Similarity**.
- Given any input movie title, return the **top 5 most similar movies** from the dataset.
- Apply **K-Means Clustering** on the TF-IDF feature space to automatically discover thematic
  groupings among movies, demonstrating unsupervised learning in practice.
- Visualize the similarity structure of the dataset using **heatmaps** and **bar charts**.
- Evaluate whether the discovered clusters and recommendations align with human intuition about
  genre and thematic similarity.
## 4. Scope of the Project
 
| In Scope | Out of Scope |
|---|---|
| Content-based filtering using movie metadata | Collaborative filtering using user ratings |
| A manually curated dataset of 51 movies | Integration with live movie APIs (e.g., TMDB) |
| TF-IDF + Cosine Similarity recommendation engine | Real-time deployment / web interface |
| K-Means clustering for thematic grouping | Deep learning–based recommenders |
| Static visualizations (heatmaps, bar charts) | Personalized, user-specific recommendations |
 
## 5. Dataset Description
 
A custom dataset of **51 popular movies** was manually created with the following metadata fields:
 
| Field | Description | Example |
|---|---|---|
| `title` | Movie name | Inception |
| `genre` | Primary genre(s) | Sci-Fi Thriller |
| `director` | Director's name | Christopher Nolan |
| `keywords` | Key thematic words | dream heist memory reality |
| `description` | One-sentence plot summary | A thief enters dreams... |
 
All five fields are concatenated into a single `combined` text field per movie, which serves as the
input to the TF-IDF vectorizer.
 
## 6. Proposed Approach
 
1. **Feature Extraction** — Convert each movie's combined text field into a numerical vector using
   TF-IDF (`ngram_range=(1,2)`, English stop words removed).
2. **Similarity Computation** — Compute a 51×51 cosine similarity matrix across all movies.
3. **Recommendation Logic** — For a given input movie, rank all other movies by similarity score and
   return the top N (default 5) matches.
4. **Clustering** — Apply K-Means (k=6) on the TF-IDF matrix to group movies into thematic clusters.
5. **Evaluation & Visualization** — Validate recommendations qualitatively (e.g., genre consistency)
   and visualize similarity structure and cluster composition.
## 7. Expected Outcomes
 
- A working recommendation function that returns logically consistent, genre-aligned movie
  suggestions for any input movie in the dataset.
- Six interpretable movie clusters (e.g., Dark Thrillers, Family/Adventure, Sci-Fi/Survival,
  Crime/Drama, Comedy, Action/War) that align with human intuition.
- Visual artifacts (similarity heatmap, cluster bar chart) that make the model's internal structure
  understandable.
## 8. Tools and Technologies
 
| Component | Technology |
|---|---|
| Programming Language | Python 3.x |
| Vectorization | `scikit-learn` (TfidfVectorizer) |
| Similarity Metric | `scikit-learn` (cosine_similarity) |
| Clustering | `scikit-learn` (KMeans) |
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
 
## 9. Known Limitations
 
- **Cold start problem** — the system cannot recommend movies to a new user with no watch history,
  nor recommend movies outside the dataset.
- **Small dataset size** — with only 51 movies and short combined text features, cosine similarity
  scores remain low (typically 0.10–0.35), reducing discriminative power.
- **Manual k selection** — the number of clusters (k=6) was chosen using domain knowledge rather than
  an objective method such as the Elbow Method.
 
