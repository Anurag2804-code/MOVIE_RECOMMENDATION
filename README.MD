🎬 Movie Recommendation System
A beginner-level Machine Learning project that recommends similar movies based on content features (genre, director, keywords, description) using TF-IDF Vectorization and Cosine Similarity, with an additional K-Means Clustering layer to group movies by theme.

📌 Problem Statement
Streaming platforms recommend movies all the time — but how do they actually work?
This project builds a simple content-based recommendation engine that, given a movie you like, finds the most similar movies from a dataset of 51 popular films.

🚀 Features
🔍 Content-Based Recommendations — Uses TF-IDF + Cosine Similarity
🎯 Top-N Results — Returns top 5 most similar movies with similarity scores
🗂️ K-Means Clustering — Groups movies into 6 thematic clusters automatically
📊 4 Visualizations — Genre bar chart, similarity heatmap, recommendation bar chart, cluster pie chart
🎞️ 50+ Movie Database — Hand-crafted metadata across 12 genres
🛠️ Tech Stack
Tool	Purpose
Python 3.x	Core language
pandas	Data handling
numpy	Numerical operations
scikit-learn	TF-IDF, Cosine Similarity, K-Means
matplotlib	Plotting
seaborn	Statistical visualizations
📁 Project Structure
movie_recommender/
│
├── movie_recommender.py     # Main script (data + models + recommendations + plots)
├── plots/                   # Auto-generated output plots
│   ├── 01_genre_distribution.png
│   ├── 02_similarity_heatmap.png
│   ├── 03_inception_recommendations.png
│   └── 04_cluster_distribution.png
└── README.md
⚙️ Setup & Installation
1. Clone the Repository
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
2. Install Dependencies
pip install pandas numpy scikit-learn matplotlib seaborn
3. Run the Project
python movie_recommender.py
The script will:

Load 51 movies with metadata
Build the TF-IDF matrix and compute cosine similarities
Print recommendations for 5 sample movies
Show K-Means clustering results
Save all plots to the plots/ folder
🔍 How Recommendations Work
Movie → Extract Features (genre + director + keywords + description)
      → TF-IDF Vectorization (convert text to numbers)
      → Cosine Similarity (find closest movies)
      → Return Top-N Matches
Cosine similarity measures the angle between two feature vectors. A score closer to 1.0 means very similar; closer to 0.0 means very different.

🎯 Sample Output
🎬 Because you liked → Inception
 Rank   Movie              Genre          Similarity
    1   Interstellar       Sci-Fi Drama   0.1079
    2   The Dark Knight    Action Thriller 0.0942
    3   Dunkirk            War Action     0.0676
    4   The Matrix         Sci-Fi Action  0.0658
    5   La La Land         Romance Musical 0.0558

🎬 Because you liked → Toy Story
 Rank   Movie          Genre              Similarity
    1   Finding Nemo   Animation Family   0.1181
    2   Coco           Animation Family   0.0760
    3   The Lion King  Animation Family   0.0751
    4   Up             Animation Adventure 0.0732
    5   Shrek          Animation Comedy   0.0427
🗂️ K-Means Clusters (6 Groups)
Cluster	Dominant Theme	Example Movies
0	Dark Thrillers	Parasite, Get Out, IT
1	Family/Adventure	Toy Story, Harry Potter, Up
2	Sci-Fi/Survival	Inception, Interstellar, The Martian
3	Crime/Drama	The Godfather, Pulp Fiction, Joker
4	Comedy	The Hangover, Home Alone, Shrek
5	Action/War	The Dark Knight, 1917, Gladiator
📈 Technical Details
Component	Details
Vectorizer	TF-IDF with 1-2 word n-grams, English stopwords removed
Matrix shape	51 movies × 1096 vocabulary terms
Similarity metric	Cosine Similarity
Clustering	K-Means (k=6, random_state=42)
🧠 Concepts Used (from Course)
Natural Language Processing (TF-IDF)
Similarity Metrics (Cosine Similarity)
Unsupervised Learning (K-Means Clustering)
Feature Extraction from text
Content-Based Filtering
💡 Limitations & Future Improvements
Uses a small manually created dataset (can be extended with TMDB or MovieLens datasets)
Content-based only — no user behavior/collaborative filtering
Future: Add a web interface using Streamlit
👤 Author
RANVEER SINGH KUSHWAH
Registration number: 25BCE10194 Course: Fundamentals of AI and ML
Platform: VITyarthi

📄 License
This project is created for academic purposes as part of the BYOP (Bring Your Own Project) assignment.
