#  Movie Recommendation System
#  Author : RANVEER SINGH KUSHWAH

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import warnings, os
warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)


#DATASET (50 movies, hand-crafted metadata)

movies_data = [
    # title, genre, director, cast_keywords, description
    ("The Dark Knight",    "Action Thriller",  "Christopher Nolan",
     "batman joker gotham hero",
     "A masked vigilante faces Gotham's greatest villain the chaotic Joker in an epic battle of wills"),
    ("Inception",          "Sci-Fi Thriller",  "Christopher Nolan",
     "dream heist memory reality",
     "A thief enters dreams to plant an idea using mind-bending technology"),
    ("Interstellar",       "Sci-Fi Drama",     "Christopher Nolan",
     "space time wormhole astronaut",
     "Astronauts travel through a wormhole to find a new home for humanity"),
    ("The Matrix",         "Sci-Fi Action",    "Lana Wachowski",
     "simulation hacker neo rebel",
     "A hacker discovers reality is a simulation and joins a rebellion"),
    ("Avengers Endgame",   "Action Adventure", "Anthony Russo",
     "superheroes thanos iron man captain",
     "Superheroes assemble to undo the damage caused by Thanos"),
    ("Spider-Man",         "Action Adventure", "Sam Raimi",
     "spiderman hero new york villain",
     "A teenager bitten by a spider gains powers and fights crime"),
    ("Parasite",           "Thriller Drama",   "Bong Joon-ho",
     "class wealth family deceit",
     "A poor family schemes to work for a wealthy household"),
    ("Joker",              "Crime Drama",      "Todd Phillips",
     "joker crime batman gotham clown",
     "A failed comedian descends into madness and becomes the Joker"),
    ("The Godfather",      "Crime Drama",      "Francis Coppola",
     "mafia family crime power loyalty",
     "A powerful mafia family's patriarch transfers control to his reluctant son"),
    ("Goodfellas",         "Crime Drama",      "Martin Scorsese",
     "mafia crime gang loyalty betrayal",
     "The rise and fall of a mob associate in the Lucchese crime family"),
    ("Pulp Fiction",       "Crime Thriller",   "Quentin Tarantino",
     "hitman crime nonlinear story",
     "Intertwining tales of crime in Los Angeles told out of sequence"),
    ("Fight Club",         "Thriller Drama",   "David Fincher",
     "identity rebellion underground violence",
     "A bored office worker forms an underground fight club"),
    ("Forrest Gump",       "Drama Romance",    "Robert Zemeckis",
     "life journey love history war",
     "A man with low IQ witnesses major American historical events"),
    ("Titanic",            "Romance Drama",    "James Cameron",
     "love ship tragedy ocean class",
     "Two people fall in love aboard a doomed ocean liner"),
    ("La La Land",         "Romance Musical",  "Damien Chazelle",
     "love music jazz dreams aspiration",
     "A musician and actress fall in love while chasing their dreams"),
    ("The Notebook",       "Romance Drama",    "Nick Cassavetes",
     "love memory romance elderly",
     "An old man reads a love story to a woman with dementia"),
    ("Toy Story",          "Animation Family", "John Lasseter",
     "toys friendship adventure animation",
     "Toys come alive when humans leave and go on adventures"),
    ("Finding Nemo",       "Animation Family", "Andrew Stanton",
     "ocean fish family adventure friendship",
     "A clownfish searches the ocean for his missing son"),
    ("The Lion King",      "Animation Family", "Roger Allers",
     "lion kingdom africa family betrayal",
     "A lion cub must reclaim the throne from his treacherous uncle"),
    ("Shrek",              "Animation Comedy", "Andrew Adamson",
     "ogre fairytale comedy adventure",
     "A grumpy ogre must rescue a princess to save his swamp"),
    ("The Hangover",       "Comedy",           "Todd Phillips",
     "friends comedy vegas bachelor party",
     "Friends wake up with no memory after a wild bachelor party"),
    ("Superbad",           "Comedy",           "Greg Mottola",
     "teenagers comedy high school party",
     "High school friends try to get alcohol for a party"),
    ("Home Alone",         "Comedy Family",    "Chris Columbus",
     "kid comedy christmas burglars",
     "A boy defends his home against burglars during Christmas"),
    ("Rush Hour",          "Action Comedy",    "Brett Ratner",
     "buddy comedy cop action",
     "A Hong Kong detective teams with an LAPD cop"),
    ("Get Out",            "Horror Thriller",  "Jordan Peele",
     "horror race thriller social",
     "A Black man visits his girlfriend's white family and discovers dark secrets"),
    ("IT",                 "Horror",           "Andy Muschietti",
     "horror clown children fear",
     "Children are terrorised by a shape-shifting entity in the form of a clown"),
    ("A Quiet Place",      "Horror Sci-Fi",    "John Krasinski",
     "silence survival horror monster",
     "A family lives in silence to avoid sound-hunting creatures"),
    ("Us",                 "Horror Thriller",  "Jordan Peele",
     "doppelganger horror family thriller",
     "A family is terrorised by their sinister doubles"),
    ("Harry Potter",       "Fantasy Adventure","Chris Columbus",
     "wizard magic school friendship",
     "A young wizard discovers his identity and battles dark forces"),
    ("Lord of the Rings",  "Fantasy Adventure","Peter Jackson",
     "ring quest fellowship elf dwarf",
     "A fellowship embarks on a quest to destroy a powerful ring"),
    ("The Hobbit",         "Fantasy Adventure","Peter Jackson",
     "hobbit dragon adventure quest",
     "A hobbit goes on an unexpected quest with dwarves and a wizard"),
    ("Doctor Strange",     "Fantasy Action",   "Scott Derrickson",
     "wizard magic multiverse superhero",
     "A surgeon becomes a powerful sorcerer after a career-ending accident"),
    ("Dune",               "Sci-Fi Adventure", "Denis Villeneuve",
     "desert planet politics power spice",
     "A young man navigates political turmoil on a desert planet"),
    ("Blade Runner 2049",  "Sci-Fi Drama",     "Denis Villeneuve",
     "replicant future dystopia identity",
     "A replicant hunter uncovers a secret that threatens civilisation"),
    ("Arrival",            "Sci-Fi Drama",     "Denis Villeneuve",
     "alien language time perception",
     "A linguist attempts to communicate with alien visitors"),
    ("Whiplash",           "Drama Music",      "Damien Chazelle",
     "music ambition jazz drums obsession",
     "A young drummer is pushed to his limits by a demanding instructor"),
    ("1917",               "War Drama",        "Sam Mendes",
     "war world war mission soldier",
     "Two soldiers must cross enemy territory to deliver a vital message"),
    ("Dunkirk",            "War Action",       "Christopher Nolan",
     "war rescue beach soldier navy",
     "Allied soldiers are evacuated from the beaches of Dunkirk"),
    ("Saving Private Ryan","War Drama",        "Steven Spielberg",
     "war soldier mission wwii",
     "A group of soldiers search for a paratrooper behind enemy lines"),
    ("Gladiator",          "Action Drama",     "Ridley Scott",
     "roman warrior revenge arena",
     "A Roman general seeks revenge against the emperor who murdered his family"),
    ("Braveheart",         "Action Drama",     "Mel Gibson",
     "scotland warrior freedom battle",
     "A Scottish warrior leads a revolt against English rule"),
    ("300",                "Action Drama",     "Zack Snyder",
     "sparta battle war warrior",
     "300 Spartan warriors face a massive Persian army"),
    ("The Revenant",       "Adventure Drama",  "Alejandro Inarritu",
     "survival wilderness revenge nature",
     "A frontiersman survives brutal conditions to seek revenge"),
    ("Into the Wild",      "Adventure Drama",  "Sean Penn",
     "journey wilderness nature freedom",
     "A young man abandons society and travels to Alaska"),
    ("Cast Away",          "Drama Survival",   "Robert Zemeckis",
     "island survival alone ocean",
     "A FedEx employee is stranded alone on an uninhabited island"),
    ("The Martian",        "Sci-Fi Survival",  "Ridley Scott",
     "mars survival astronaut science",
     "An astronaut is stranded on Mars and must survive alone"),
    ("Gravity",            "Sci-Fi Drama",     "Alfonso Cuarón",
     "space survival astronaut debris",
     "Two astronauts try to survive after their shuttle is destroyed"),
    ("Soul",               "Animation Drama",  "Pete Docter",
     "music soul purpose life jazz",
     "A musician has an out-of-body experience and discovers the meaning of life"),
    ("Inside Out",         "Animation Drama",  "Pete Docter",
     "emotions mind childhood memory",
     "A girl's emotions help her navigate life after a big move"),
    ("Up",                 "Animation Adventure","Pete Docter",
     "adventure balloon journey friendship old",
     "An old man ties thousands of balloons to his house and flies away"),
    ("Coco",               "Animation Family", "Lee Unkrich",
     "music family mexico death heritage",
     "A boy journeys to the land of the dead to meet his musician ancestor"),
]

columns = ["title","genre","director","keywords","description"]
df = pd.DataFrame(movies_data, columns=columns)

# Combined feature for TF-IDF
df["combined"] = (df["genre"] + " " + df["director"] + " " +
                  df["keywords"] + " " + df["description"])

print("=" * 60)
print("  MOVIE RECOMMENDATION SYSTEM")
print("=" * 60)
print(f"\nTotal movies in database : {len(df)}")
print("\nGenre distribution :")
genre_main = df["genre"].str.split().str[0]
print(genre_main.value_counts().to_string())

#2.TF-IDF + COSINE SIMILARITY

tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df["combined"])
cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(f"\nTF-IDF matrix shape : {tfidf_matrix.shape}")
print(f"Vocabulary size     : {len(tfidf.vocabulary_)}")

# 3.  RECOMMENDATION FUNCTION

def recommend(title, n=5):
    """Return top-n similar movies for a given title."""
    title_lower = title.lower()
    matches = df[df["title"].str.lower().str.contains(title_lower)]
    if matches.empty:
        return f"'{title}' not found in database."

    idx       = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:n]

    result = pd.DataFrame([{
        "Rank"       : i+1,
        "Movie"      : df.iloc[s[0]]["title"],
        "Genre"      : df.iloc[s[0]]["genre"],
        "Similarity" : round(s[1], 4)
    } for i, s in enumerate(sim_scores)])
    return result


#4.SAMPLE RECOMMENDATIONS

test_movies = ["Inception", "The Godfather", "Toy Story",
               "Get Out", "Interstellar"]

print("\n" + "─"*60)
print("  SAMPLE RECOMMENDATIONS")
print("─"*60)
for movie in test_movies:
    print(f"\n🎬  Because you liked  →  {movie}")
    recs = recommend(movie)
    print(recs.to_string(index=False))

# 5. K-MEANS GENRE CLUSTERING

k = 6
km = KMeans(n_clusters=k, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(tfidf_matrix.toarray())

print("\n─"*30)
print("  K-MEANS CLUSTERING (6 clusters)")
print("─"*30)
for c in range(k):
    titles = df[df["cluster"] == c]["title"].tolist()
    print(f"\n  Cluster {c}: {titles}")

# 6.  VISUALISATIONS

# 6a. Genre bar chart

genre_counts = genre_main.value_counts()
plt.figure(figsize=(10, 5))
bars = plt.bar(genre_counts.index, genre_counts.values,
               color=sns.color_palette("tab10", len(genre_counts)),
               edgecolor="white")
plt.title("Movie Count by Primary Genre", fontsize=14, fontweight="bold")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=35, ha="right")
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.1, str(int(bar.get_height())),
             ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("plots/01_genre_distribution.png", dpi=150)
plt.close()

# 6b. Cosine similarity heatmap (first 15 movies)

plt.figure(figsize=(12, 10))
subset = cosine_sim[:15, :15]
sns.heatmap(subset, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=df["title"][:15], yticklabels=df["title"][:15],
            linewidths=0.3)
plt.title("Cosine Similarity – First 15 Movies", fontsize=13, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("plots/02_similarity_heatmap.png", dpi=150)
plt.close()

# 6c. Recommendation bar chart for "Inception"

recs_inc = recommend("Inception")
plt.figure(figsize=(8, 4))
plt.barh(recs_inc["Movie"], recs_inc["Similarity"],
         color=sns.color_palette("Blues_r", 5), edgecolor="white")
plt.xlabel("Cosine Similarity Score")
plt.title("Top 5 Recommendations for 'Inception'", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/03_inception_recommendations.png", dpi=150)
plt.close()

# 6d. Cluster distribution pie chart

cluster_sizes = df["cluster"].value_counts().sort_index()
plt.figure(figsize=(7, 7))
plt.pie(cluster_sizes, labels=[f"Cluster {i}" for i in cluster_sizes.index],
        autopct="%1.1f%%",
        colors=sns.color_palette("tab10", k),
        startangle=140, wedgeprops={"edgecolor":"white"})
plt.title("K-Means Cluster Distribution", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/04_cluster_distribution.png", dpi=150)
plt.close()

print("\n✅  All plots saved to  ./plots/")
print("=" * 60)
