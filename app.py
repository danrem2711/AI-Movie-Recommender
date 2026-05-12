from flask import Flask, render_template, request, redirect
import pandas as pd
import requests
import os
import ast
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "favorites.db")


def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT UNIQUE,
            genre TEXT,
            rating REAL,
            poster TEXT,
            trailer TEXT
        )
    """)

    conn.commit()
    conn.close()


init_database()


movies = pd.read_csv(os.path.join(BASE_DIR, "tmdb_5000_movies.csv"))

movies = movies.rename(columns={
    "overview": "description",
    "vote_average": "rating"
})

movies["title"] = movies["title"].astype(str)
movies["description"] = movies["description"].fillna("")
movies["rating"] = movies["rating"].fillna(0)

def clean_genres(genre_text):
    try:
        genres = ast.literal_eval(genre_text)
        return ", ".join([genre["name"] for genre in genres])
    except:
        return ""

movies["genre"] = movies["genres"].apply(clean_genres)

available_genres = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Drama",
    "Family",
    "Fantasy",
    "Horror",
    "Mystery",
    "Romance",
    "Science Fiction",
    "Thriller"
]
movies["tmdb_id"] = movies["id"]

TMDB_API_KEY = "b0a93cfaf1b1507833ac9b9e98b9e887"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_VIDEO_URL = "https://api.themoviedb.org/3/movie/{movie_id}/videos"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

movies["combined_features"] = (
    movies["genre"].fillna("") + " " +
    movies["description"].fillna("")
)

vectorizer = CountVectorizer()
feature_matrix = vectorizer.fit_transform(movies["combined_features"])
similarity = cosine_similarity(feature_matrix)


def get_movie_details_by_id(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"

        params = {
            "api_key": TMDB_API_KEY
        }

        response = requests.get(url, params=params)
        data = response.json()

        poster_path = data.get("poster_path")

        poster = (
            TMDB_IMAGE_BASE_URL + poster_path
            if poster_path
            else "https://placehold.co/500x750?text=No+Poster"
        )

        genres = ", ".join([genre["name"] for genre in data.get("genres", [])])

        return {
            "poster": poster,
            "title": data.get("title", "Unknown"),
            "overview": data.get("overview", "No overview available."),
            "release_date": data.get("release_date", "Unknown"),
            "runtime": data.get("runtime", "Unknown"),
            "rating": data.get("vote_average", 0),
            "genres": genres,
            "language": data.get("original_language", "Unknown").upper(),
            "status": data.get("status", "Unknown")
        }

    except:
        return {
            "poster": "https://placehold.co/500x750?text=No+Poster",
            "title": "Unknown",
            "overview": "No overview available.",
            "release_date": "Unknown",
            "runtime": "Unknown",
            "rating": 0,
            "genres": "",
            "language": "Unknown",
            "status": "Unknown"
        }

def get_movie_trailer(movie_id):
    try:
        url = TMDB_VIDEO_URL.format(movie_id=movie_id)

        params = {
            "api_key": TMDB_API_KEY
        }

        response = requests.get(url, params=params)
        data = response.json()

        videos = data.get("results", [])

        for video in videos:
            if video.get("site") == "YouTube" and video.get("type") == "Trailer":
                key = video.get("key")
                return f"https://www.youtube.com/embed/{key}"

        for video in videos:
            if video.get("site") == "YouTube":
                key = video.get("key")
                return f"https://www.youtube.com/embed/{key}"

        return ""

    except:
        return ""


def recommend_movies(movie_title, selected_genre="All"):
    movie_title = movie_title.lower()
    matching_movies = movies[movies["title"].str.lower() == movie_title]

    if matching_movies.empty:
        return []

    movie_index = matching_movies.index[0]

    similarity_scores = list(enumerate(similarity[movie_index]))
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations = []

    for index, score in sorted_movies[1:30]:
        title = movies.iloc[index]["title"]
        movie_id = movies.iloc[index]["tmdb_id"]
        movie_genre = movies.iloc[index]["genre"]

        if selected_genre != "All" and selected_genre not in movie_genre:
            continue

        recommendations.append({
            "title": title,
            "genre": movie_genre,
            "description": movies.iloc[index]["description"],
            "rating": movies.iloc[index]["rating"],
            "poster": get_movie_details_by_id(movie_id)["poster"],
            "trailer": get_movie_trailer(movie_id),
            "match_score": round(score * 100)
        })

        if len(recommendations) == 6:
            break

    return recommendations


def get_dataset_poster(poster_path):
    if pd.notna(poster_path) and poster_path != "":
        return TMDB_IMAGE_BASE_URL + poster_path

    return "https://placehold.co/500x750?text=No+Poster"

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    selected_movie = ""
    selected_poster = ""
    selected_trailer = ""
    selected_genre = "All"
    selected_details = None

    movie_titles = sorted(movies["title"].dropna().unique().tolist())

    if request.method == "POST":
        selected_movie = request.form.get("movie")
        selected_genre = request.form.get("genre")

        matching_movies = movies[movies["title"].str.lower() == selected_movie.lower()]

        if not matching_movies.empty:
            selected_id = matching_movies.iloc[0]["tmdb_id"]
            selected_details = get_movie_details_by_id(selected_id)
            selected_poster = selected_details["poster"]
            selected_trailer = get_movie_trailer(selected_id)
            recommendations = recommend_movies(selected_movie, selected_genre)

    return render_template(
        "index.html",
        movie_titles=movie_titles,
        recommendations=recommendations,
        selected_movie=selected_movie,
        selected_poster=selected_poster,
        selected_trailer=selected_trailer,
        genres=available_genres,
        selected_genre=selected_genre,
        selected_details=selected_details
    )

@app.route("/save_favorite", methods=["POST"])
def save_favorite():
    title = request.form.get("title")
    genre = request.form.get("genre")
    rating = request.form.get("rating")
    poster = request.form.get("poster")
    trailer = request.form.get("trailer")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR IGNORE INTO favorites (title, genre, rating, poster, trailer)
        VALUES (?, ?, ?, ?, ?)
    """, (title, genre, rating, poster, trailer))

    conn.commit()
    conn.close()

    return redirect(request.referrer or "/")

@app.route("/favorites")
def favorites():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id, title, genre, rating, poster, trailer FROM favorites")
    rows = cursor.fetchall()

    conn.close()

    favorite_movies = []

    for row in rows:
        favorite_movies.append({
            "id": row[0],
            "title": row[1],
            "genre": row[2],
            "rating": row[3],
            "poster": row[4],
            "trailer": row[5]
        })

    return render_template("favorites.html", favorite_movies=favorite_movies)

@app.route("/remove_favorite/<int:favorite_id>", methods=["POST"])
def remove_favorite(favorite_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM favorites WHERE id = ?", (favorite_id,))

    conn.commit()
    conn.close()

    return redirect("/favorites")

if __name__ == "__main__":
    app.run(debug=True)