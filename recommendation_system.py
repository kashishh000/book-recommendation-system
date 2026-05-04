#===================== BOOK RECOMMENDATION PROJECT =====================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 130

# ===================== 1. LOAD DATASET =====================
books_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
ratings_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"

books = pd.read_csv(books_url)
ratings = pd.read_csv(ratings_url)

# Merge data
df = ratings.merge(books[['book_id','title','authors']], on='book_id')
df.rename(columns={'authors':'author'}, inplace=True)

print("Dataset loaded successfully!")
print(df.head())

# ===================== 2. CLEAN SIMPLE GRAPHS (NO WARNINGS) =====================

# -------- GRAPH 1: Ratings Distribution --------
plt.figure(figsize=(5,3))
sns.countplot(data=df, x='rating', hue='rating', palette='pastel', legend=False)
plt.title("Rating Count")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

# -------- GRAPH 2: Top 10 Most Rated Books --------
top10 = df['title'].value_counts().head(10)

plt.figure(figsize=(6,4))
sns.barplot(y=top10.index[::-1], x=top10.values[::-1], palette="coolwarm")
plt.title("Top 10 Most Rated Books")
plt.xlabel("Ratings")
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

# -------- GRAPH 3: Mini Heatmap (10x10) --------
pivot = df.pivot_table(index='user_id', columns='title', values='rating').fillna(0)

small_users = pivot.index[:10]
small_books = pivot.columns[:10]

plt.figure(figsize=(5,4))
sns.heatmap(pivot.loc[small_users, small_books], cmap="Greens")
plt.title("Mini Ratings Heatmap")
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

# ===================== 3. SIMPLE RECOMMENDER ENGINE =====================

# Use only top 10 books for easy/simple model
popular_books = df['title'].value_counts().head(10).index.tolist()
df_pop = df[df['title'].isin(popular_books)]
pivot_pop = df_pop.pivot_table(index='user_id', columns='title', values='rating').fillna(0)

# Book-book similarity
similarity_matrix = cosine_similarity(pivot_pop.T)
sim_df = pd.DataFrame(similarity_matrix, index=pivot_pop.columns, columns=pivot_pop.columns)

def recommend(book, topn=5):
    """Return topn similar books using cosine similarity"""
    if book not in sim_df.index:
        return "Book not found! Choose from the Top 10 list."
    scores = sim_df[book].sort_values(ascending=False)[1:topn+1]
    return scores

# ===================== 4. DEMO OUTPUT =====================
print("\nExample Recommendation:")
input_book = popular_books[0]  # first most popular book
print("Input Book:", input_book)
print("\nRecommended Books:\n")
print(recommend(input_book, topn=5))
