import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load the movie data
data = pd.read_csv("movie_data.csv")

# Separate the plot summaries and genres
plot_summaries = data["plot_summary"]
genres = data["genre"]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the plot summaries into vectors
plot_summaries_tfidf = vectorizer.fit_transform(plot_summaries)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(plot_summaries_tfidf, genres, test_size=0.2)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Evaluate the model on the testing set
accuracy = classifier.score(X_test, y_test)
print("Accuracy:", accuracy)

# Use the model to predict the genre of a new movie
new_movie_summary = "A group of friends go on a camping trip and are stalked by a mysterious killer."
new_movie_summary_tfidf = vectorizer.transform([new_movie_summary])
predicted_genre = classifier.predict(new_movie_summary_tfidf)
print("Predicted genre:", predicted_genre)
