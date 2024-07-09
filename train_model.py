import argparse
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

nltk.download('stopwords')

def train_model(data_file, vectorizer_type, language='portuguese'):
    # Load data
    data = pd.read_csv(data_file)
    sentences = data['sentence'].values
    labels = data['label'].values

    # Load stopwords based on language
    stop_words = stopwords.words(language)

    # Choose vectorizer
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words=stop_words)
    elif vectorizer_type == 'count':
        vectorizer = CountVectorizer(stop_words=stop_words)
    else:
        raise ValueError("Unsupported vectorizer type. Choose 'count' or 'tfidf'.")

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(sentences, labels, test_size=0.2, random_state=42)

    # Create and train the model
    model = make_pipeline(vectorizer, MultinomialNB())
    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred) * 100  # Convert accuracy to percentage

    # Save the model and vectorizer
    with open('truth_lie_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('vectorizer.pkl', 'wb') as vec_file:
        pickle.dump(vectorizer, vec_file)

    num_sentences = len(sentences)
    num_tokens = len(vectorizer.get_feature_names_out())

    print(f"Model training complete and saved to 'truth_lie_model.pkl' and 'vectorizer.pkl'.")
    print(f"Trained on {num_sentences} sentences.")
    print(f"Learned {num_tokens} tokens.")
    print(f"Accuracy on validation set: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to classify text as truth or lie.")
    parser.add_argument('--data', type=str, default='train_data.csv', help='CSV file containing training data.')
    parser.add_argument('--vectorizer', type=str, default='tfidf', help='Vectorizer type: "count" or "tfidf".')
    parser.add_argument('--language', type=str, default='portuguese', help='Language for stopwords (default: "portuguese").')

    args = parser.parse_args()

    train_model(args.data, args.vectorizer, args.language)
