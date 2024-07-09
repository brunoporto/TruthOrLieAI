import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np

# Loading training data from CSV
data = pd.read_csv('train_data.csv')

sentences = data['sentence'].values
labels = data['label'].values

# Converting sentences to word count vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Creating and training the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy * 100:.2f}%')

# Function to predict if a new sentence is true or false
def predict_truth_lie(sentence):
    sentence_vectorized = vectorizer.transform([sentence])
    prediction = model.predict(sentence_vectorized)
    return "True" if prediction[0] == 1 else "False"

# Testing the prediction function with examples
examples = [
    "The moon is made of cheese.",
    "Cats can see in the dark.",
    "Elephants are the largest land mammals."
]

for example in examples:
    print(f"The sentence '{example}' is: {predict_truth_lie(example)}")

# Adding new information for retraining if the AI is wrong
new_sentences = ["Elephants are the largest land mammals."]
new_labels = [1]  # 1 for true, 0 for false

# Vectorizing the new sentences
new_sentences_vectorized = vectorizer.transform(new_sentences)

# Adding the new data to the training set
X_train = np.vstack([X_train.toarray(), new_sentences_vectorized.toarray()])
y_train = np.hstack([y_train, new_labels])

# Retraining the model with the new data
model.fit(X_train, y_train)
