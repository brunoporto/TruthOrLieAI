import joblib

# Loading the trained model and vectorizer
model = joblib.load('truth_lie_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Function to predict if a new sentence is true or false
def predict_truth_lie(sentence):
    sentence_vectorized = vectorizer.transform([sentence])
    prediction = model.predict(sentence_vectorized)
    return "Verdade" if prediction[0] == 1 else "Mentira"

# Testing the prediction function with examples
examples = [
    "A lua é feita de queijo.",
    "Gatos podem ver no escuro.",
    "Elefantes são os maiores mamíferos terrestres.",
    "Um ser humano possui dois corações.",
    "Um coração pertence a dois seres humanos.",
]

for example in examples:
    print(f"A frase '{example}' é: {predict_truth_lie(example)}")
