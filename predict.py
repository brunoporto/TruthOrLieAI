import pickle
import numpy as np

# Load the model and feature vectorizer
with open('truth_lie_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def predict_truth_lie(sentence):
    # Transform the input sentence using the feature vectorizer
    sentence_vectorized = model.named_steps['tfidfvectorizer'].transform([sentence])
    
    # Make the prediction using the loaded model
    prediction_proba = model.named_steps['multinomialnb'].predict_proba(sentence_vectorized)
    prediction = np.argmax(prediction_proba)
    confidence = prediction_proba[0][prediction] * 100
    
    result = "VERDADE" if prediction == 1 else "MENTIRA"
    return result, confidence

# Example sentences
examples = [
    "O céu é azul.",
    "Os cães podem voar.",
    "A água ferve a 100 graus Celsius.",
    "O corpo humano tem dois corações.",
]

for example in examples:
    result, confidence = predict_truth_lie(example)
    print(f"Tenho {confidence:.2f}% de certeza de que a frase '{example}' é {result}")
