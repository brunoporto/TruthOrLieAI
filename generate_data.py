import argparse
import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text, stop_words, lemmatizer):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = nltk.word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize the text
    return ' '.join(tokens)

def generate_phrases(generator, num_phrases, truth=True, stop_words=None, lemmatizer=None, prompt_truth=None, prompt_lie=None):
    phrases = []
    prompt = prompt_truth if truth else prompt_lie

    for _ in tqdm(range(num_phrases), desc="Generating phrases"):
        generated = generator(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']
        preprocessed = preprocess_text(generated, stop_words, lemmatizer)
        phrases.append(preprocessed)
    
    return phrases

def generate_data(num_phrases, output_file, language, prompt_truth, prompt_lie):
    model_name = "pierreguillou/gpt2-small-portuguese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    stop_words = stopwords.words(language)
    lemmatizer = WordNetLemmatizer()

    num_true = num_phrases // 2
    num_lies = num_phrases - num_true

    true_phrases = generate_phrases(generator, num_true, truth=True, stop_words=stop_words, lemmatizer=lemmatizer, prompt_truth=prompt_truth, prompt_lie=prompt_lie)
    lie_phrases = generate_phrases(generator, num_lies, truth=False, stop_words=stop_words, lemmatizer=lemmatizer, prompt_truth=prompt_truth, prompt_lie=prompt_lie)

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['sentence', 'label'])
        for phrase in true_phrases:
            writer.writerow([phrase, 1])
        for phrase in lie_phrases:
            writer.writerow([phrase, 0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate true and false phrases for text classification.")
    parser.add_argument('--num_phrases', type=int, default=100, help='Total number of phrases to generate.')
    parser.add_argument('--output', type=str, default='generated_data.csv', help='Output CSV file name.')
    parser.add_argument('--language', type=str, default='portuguese', help='Language for stopwords.')
    parser.add_argument('--prompt_truth', type=str, default='Gere um fato verdadeiro:', help='Prompt to generate true facts.')
    parser.add_argument('--prompt_lie', type=str, default='Gere uma declaração falsa:', help='Prompt to generate false statements.')

    args = parser.parse_args()

    generate_data(args.num_phrases, args.output, args.language, args.prompt_truth, args.prompt_lie)
