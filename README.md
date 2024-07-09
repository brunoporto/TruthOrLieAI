
# TruthOrLieAI

## Introduction

This repository contains a simple example of Artificial Intelligence (AI) that classifies whether a sentence is Truth or Lie. We use Python and the `scikit-learn` library to create and train our text classification model.

### What is Text Classification?

Text classification is a branch of machine learning and natural language processing (NLP) that deals with assigning predefined categories to text data. It is widely used in various applications such as sentiment analysis, spam detection, and topic categorization.

### Other Branches of Machine Learning and NLP

Machine learning and NLP encompass a wide range of techniques and tasks, including:

- **Sentiment Analysis**: Determining whether a text expresses positive, negative, or neutral sentiment.
- **Named Entity Recognition (NER)**: Identifying and classifying proper nouns in text (e.g., names of people, organizations, locations).
- **Machine Translation**: Automatically translating text from one language to another.
- **Speech Recognition**: Converting spoken language into text.
- **Text Summarization**: Generating a concise summary of a longer text.
- **Question Answering**: Building systems that can answer questions posed in natural language.
- **Language Modeling**: Predicting the next word in a sequence of words, useful in text generation tasks.
- **Topic Modeling**: Discovering abstract topics within a collection of documents.
- **Part-of-Speech Tagging**: Assigning parts of speech (e.g., nouns, verbs, adjectives) to each word in a sentence.

### Model and Library

In this project, we use the Naive Bayes classifier, a probabilistic model particularly well-suited for text classification tasks due to its simplicity and effectiveness. The `scikit-learn` library is an open-source machine learning library in Python that provides simple and efficient tools for data mining and data analysis. It is built on NumPy, SciPy, and matplotlib.

### Why `scikit-learn`?

`scikit-learn` is a powerful library that offers:

- A variety of supervised and unsupervised learning algorithms.
- Tools for model fitting, data preprocessing, model selection, and evaluation.
- Comprehensive documentation and a supportive community.
- Easy integration with other Python libraries.

### Applications of Text Classification

Text classification models can solve numerous real-world problems, including:

- **Sentiment Analysis**: Determining whether a text expresses positive, negative, or neutral sentiment.
- **Spam Detection**: Identifying whether an email or message is spam or not.
- **Topic Categorization**: Classifying documents into predefined categories based on their content.
- **Language Detection**: Identifying the language in which a given text is written.

## Requirements

Before you begin, make sure you have the following requirements installed on your system:

- Python 3.6 or higher
- pip (Python package installer)

## Installation

Follow these steps to install and set up the environment:

1. Clone the repository to your computer:

    ```bash
    git clone https://github.com/brunoporto/TruthOrLieAI.git
    cd TruthOrLieAI
    ```

2. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```
    - On macOS and Linux:
      ```bash
      source venv/bin/activate
      ```

4. Upgrade `pip` and `setuptools`:

    ```bash
    pip install --upgrade pip setuptools
    ```

5. Install the necessary dependencies:

    ```bash
    pip install -r requirements.txt
    ```

6. For macOS users, if you encounter issues with `distutils`, install it via Homebrew:

    ```bash
    brew install python3-distutils
    ```

## Usage

### Generating Data

To generate a dataset of truth and lie sentences, use the `generate_data.py` script. This script generates an equal number of true and false statements and saves them to a CSV file. 

Run the following command to generate the data:

```bash
python generate_data.py
```

- `--num_phrases`: The total number of phrases to generate (half will be true and half will be false).
- `--output`: The output CSV file where the generated data will be saved.
- `--language`: The language to use for generating phrases (default is `portuguese`).
- `--prompt_truth`: The prompt to use for generating true phrases.
- `--prompt_lie`: The prompt to use for generating false phrases.

Example usage:

```bash
python generate_data.py --num_phrases 100 --output generated_data.csv --language portuguese --prompt_truth "A verdade é" --prompt_lie "A mentira é"
```

This will generate 100 phrases (50 true and 50 false) and save them to `data.csv`.

### Training the Model

To train the model, run the following command:

    ```bash
    python train_model.py
    ```

To train the model using `CountVectorizer`:

    ```bash
    python train_model.py --vectorizer count
    ```

To train the model using `TfidfVectorizer` and a different dataset `train_data_50000.csv`:

    ```bash
    python train_model.py --data train_data_50000.csv --vectorizer tfidf
    ```

This will load the data from the specified CSV file, train the model, and save the trained model and vectorizer to disk.

### Making Predictions

To use the trained model to make predictions, run the following command:

    ```bash
    python predict.py
    ```

This will load the trained model and vectorizer, and print the predictions for some example sentences. You can edit the `predict.py` script to test with your own sentences.

### Explanation of Saved Files

- `truth_lie_model.pkl`: This file contains the trained Naive Bayes model. The model is used to predict whether a new sentence is Truth or Lie based on the patterns it learned during training.
- `vectorizer.pkl`: This file contains the fitted `CountVectorizer` or `TfidfVectorizer` used to convert text data into numerical format (word count vectors). It ensures that any new text data is transformed in the same way as the training data.

## Contribution

If you want to contribute to this project, follow these steps:

1. Fork this repository.
2. Create a branch for your feature or bug fix (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
