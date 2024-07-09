# TruthOrLieAI

## Introduction

This repository contains a simple example of Artificial Intelligence (AI) that classifies whether a sentence is true or false. We use Python and the `scikit-learn` library to create and train our text classification model.

### What is Text Classification?

Text classification is a branch of machine learning and natural language processing (NLP) that deals with assigning predefined categories to text data. It is widely used in various applications such as sentiment analysis, spam detection, and topic categorization.

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
    git clone https://github.com/your-username/TruthOrLieAI.git
    cd TruthOrLieAI
    ```

2. Create a virtual environment (recommended) and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the necessary dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

After installing the dependencies, you can run the script to train the model and test some sentences:

1. Run the main script:

    ```bash
    python main.py
    ```

2. You will see output in the console showing the model's accuracy and predictions for some example sentences.

3. To test with your own sentences, edit the `predict_truth_lie` function in the `main.py` script or add new sentences to the examples list.

## Contribution

If you want to contribute to this project, follow these steps:

1. Fork this repository.
2. Create a branch for your feature or bug fix (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to adjust this README as necessary to better suit your project and users!
