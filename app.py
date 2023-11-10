from flask import Flask, render_template, request
import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the CountVectorizer and Naive Bayes model
count_vectorizer = joblib.load(r"E:\Data Science class\NLP\countvectorizer.pkl")
loaded_model = joblib.load(r"E:\Data Science class\NLP\NLPsentiment_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']

        # Split the input text into sentences
        sentences = text.split('.')

        predictions = []

        for sentence in sentences:
            # Preprocess each sentence
            new_data_vectorized = count_vectorizer.transform([sentence])

            # Convert the input to an array-like object
            new_data_array = new_data_vectorized.toarray()

            # Predict sentiment for each sentence
            prediction = loaded_model.predict(new_data_array)[0]

            predictions.append(prediction)

        return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)