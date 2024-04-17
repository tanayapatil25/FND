
from flask import Flask, request, render_template
import joblib

# Load trained model and TF-IDF vectorizer
svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    title = request.form['title']
    text = request.form['text']

    # Preprocess input data

    # Feature extraction using TF-IDF
    input_text_tfidf = tfidf_vectorizer.transform([text])

    # Make prediction
    prediction = svm_model.predict(input_text_tfidf)

    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
