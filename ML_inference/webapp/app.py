from flask import Flask, render_template, request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the TF-IDF vectorizer
vectorizer = joblib.load('./model/tfidf_vectorizer.pkl')

# Load the trained model
model = joblib.load('./model/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        
        # Preprocess the input text using the loaded TF-IDF vectorizer
        text_vectorized = vectorizer.transform([text])
        
        # Predict sentiment
        prediction = model.predict(text_vectorized)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"

    # Determine the emoji based on the sentiment
        if sentiment == 'Positive':
            emoji = '😊' # Happy face for positive sentiment
        else:
            emoji = '😞' # Sad face for negative sentiment

    # Render your template with the prediction and emoji
        return render_template('index.html', text=text, prediction=sentiment, emoji=emoji)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
