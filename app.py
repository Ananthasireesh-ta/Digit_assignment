import os
import sys
from flask import Flask, request, render_template, jsonify
from sentence_transformers import SentenceTransformer
import joblib
import pandas as pd
import re
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder='templates')


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# Load the pre-trained sentence transformer model
embed_model_path = resource_path('all-MiniLM-L6-v2')
embed_model = SentenceTransformer(embed_model_path)

# Load the saved K-means model
kmeans_model_path = resource_path('kmeans_model.sav')
kmeans_loaded = joblib.load(kmeans_model_path)

# Load the topics
topics_path = resource_path('top_words_per_cluster.csv')
topics = pd.read_csv(topics_path)


def preprocess_text(text):
    # Convert text to lower
    text = text.lower()
    # Remove hyperlinks and URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove characters like /n and /r
    text = text.replace('\n', ' ').replace('\r', ' ')
    return text


def inference(text, embed_model, kmeans_loaded):
    text = preprocess_text(text)
    # Create embeddings
    embeddings = embed_model.encode([text])
    # Predict cluster using K-means model
    cluster_label = kmeans_loaded.predict(embeddings.reshape(-1, 384))
    return cluster_label


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    pred_cluster = inference(text, embed_model, kmeans_loaded)
    cluster_number = int(pred_cluster[0])
    top_words = topics[topics['Cluster_No'] == cluster_number]['Top_5_words'].values[0]
    return jsonify(cluster=cluster_number, top_words=top_words)


if __name__ == '__main__':
    app.run(debug=True)
