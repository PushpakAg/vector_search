from flask import Flask, request, render_template
import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd

embedder = SentenceTransformer('all-MiniLM-L6-v2')
app = Flask(__name__)

# Load the dataframe
df = pd.read_csv(r"C:\Users\pushpak\Downloads\11000 Medicine details COPY_exported\11000 Medicine details COPY_exported.csv")
@app.route('/')
def index():
    return render_template('index.html', received_text="", results=None)

@app.route('/submit', methods=['POST'])
def submit():
    text = request.form['text']
    embeddings = torch.load(r"C:\Users\pushpak\Documents\emails_data\ui\embeddings.pth")  # Provide the correct path
    top_k = 10
    query_embedding = embedder.encode(text, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)
    hits = hits[0]
    results = []
    for hit in hits:
        hit_id = hit['corpus_id']
        article_data = df.iloc[hit_id]
        title = article_data["Medicine Name"]
        results.append((title, hit['score']))
    return render_template('index.html', received_text=text, results=results)

if __name__ == '__main__':
    app.run(debug=True)
