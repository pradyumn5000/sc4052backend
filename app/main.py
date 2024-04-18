from typing import Union
import io
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
import requests
import pandas as pd
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://textranksc4052.azurewebsites.net",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_origin_regex=r"https:\/\/textranksc4052\.azurewebsites\.net$",
)

def crawl(current_url):
    text_page = ""    
    # Fetch the page content
    try:
        response = requests.get(current_url, verify=False)  # Disables SSL certificate verification

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from the page under the <div class="caas-body">
        for div in soup.find_all('div', class_='caas-body'):
            # Find all <p> tags and extract the text
            for p in div.find_all('p'):
                text_page += p.get_text()+" "
    except requests.exceptions.RequestException:
        return None
    return text_page

def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def generate_plot(sim_mat):
    nx_graph = nx.from_numpy_array(sim_mat)
    pos = nx.spring_layout(nx_graph, k=0.1)
    plt.figure(figsize=(8, 6))
    nx.draw(nx_graph, pos=pos, with_labels=False, node_size=20, width=0.2)
    # Convert plot to image bytes
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    plt.close()  # Close the plot to free up memory
    img_bytes.seek(0)
    return img_bytes

def text_rank(nx_graph):
    num_iter = 10000
    threshold = 1e-20
    n_graph = nx_graph.number_of_nodes()
    M_graph = np.zeros((n_graph, n_graph))  # Web Graph Matrix M
    E_graph = np.ones((1, n_graph)) / n_graph
    M_graph = nx.adjacency_matrix(nx_graph).toarray().astype(float)
    M_graph = np.abs(M_graph)
    dead_ends_graph = np.where(~M_graph.any(axis=0))[0]

    for col in dead_ends_graph:
        M_graph[:, col] = 1 / n_graph
    M_graph /= M_graph.sum(axis=1)
    teleportation = 0.2
    E_graph = np.ones((1, n_graph)) / n_graph
    E_new_graph = np.zeros((1, n_graph))
    for i in range(num_iter):
        # Calculate the new distribution vector
        E_new_graph = teleportation * np.dot(E_graph, M_graph.T) + (1 - teleportation) / n_graph
        if np.all(np.abs(E_new_graph - E_graph) < threshold):
            break
        E_graph = E_new_graph.copy()
    flattened_data = E_graph.flatten()

    # Get indices of the top 10 highest values
    top_10_indices = np.argsort(flattened_data)[-5:]

    return top_10_indices

def summarize(text):
    sentences = []
    sentences = (sent_tokenize(text))
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    clean_sentences = [s.lower() for s in clean_sentences]
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    word_embeddings = {}
    f = open('/code/glove.6B.50d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((50,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((50,))
        sentence_vectors.append(v)
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,50), sentence_vectors[j].reshape(1,50))[0,0]
    nx_graph = nx.from_numpy_array(sim_mat)
    plot_img_bytes = generate_plot(sim_mat)
    scores = text_rank(nx_graph)
    summary = ""
    for index in sorted(scores):
        summary += sentences[index]
    return summary, plot_img_bytes

@app.get("/")
def read_root():
    return {"Hello": "World"}

class URLInput(BaseModel):
    url: str

@app.post("/textrank/")
async def perform_textrank(url_input: URLInput):
    try:
        url = url_input.url
        text = crawl(url)
        summary, plot_image = summarize(text)
        
        # Convert plot_image to bytes
        plot_image_bytes = plot_image.getvalue()
        plot_image_base64 = base64.b64encode(plot_image_bytes).decode('utf-8')

        # Return summary and plot image as JSON response
        return JSONResponse(
            content={"text_summary": summary, "plot_image": plot_image_base64},
            media_type="application/json",
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
