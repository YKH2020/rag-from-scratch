import pinecone
import os
import numpy as np
import gradio as gr

from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from rouge_score import rouge_scorer
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

INDEX_NAME = "yash-index"

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

model = SentenceTransformer('all-MiniLM-L6-v2')

client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

def cosine_similarity(vec, vec2):
    vec = np.array(vec)
    vec2 = np.array(vec2)
    return np.dot(vec, vec2) / (np.linalg.norm(vec) * np.linalg.norm(vec2))

def get_answer(query):
    embedding = model.encode(query).tolist()

    vec_response = index.fetch(ids=[str(i) for i in range(1000)])
    pc_embeddings = []
    pc_metadata = []

    for id, info in vec_response.vectors.items():
        pc_embeddings.append(info.values)
        pc_metadata.append(info.metadata["text"])

    top_k = 5
    cosine_scores = [(i, cosine_similarity(embedding, vec)) for i, vec in enumerate(pc_embeddings)]
    top_k_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)[:top_k]

    retrieved_contexts = [pc_metadata[i] for i, _ in top_k_scores]
    retrieved_context = "\n\n".join(retrieved_contexts)

    prompt = f"""Below is some context retrieved from our documents:
    {retrieved_context}

    Based on the above context, answer the following question:
    {query}

    Answer:"""

    response_text = ""
    stream = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        stream=True
    )

    for chunk in stream:
        response_text += chunk.choices[0].delta.content

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    ground_truth = "Solomon Northup"
    rouge_scores = scorer.score(ground_truth, response_text)

    rouge_results = "\n".join([f"{k}: Precision: {v.precision:.4f}, Recall: {v.recall:.4f}, F1: {v.fmeasure:.4f}" for k, v in rouge_scores.items()])

    return response_text, rouge_results

iface = gr.Interface(
    fn=get_answer,
    inputs=gr.Textbox(label="Enter your question"),
    outputs=[gr.Textbox(label="Generated Answer"), gr.Textbox(label="ROUGE Scores")],
    title="RAG from Scratch with Pinecone and Hugging Face",
    description="Ask Away! But only on indexed data :)"
)

iface.launch()