import pinecone
import re
import os
import numpy as np

from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
from pathlib import Path
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

INDEX_NAME = "yash-index"  # Pinecone index

# # --------- Extract your text data (can use library for parsing only) -------- #

dir = Path("scenes_drama")

scripts = {}

for f in dir.glob("*.txt"):
    data = f.read_text(encoding="utf-8")
    scripts[f.name] = data

# # --------------------------------- Chunk it --------------------------------- #

def simple_sentence_tokenize(text):
    """
    Tokenizes text using punctuation as sentence boundaries.
    """
    text = text.replace("\n", " ")
    return re.split(r'(?<=[.!?])\s+', text)

def chunk_text(text, max_words=500, overlap_sentences=2):
    """
    Splits texts into max_words words per chunk, with overlap_sentences sentences overlapping between chunks.
    """
    sentences = simple_sentence_tokenize(text)
    chunks = []
    chunk = []
    word_count = 0

    for sentence in sentences:
        words_in_sentence = sentence.split()
        num_words = len(words_in_sentence)
        
        if word_count + num_words > max_words and chunk:
            chunk = " ".join(chunk)
            chunks.append(chunk)
            
            if overlap_sentences > 0:
                chunk = chunk[-overlap_sentences:]
                word_count = sum(len(s.split()) for s in chunk)
            else:
                chunk = []
                word_count = 0
        
        chunk.append(sentence)
        word_count += num_words

    if chunk:
        chunks.append(" ".join(chunk))
    
    return chunks

script_chunks = {}
for filename, text in scripts.items():
    chunks = chunk_text(text, max_words=500, overlap_sentences=2)
    script_chunks[filename] = chunks

for filename, chunks in script_chunks.items():
    print(f"{filename} -> {len(chunks)} chunks")

all_chunks = []
for chunks in script_chunks.values():
    all_chunks.extend(chunks)

print(f"Total number of chunks from all scripts: {len(all_chunks)}")

# -------------------------- Store it in a database -------------------------- #

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(all_chunks).tolist()

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# if INDEX_NAME not in pc.list_indexes():
#     pc.create_index(
#         name=INDEX_NAME, 
#         dimension=384,
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )

# IMPORTANT: Connecting to index
index = pc.Index(INDEX_NAME)

# Note: Got this from the Pinecone documentation
vectors_to_upsert = []
for i, (chunk, vec) in enumerate(zip(all_chunks, embeddings)):
    vectors_to_upsert.append((str(i), vec, {"text": chunk}))

batch_size = 100

for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i + batch_size]
    index.upsert(vectors=batch)
    print(f"Upserted batch {i // batch_size + 1} of {len(vectors_to_upsert) // batch_size + 1}")

print(f"Upserted {len(vectors_to_upsert)} vectors into the Pinecone index '{INDEX_NAME}'.")

# ----------------- Perform a retrieval using semantic search ---------------- #

def cosine_similarity(vec, vec2):
    vec = np.array(vec)
    vec2 = np.array(vec2)
    return np.dot(vec, vec2) / (np.linalg.norm(vec) * np.linalg.norm(vec2))

query = "Who lightly plays their violin in the movie 12 years a slave?"
embedding = model.encode(query).tolist()

vec_response = index.fetch(ids=[str(i) for i in range(1000)])
pc_embeddings = []
pc_metadata = []

for id, info in vec_response.vectors.items():
    pc_embeddings.append(info.values)
    pc_metadata.append(info.metadata["text"])

top_k = 5
cosine_scores = []

for i, vec in enumerate(pc_embeddings):
    score = cosine_similarity(embedding, vec)
    cosine_scores.append((i, score))

cosine_scores = sorted(cosine_scores, key=lambda x: x[1], reverse=True)
top_k_scores = cosine_scores[:top_k]

retrieved_contexts = []
for i, score in top_k_scores:
    retrieved_contexts.append(pc_metadata[i])

retrieved_context = "\n\n".join(retrieved_contexts)

# Create prompt
prompt = f"""Below is some context retrieved from our documents:
{retrieved_context}

Based on the above context, answer the following question:
{query}

Answer:"""

print("\nConstructed Prompt:\n", prompt)

# ---------------- Insert relevant context into the LLM prompt --------------- #

# Partial from HuggingFaceAPI documentation, modified for our use case
client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

messages = [
	{
		"role": "user",
		"content": prompt
	}
]

response_text = ""
stream = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3", 
    messages=messages, 
    max_tokens=500,
    stream=True
)

for chunk in stream:
    part = chunk.choices[0].delta.content
    response_text += part
    print(part, end="")

# ------------------------- Ragas Evaluation Metrics ------------------------- #

'''
Initially tried to use the evaluate library from HuggingFace, but it was not working as expected as crucial dependencies for rouge
and other evaluation metrics were unavailable without cloning the library's repo.
'''

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

eval_samples = [
    {
        "question": query,
        "answer": response_text,
        "retrieved_contexts": retrieved_contexts,
        "ground_truth": "Solomon Northup"
    },
]

for sample in eval_samples:
    references = [sample["ground_truth"].split()]
    predictions = sample["answer"].split()

    rouge_scores = scorer.score(sample["ground_truth"], sample["answer"])

    print()
    print(f'\nQuestion: {sample["question"]}')
    print(f'Ground Truth: {sample["ground_truth"]}')
    print(f'Generated Answer: {sample["answer"]}')
    
    print("ROUGE Scores:")
    for rouge_type, score in rouge_scores.items():
        print(f'  {rouge_type.upper()}: Precision: {score.precision:.4f}, Recall: {score.recall:.4f}, F1 Score: {score.fmeasure:.4f}')
    print("\n" + "-"*50 + "\n")