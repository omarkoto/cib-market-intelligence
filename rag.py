import json
import os

import chromadb
import requests
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from auth import ROLE_SOURCES

DB_FOLDER = "chroma_db"
COLLECTION_NAME = "articles"
MODEL_NAME = "all-MiniLM-L6-v2"  # must be the same model used in ingest.py
SETTINGS_FILE = "settings.json"  # which LLM provider and model to use
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MAX_DISTANCE = 1.10  # if the best chunk is farther than this, the question is off-topic
NO_ANSWER_TEXT = (
    "I can't answer that from this knowledge base. It contains historical "
    "Reuters financial news (1987) on earnings, acquisitions, oil, trade, "
    "currencies, interest rates, shipping and grain."
)

embedding_model = SentenceTransformer(MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
collection = chroma_client.get_collection(COLLECTION_NAME)


def retrieve(question, k=5, category=None, role="member"):
    question_embedding = embedding_model.encode([question]).tolist()

    # Access is enforced here, in retrieval, so the LLM never sees chunks
    # the user is not allowed to read. A prompt rule could be ignored by the
    # LLM, but a chunk that is never retrieved can never leak.
    source_filter = {"source": {"$in": ROLE_SOURCES[role]}}

    # Add the category filter only when the user chose one.
    if category is None:
        where_filter = source_filter
    else:
        where_filter = {"$and": [source_filter, {"category": category}]}

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=k * 2,
        where=where_filter,
    )

    # Chroma returns lists inside lists (one list per question); we asked one question.
    texts = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Chroma returns results sorted from closest to farthest, so the first
    # time we see an article_id, that chunk is its best (smallest distance).
    chunks = []
    seen_articles = []
    for i in range(len(texts)):
        article_id = metadatas[i]["article_id"]
        if article_id in seen_articles:
            continue
        seen_articles.append(article_id)
        chunk = {
            "text": texts[i],
            "article_id": article_id,
            "category": metadatas[i]["category"],
            "source": metadatas[i]["source"],
            "distance": distances[i],
        }
        chunks.append(chunk)
        if len(chunks) == k:
            break
    return chunks


# Reads settings.json and returns it as a dictionary.
# WHY: the admin can change the provider and model without editing code.
def load_settings():
    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# Writes the settings dictionary back to settings.json.
# WHY: the admin's choice must survive after the page reloads.
def save_settings(settings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


# Returns the text "provider/model" for the current settings, e.g. "groq/openai/gpt-oss-120b".
# WHY: the app shows it under each answer so the user knows which model replied.
def current_model_label():
    settings = load_settings()
    if settings["provider"] == "ollama":
        return "ollama/" + settings["ollama_model"]
    return "groq/" + settings["groq_model"]


# Asks the local Ollama server which models are installed; returns their names.
# WHY: the admin picks from this list. If Ollama is not running we return an
# empty list instead of crashing, and the app shows a helpful warning.
def list_local_models():
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=3)
        response.raise_for_status()
        names = []
        for model in response.json()["models"]:
            names.append(model["name"])
        return names
    except Exception:
        return []


# Sends a prompt to the LLM and returns its reply as text.
# WHY: this is the ONLY place that talks to an LLM provider. It reads
# settings.json to choose between Groq (cloud) and Ollama (local).
def ask_llm(prompt):
    settings = load_settings()
    messages = [{"role": "user", "content": prompt}]

    if settings["provider"] == "ollama":
        if settings["ollama_model"] == "":
            raise ValueError("No local model selected. Ask the admin to choose one.")
        body = {
            "model": settings["ollama_model"],
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }
        response = requests.post(OLLAMA_URL, json=body, timeout=300)
        response.raise_for_status()
        return response.json()["message"]["content"]

    load_dotenv()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=settings["groq_model"],
        messages=messages,
        temperature=0,  # 0 = same question gives the same answer, no creativity
    )
    return response.choices[0].message.content


def answer(question, category=None, role="member"):
    chunks = retrieve(question, category=category, role=role)

    # If even the best chunk is too far away, the question is off-topic.
    # We skip the LLM so it cannot make up an answer (and we save a call).
    if len(chunks) == 0 or chunks[0]["distance"] > MAX_DISTANCE:
        return NO_ANSWER_TEXT, [], ""  # no model was used

    articles_text = ""
    for chunk in chunks:
        articles_text = articles_text + "[" + chunk["article_id"] + "] " + chunk["text"] + "\n\n"

    prompt = (
        "You are a financial news assistant. "
        "Answer ONLY using the articles below. "
        "Cite article IDs like [train-12]. "
        "If the answer is not in the articles, say you don't know. "
        "Note: numbers were removed from this dataset, so do not invent figures. "
        "Never quote text that contains a removed number (like '¥ …' or 'dlrs a barrel'); "
        "paraphrase instead. "
        "Describe trends as what the 1987 Reuters reports said, not as current market conditions. "
        "For deals, distinguish the acquirer, the target, a takeover bid, "
        "and the sale of a division. "
        "Use exactly this citation format: [train-12].\n\n"
        "Articles:\n" + articles_text +
        "Question: " + question
    )

    reply = ask_llm(prompt)

    # The LLM does not always follow the citation format rule and sometimes
    # writes 【train-12】 instead of [train-12]. We fix it here so citations
    # always use plain square brackets.
    reply = reply.replace("【", "[")
    reply = reply.replace("】", "]")
    return reply, chunks, current_model_label()


if __name__ == "__main__":
    reply, sources, model = answer("What is happening with OPEC and oil prices?")
    print("ANSWER (" + model + "):")
    print(reply)
    print()
    print("SOURCES:")
    for source in sources:
        print(source["article_id"], source["category"], round(source["distance"], 3))
