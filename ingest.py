import os
import glob

import chromadb
from sentence_transformers import SentenceTransformer

DATA_FOLDER = "data"
DB_FOLDER = "chroma_db"
COLLECTION_NAME = "articles"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 200  # words per chunk
CHUNK_OVERLAP = 40  # words shared between two neighbour chunks
BATCH_SIZE = 500  # how many chunks we embed and store at once


def load_articles(file_path):
    articles = []
    # "r8-train-all-terms.txt" -> "train"
    file_name = os.path.basename(file_path)
    file_tag = file_name.split("-")[1]
    # "r8-train-all-terms.txt" -> "r8-train" (used to control who can read it)
    source = "r8-" + file_tag

    with open(file_path, "r", encoding="utf-8") as f:
        line_number = 0
        for line in f:
            line = line.strip()
            if line == "":
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            category = parts[0]
            text = parts[1]
            article_id = file_tag + "-" + str(line_number)
            articles.append((article_id, category, text, source))
            line_number = line_number + 1
    return articles


def chunk_text(text):
    words = text.split()

    # Short article: keep it as one single chunk.
    if len(words) <= CHUNK_SIZE:
        return [text]

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = start + step
    return chunks


def build_chunks(articles):
    ids = []
    texts = []
    metadatas = []
    for article_id, category, text, source in articles:
        pieces = chunk_text(text)
        chunk_number = 0
        for piece in pieces:
            ids.append(article_id + "-chunk" + str(chunk_number))
            texts.append(piece)
            metadatas.append(
                {"category": category, "article_id": article_id, "source": source}
            )
            chunk_number = chunk_number + 1
    return ids, texts, metadatas


def store_chunks(ids, texts, metadatas):
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=DB_FOLDER)

    existing_names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    for start in range(0, len(texts), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_texts = texts[start:end]
        embeddings = model.encode(batch_texts).tolist()
        collection.add(
            ids=ids[start:end],
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=metadatas[start:end],
        )
        print("Stored", min(end, len(texts)), "of", len(texts), "chunks")


def main():
    all_articles = []
    for file_path in sorted(glob.glob(os.path.join(DATA_FOLDER, "r8-*.txt"))):
        articles = load_articles(file_path)
        print("Loaded", len(articles), "articles from", file_path)
        all_articles = all_articles + articles

    ids, texts, metadatas = build_chunks(all_articles)
    store_chunks(ids, texts, metadatas)

    print("Done. Articles stored:", len(all_articles))
    print("Chunks stored:", len(ids))


if __name__ == "__main__":
    main()
