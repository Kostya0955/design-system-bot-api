import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# 1. Загружаем переменные окружения
load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# 2. Инициализируем клиентов
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

client = OpenAI(api_key=OPENAI_API_KEY)

# 3. Нарезка текста на куски
def split_text(text: str, max_chars: int = 600):
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        if len(current) + len(p) + 2 <= max_chars:
            current = (current + "\n\n" + p) if current else p
        else:
            if current:
                chunks.append(current.strip())
            if len(p) <= max_chars:
                current = p
            else:
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i : i + max_chars].strip())
                current = ""

    if current:
        chunks.append(current.strip())

    return chunks

# 4. Читаем файл c документацией
DOC_PATH = "design-system.md"

with open(DOC_PATH, "r", encoding="utf-8") as f:
    full_text = f.read()

chunks = split_text(full_text, max_chars=600)
print(f"Нарезали файл на {len(chunks)} кусочков")

# 5. Считаем эмбеддинги через OpenAI и готовим вектора
def build_vectors(chunks, batch_size: int = 32):
    vectors = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]

        # Вызываем OpenAI Embeddings
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )

        for i, item in enumerate(resp.data):
            embedding = item.embedding
            global_idx = start + i

            vectors.append(
                {
                    "id": f"design-system-{global_idx}",
                    "values": embedding,
                    "metadata": {
                        "text": chunks[global_idx],
                        "source": "design-system.md",
                        "chunk_id": global_idx,
                    },
                }
            )

    return vectors

print("Считаем эмбеддинги и готовим вектора...")
vectors = build_vectors(chunks)

print(f"Отправляем {len(vectors)} векторов в Pinecone...")
upsert_response = index.upsert(vectors=vectors)

print("Готово! Документация загружена в индекс.")
print(upsert_response)
