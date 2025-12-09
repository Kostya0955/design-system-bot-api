import os
import glob
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# 1. Загружаем переменные окружения
load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

client = OpenAI(api_key=OPENAI_API_KEY)


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


def extract_component_name(text: str, filename: str) -> str:
    # пробуем взять первую строку вида "# Название"
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    # запасной вариант — имя файла
    base = os.path.splitext(os.path.basename(filename))[0]
    return base.capitalize()


def build_vectors(chunks_with_meta, batch_size: int = 32):
    vectors = []

    # chunks_with_meta: список словарей {text, source, component}
    texts = [c["text"] for c in chunks_with_meta]

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )

        for i, item in enumerate(resp.data):
            embedding = item.embedding
            global_idx = start + i
            meta = chunks_with_meta[global_idx]

            vectors.append(
                {
                    "id": f"{meta['source']}-{meta['chunk_id']}",
                    "values": embedding,
                    "metadata": {
                        "text": meta["text"],
                        "source": meta["source"],
                        "chunk_id": meta["chunk_id"],
                        "component": meta["component"],
                    },
                }
            )

    return vectors


if __name__ == "__main__":
    # 1. собираем все md-файлы
    md_files = glob.glob("*.md")
    print("Нашли md-файлы:", md_files)

    all_chunks_with_meta = []
    chunk_counter = 0

    for path in md_files:
        with open(path, "r", encoding="utf-8") as f:
            full_text = f.read()

        component_name = extract_component_name(full_text, path)
        chunks = split_text(full_text, max_chars=600)

        print(f"{path}: нарезали на {len(chunks)} кусочков, компонент: {component_name}")

        for i, chunk in enumerate(chunks):
            all_chunks_with_meta.append(
                {
                    "text": chunk,
                    "source": os.path.basename(path),
                    "chunk_id": i,
                    "component": component_name,
                }
            )
            chunk_counter += 1

    print(f"Всего кусочков по всем файлам: {chunk_counter}")

    print("Считаем эмбеддинги и готовим вектора...")
    vectors = build_vectors(all_chunks_with_meta)

    print(f"Отправляем {len(vectors)} векторов в Pinecone...")
    upsert_response = index.upsert(vectors=vectors)

    print("Готово! Документация загружена в индекс.")
    print(upsert_response)
