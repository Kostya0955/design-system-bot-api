from openai import OpenAI
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Pinecone client + index
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])


def get_context(question: str, top_k: int = 3):
    # считаем эмбеддинг вопроса
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[question],
    )
    vector = emb.data[0].embedding

    # ищем похожие кусочки в Pinecone
    res = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
    )

    chunks = []
    components = []

    for match in res["matches"]:
        metadata = match.get("metadata") or {}
        text = metadata.get("text", "")
        component = metadata.get("component")

        if text:
            chunks.append(text)
        if component:
            components.append(component)

    context_text = "\n\n---\n\n".join(chunks)
    main_component = components[0] if components else None

    return context_text, main_component


def answer_question(question: str) -> str:
    context, main_component = get_context(question)

    system_prompt = (
        "Ты ассистент по документации дизайн-системы. "
        "Отвечай КРАТКО и строго на основе контекста. "
        "Если в контексте нет точного ответа, честно скажи, "
        "что в документации это не описано, и не выдумывай значения."
    )

    user_prompt = (
        f"Вопрос:\n{question}\n\n"
        f"Контекст из документации:\n{context}"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer_text = resp.choices[0].message.content.strip()

    # добавляем название компонента в начало ответа
    if main_component:
        return f"{main_component}\n\n{answer_text}"
    else:
        return answer_text
