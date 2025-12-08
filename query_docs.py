import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# 1. Загружаем ключи из .env
load_dotenv()
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# 2. Инициализируем Pinecone и OpenAI
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

client = OpenAI(api_key=OPENAI_API_KEY)

# 3. Получаем контекст из Pinecone по вопросу
def get_context(question: str, top_k: int = 3) -> str:
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
    for match in res["matches"]:
        metadata = match.get("metadata") or {}
        text = metadata.get("text", "")
        if text:
            chunks.append(text)

    return "\n\n---\n\n".join(chunks)

# 4. Формируем ответ GPT на основе контекста
def answer_question(question: str) -> str:
    context = get_context(question)

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
        model="gpt-4o-mini",  # если будет ошибка про модель, можно поменять на 'gpt-4o'
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return resp.choices[0].message.content.strip()

# 5. Простой тест при запуске файла напрямую
if __name__ == "__main__":
    question = "Какой отступ между лоудером и текстом?"
    print("Вопрос:", question)
    answer = answer_question(question)
    print("\nОтвет:\n", answer)
