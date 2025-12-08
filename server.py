from fastapi import FastAPI
from pydantic import BaseModel
from query_docs import answer_question

app = FastAPI(title="Design System Docs API")


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {"status": "ok", "message": "Design System Docs API работает"}


@app.post("/ask")
def ask(q: Question):
    """
    Принимает JSON вида {"question": "текст вопроса"}
    и возвращает {"answer": "...ответ..."}.
    """
    answer = answer_question(q.question)
    return {"answer": answer}
