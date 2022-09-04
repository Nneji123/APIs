from transformers import pipeline

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)


def get_answer(question: str, context: str):
    QA_input = {"question": question, "context": context}
    res = nlp(QA_input)
    ans = res["answer"]
    scores = res["score"]
    vals = {"answer": ans, "score": round(scores, 2)}
    return vals
