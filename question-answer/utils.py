from transformers import pipeline

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

def get_answer(question:str, context:str):
    """
    The get_answer function takes a question and context as input, and returns the answer string
    and score of the best answer found in the context. The function first tokenizes both question
    and context using spacy's nlp object, then passes them to pipeline.get_answer which returns 
    the answer string and score for each sentence in the document.
    
    Args:
        question:str: Store the question string
        context:str: Pass the context of the question
    
    Returns:
        A dictionary with the answer and score
    
    Doc Author:
        Ifeanyi
    """
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)
    ans = res['answer']
    scores = res['score']
    vals = {"answer": ans, "score": round(scores, 2)}
    return vals

#print(get_answer("What is the name of the first president of the United States?", "George Washington was the first president of the United States of America."))