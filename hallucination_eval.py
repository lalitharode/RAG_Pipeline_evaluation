from langchain.evaluation import load_evaluator

def evaluate_hallucination(query, response):
    evaluator = load_evaluator("hallucination")
    results = evaluator.evaluate(
        predictions=[{"query": query, "answer": response}],
        references=[{"query": query, "answer": "Expected true facts"}]
    )
    return results
