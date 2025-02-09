from langchain.evaluation import load_evaluator

def evaluate_retrieval(query, retriever):
    evaluator = load_evaluator("retrieval")
    results = evaluator.evaluate(
        predictions=[{"query": query, "context": retriever.get_relevant_documents(query)}],
        references=[{"query": query, "context": ["Expected context here"]}]
    )
    return results
