from langchain.evaluation import LLMQAEvaluator
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

def evaluate_generation(query, response):
    evaluator = LLMQAEvaluator(llm=llm)
    results = evaluator.evaluate(
        predictions=[{"query": query, "answer": response}],
        references=[{"query": query, "answer": "Expected answer here"}]
    )
    return results
