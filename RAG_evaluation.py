from sklearn.metrics import precision_score, recall_score, f1_score
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import random

import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


nltk.download('punkt')  # Required for BLEU score calculation



def evaluate_retrieval(query, retriever):
    """
    Evaluate the retrieval step of the RAG pipeline with precision, recall, and F1 score.

    Returns:
    - dict: Retrieval evaluation metrics.
    """
    try:
        # Mock ground-truth documents for demonstration (replace with actual ground truth)
        ground_truth_docs = ["The methodology for rainfall prediction involves data collection, data cleaning, data analysis, preprocessing, feature scaling, feature selection, data splitting, model training, and prediction. Ten different regression models, including Regression, Polynomial Regression, and Lasso Regression, are trained and compared. The goal is to determine the best-performing model for rainfall prediction."]  
        
        retrieved_docs = retriever.get_relevant_documents(query)
        # Compare ground-truth with retrieved docs
        retrieved_texts = [doc.page_content for doc in retrieved_docs]
        
        def similarity(text1, text2):
            return SequenceMatcher(None, text1, text2).ratio()
        similarity_threshold = 0.7  # Consider a retrieved doc relevant if it has at least 70% similarity
        
        matches = [any(similarity(gt, ret) > similarity_threshold for ret in retrieved_texts) for gt in ground_truth_docs]

        # matches = [doc in retrieved_texts for doc in ground_truth_docs]   # exact matching
        
        precision = sum(matches) / len(retrieved_texts) if retrieved_texts else 0
        recall = sum(matches) / len(ground_truth_docs) if ground_truth_docs else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

        # return {
        #     "retrieval_status": "Success",
        #     "precision": round(precision, 2),
        #     "recall": round(recall, 2),
        #     "f1_score": round(f1, 2),
        #     "retrieved_docs_count": len(retrieved_docs),
        #     "retrieved_docs_preview": [doc.page_content[:100] for doc in retrieved_docs],
        # }
        
        evaluation_result = {
            "retrieval_status": "Success",
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1, 2),
            "retrieved_docs_count": len(retrieved_docs),
            "retrieved_docs_preview": [doc.page_content[:100] for doc in retrieved_docs],
        }
        logging.info(f"Retrieval Evaluation: {evaluation_result}")
        return evaluation_result
    except Exception as e:
        logging.error(f"Error in retrieval evaluation: {str(e)}")
        return {"retrieval_status": "Error", "error": str(e)}


def evaluate_generation(query, response):
    """
    Evaluate the generation step of the RAG pipeline using BLEU and ROUGE scores.

    Returns:
    - dict: Generation evaluation metrics.
    """
    try:
        generated_answer = response.get("answer", "No response generated.")
        # Mock reference answer for demonstration (replace with actual reference answer)
        reference_answer = "The methodology for rainfall prediction involves data analysis, feature engineering, and regression model training."
        
        # Calculate BLEU score
        reference_tokens = nltk.word_tokenize(reference_answer.lower())
        generated_tokens = nltk.word_tokenize(generated_answer.lower())
        bleu_score = sentence_bleu([reference_tokens], generated_tokens)

        # Calculate ROUGE score
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference_answer, generated_answer)

        # return {
        #     "generation_status": "Success",
        #     "bleu_score": round(bleu_score, 2),
        #     "rouge1_score": round(rouge_scores['rouge1'].fmeasure, 2),
        #     "rouge2_score": round(rouge_scores['rouge2'].fmeasure, 2),
        #     "rougeL_score": round(rouge_scores['rougeL'].fmeasure, 2),
        # }
        
        evaluation_result = {
            "generation_status": "Success",
            "bleu_score": round(bleu_score, 2),
            "rouge1_score": round(rouge_scores['rouge1'].fmeasure, 2),
            "rouge2_score": round(rouge_scores['rouge2'].fmeasure, 2),
            "rougeL_score": round(rouge_scores['rougeL'].fmeasure, 2),
        }
        
        logging.info(f"Generation Evaluation: {evaluation_result}")
        return evaluation_result
    except Exception as e:
        logging.error(f"Error in generation evaluation: {str(e)}")
        return {"generation_status": "Error", "error": str(e)}


def evaluate_hallucination(query, response, retrieved_docs):
    """Detect hallucinations by comparing the generated answer with retrieved documents."""
    try:
        generated_answer = response.get("answer", "").lower()
        retrieved_texts = [doc.page_content.lower() for doc in retrieved_docs]

        if not retrieved_texts:
            return {"hallucination_status": "Error", "error": "No retrieved documents available."}

        # Compute cosine similarity
        vectorizer = TfidfVectorizer().fit_transform([generated_answer] + retrieved_texts)
        similarity_scores = cosine_similarity(vectorizer[0], vectorizer[1:]).flatten()
        max_similarity = max(similarity_scores) if similarity_scores.size > 0 else 0

        hallucination_detected = max_similarity < 0.5  # If similarity is low, hallucination is likely
        confidence_score = round(max_similarity, 2)

        evaluation_result = {
            "hallucination_status": "Success",
            "hallucination_detected": hallucination_detected,
            "confidence_score": confidence_score,
        }

        logging.info(f"Hallucination Evaluation: {evaluation_result}")
        return evaluation_result

    except Exception as e:
        logging.error(f"Error in hallucination evaluation: {str(e)}")
        return {"hallucination_status": "Error", "error": str(e)}