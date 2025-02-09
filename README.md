Evaluation metrics for **RAG (Retrieval-Augmented Generation)** systems assess the quality of both the retrieval and generation components. Here's a breakdown of the key metrics used for evaluating RAG systems:

---

### 1. **Retrieval Metrics**
These evaluate the performance of the retriever in fetching relevant documents from the knowledge base.

#### a. **Precision**
- Measures the proportion of retrieved documents that are relevant.
- Formula:  
  
  Precision = {Number of Relevant Retrieved Documents}/{Total Number of Retrieved Documents}
  

#### b. **Recall**
- Measures the proportion of relevant documents that are retrieved out of all relevant documents.
- Formula:  
  
  Recall = {Number of Relevant Retrieved Documents}/{Total Number of Relevant Documents in the Dataset}
  

#### c. **F1-Score**
- The harmonic mean of precision and recall.
- Formula:  
  
  F1 = 2 *{{Precision} *{Recall}}/{{Precision} + {Recall}}
  

#### d. **Mean Reciprocal Rank (MRR)**
- Evaluates the ranking quality of the retriever by checking the position of the first relevant document in the ranked results.
- Formula:  
  
  {MRR} = {1}{|Q|} +{i=1}^{|Q|} /{1}{{Rank of First Relevant Document in Query } i}
  

#### e. **Normalized Discounted Cumulative Gain (nDCG)**
- Evaluates ranking quality by considering the relevance scores and positions of retrieved documents.

#### f. **Coverage**
- Measures whether all relevant documents are covered in the retrieved set.

---

### 2. **Generation Metrics**
These assess the quality of the responses generated by the language model.

#### a. **BLEU (Bilingual Evaluation Understudy Score)**
- Measures the overlap between generated and reference text.
- Commonly used for machine translation but applicable to text generation.

#### b. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
- Measures the overlap between the generated text and the reference text based on:
  - **ROUGE-1**: Overlap of unigrams.
  - **ROUGE-2**: Overlap of bigrams.
  - **ROUGE-L**: Longest common subsequence.

#### c. **Perplexity**
- Evaluates the fluency of the generated text by measuring how well the model predicts the next word.

#### d. **Answer Similarity**
- Measures semantic similarity between the generated answer and the reference answer, often using:
  - **Cosine Similarity**
  - **Sentence Transformers**
  - **Embedding Similarity** (e.g., using tools like `Sentence-BERT`).

#### e. **Toxicity**
- Measures whether the generated response contains offensive or toxic content.
  - Tools: Perspective API, OpenAI's moderation endpoint.

---

### 3. **Hallucination Metrics**
These assess whether the generated response contains false or unsupported information.

#### a. **Hallucination Rate**
- Measures the percentage of generated content that is factually incorrect or not grounded in the retrieved documents.

#### b. **Attribution Score**
- Evaluates whether the generated response can be traced back to the retrieved documents.

#### c. **Grounding Percentage**
- Measures the proportion of generated content that directly uses retrieved information.

#### d. **Confidence Score**
- A score representing the likelihood that the response is factually accurate.

---

### 4. **End-to-End Metrics**
These combine the evaluation of retrieval and generation components.

#### a. **Answer Accuracy**
- Measures whether the final answer aligns with a ground truth answer.

#### b. **Query Success Rate**
- The percentage of queries for which the RAG system provides a correct response.

#### c. **Human Evaluation**
- Human judges evaluate the responses on:
  - **Relevance**
  - **Fluency**
  - **Completeness**
  - **Conciseness**

---

### Tools and Frameworks for RAG Evaluation
1. **Hugging Face Evaluation Metrics** (e.g., BLEU, ROUGE).
2. **Google's Perspective API** (for toxicity detection).
3. **MLFlow Custom Metrics**.
4. **LangChain** for pipeline integration.

These metrics provide a comprehensive view of the system's performance, helping to identify weak spots and improve the RAG pipeline.


---

### **Type of RAG Evaluation Selected and Reasons behind them**  
  
For evaluating the **Retrieval-Augmented Generation (RAG) pipeline**, we selected the following three types of evaluations:  

### **1. Retrieval Evaluation**  
- **Metrics Used:** Precision, Recall, and F1-score  
- **Reasoning:**  
  - We need to assess how well the retriever fetches relevant documents from the vector database.  
  - Precision tells us how many retrieved documents are relevant.  
  - Recall measures how many relevant documents were retrieved out of all possible relevant ones.  
  - F1-score balances precision and recall.  
  - We use text similarity (SequenceMatcher) with a threshold to determine relevance.  

### **2. Generation Evaluation**  
- **Metrics Used:** BLEU Score, ROUGE Scores (ROUGE-1, ROUGE-2, ROUGE-L)  
- **Reasoning:**  
  - BLEU Score helps measure the similarity between the generated response and a reference answer based on token overlap.  
  - ROUGE-1 and ROUGE-2 capture unigram and bigram overlaps, helping to evaluate the fluency and coherence of the generated text.  
  - ROUGE-L measures the longest common subsequence, useful for evaluating long-form responses.  

### **3. Hallucination Evaluation**  
- **Metrics Used:** Cosine Similarity, Confidence Score  
- **Reasoning:**  
  - Hallucination detection ensures the model's response is grounded in the retrieved documents.  
  - We use **TF-IDF and cosine similarity** to measure how closely the generated answer aligns with the retrieved context.  
  - If similarity is below a threshold (0.5), we flag potential hallucinations.  

#### **Overall Justification:**  
These three evaluations ensure that:  
   **Retrieval quality is high** (precision/recall).  
   **Generated responses are relevant and coherent** (BLEU/ROUGE).  
   **The model is not fabricating information** (hallucination detection).  

This makes the RAG system **accurate, reliable, and factual** in answering queries based on retrieved knowledge.
