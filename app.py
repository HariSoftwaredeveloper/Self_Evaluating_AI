import streamlit as st
import pandas as pd
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer, util
import json
import os

# Azure OpenAI configuration (using provided credentials)
client = AzureOpenAI(
    api_key="API_KEY_HERE",
    azure_endpoint="https://francecentral.api.cognitive.microsoft.com/",
    api_version="2024-12-01-preview"
)

DEPLOYMENT_NAME = "gpt-4o"

# Embedding model for cosine similarity (local, no API needed)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Sample dataset (as a list of dicts). This is a small synthetic dataset for demonstration.
# Each entry has a question, generated answer (from a hypothetical RAG system), and retrieved contexts.
# Includes examples with faithful answers, hallucinations, and citations for grounding checks.
dataset = [
    {
        "question": "What is the capital of Japan?",
        "answer": "The capital of Japan is Tokyo.",
        "contexts": [
            "Japan is an island country in East Asia.",
            "The capital city of Japan is Tokyo."
        ]
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "answer": "William Shakespeare wrote Romeo and Juliet in 1597.",
        "contexts": [
            "William Shakespeare was an English playwright.",
            "Romeo and Juliet is a tragedy written by William Shakespeare."
        ]
    },
    {
        "question": "What is the speed of light?",
        "answer": "The speed of light is 299,792 km/s, and it's constant in vacuum. Also, Einstein discovered it in 1905.",
        "contexts": [
            "The speed of light is approximately 300,000 km/s.",
            "Albert Einstein developed the theory of relativity where speed of light is constant."
        ]
    },
    {
        "question": "What is photosynthesis?",
        "answer": "Photosynthesis is the process by which plants make food using sunlight [1]. It produces oxygen [3].",
        "contexts": [
            "[1] Photosynthesis uses sunlight to convert CO2 and water into glucose.",
            "[2] Plants are green due to chlorophyll.",
            "[3] Oxygen is a byproduct of photosynthesis."
        ]
    },
    {
        "question": "What causes global warming?",
        "answer": "Global warming is primarily caused by human activities like burning fossil fuels, leading to increased CO2 levels. It was first noted in the 1800s [2].",
        "contexts": [
            "Global warming refers to the rise in Earth's average temperature.",
            "Human activities, such as burning fossil fuels, release greenhouse gases like CO2."
        ]
    }
]

# Prompt for LLM evaluation (combined for efficiency)
evaluation_prompt = """
You are a RAG evaluator. Use chain of thought to compute three scores for the answer given the question and contexts.
Contexts are numbered.

Contexts:
{numbered_contexts}

Question: {question}

Answer: {answer}

Step by step:
1. Break the answer into key claims or statements.
2. For each claim, determine if it is supported or entailed by any of the contexts (be strict: must be directly inferable).
3. If the answer contains citations like [1], [2], etc., verify if the cited context specifically supports the preceding claim.
4. Faithfulness: Fraction of claims that are fully supported by the contexts (0 to 1).
5. Hallucination probability: Fraction of claims that are not supported or are contradictory to the contexts (0 to 1).
6. Citation grounding: If citations are present, fraction of citations that correctly ground the claims in the cited context. If no citations, base it on overall grounding of claims to any context (similar to faithfulness, but penalize if attribution is implied but wrong). Score 0 to 1.

Output ONLY JSON:
{{
    "reasoning": "Your detailed chain of thought reasoning",
    "faithfulness": <float 0-1>,
    "hallucination_prob": <float 0-1>,
    "citation_grounding": <float 0-1>
}}
"""

def evaluate_item(item):
    question = item["question"]
    answer = item["answer"]
    contexts = item["contexts"]
    
    # Compute retrieval score: average cosine similarity between question and each context
    question_emb = embedder.encode(question)
    context_embs = embedder.encode(contexts)
    cos_sims = [util.cos_sim(question_emb, c_emb)[0][0].item() for c_emb in context_embs]
    retrieval_score = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0
    
    # Prepare numbered contexts for prompt
    numbered_contexts = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    
    # LLM evaluation
    formatted_prompt = evaluation_prompt.format(
        numbered_contexts=numbered_contexts,
        question=question,
        answer=answer
    )
    
    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[{"role": "user", "content": formatted_prompt}],
        response_format={"type": "json_object"}
    )
    
    scores_json = json.loads(response.choices[0].message.content)
    
    return {
        "question": question,
        "retrieval_score": round(retrieval_score, 2),
        "faithfulness": round(scores_json["faithfulness"], 2),
        "hallucination_prob": round(scores_json["hallucination_prob"], 2),
        "citation_grounding": round(scores_json["citation_grounding"], 2),
        "reasoning": scores_json["reasoning"]
    }

# Streamlit app
st.title("Autonomous RAG Evaluation Framework Dashboard")

st.markdown("""
This dashboard evaluates RAG outputs using LLM-based verification (Azure OpenAI GPT-4o) and cosine similarity for retrieval scoring.
- **Retrieval Score**: Average cosine similarity between question and contexts (using sentence-transformers).
- **Faithfulness**: LLM-judged fraction of answer claims supported by contexts.
- **Hallucination Probability**: LLM-judged fraction of unsupported or contradictory claims.
- **Citation Grounding**: LLM-judged score for citation accuracy (or general grounding if no citations).

The sample dataset is hardcoded for demonstration. In production, you can upload or integrate with your RAG system.
""")

if st.button("Evaluate Dataset"):
    with st.spinner("Evaluating... (This may take time due to LLM calls)"):
        results = []
        for item in dataset:
            eval_result = evaluate_item(item)
            results.append(eval_result)
        
        df = pd.DataFrame(results)
        
        st.subheader("Evaluation Results")
        st.table(df[["question", "retrieval_score", "faithfulness", "hallucination_prob", "citation_grounding"]])
        
        st.subheader("Average Scores")
        averages = df[["retrieval_score", "faithfulness", "hallucination_prob", "citation_grounding"]].mean()
        st.write(averages)
        
        st.subheader("Detailed Reasoning")
        for i, row in df.iterrows():
            with st.expander(f"Reasoning for: {row['question']}"):
                st.write(row["reasoning"])

st.markdown("""
### How to Use in VS Code
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py` (save this code as app.py)
3. Customize: Replace the dataset with your RAG outputs (e.g., load from JSON/CSV).
4. For production monitoring, integrate with your RAG pipeline to feed live data.
""")