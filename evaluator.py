# evaluator.py
from openai import AzureOpenAI
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
from tqdm import tqdm
import re

client = AzureOpenAI(
    azure_endpoint="https://francecentral.api.cognitive.microsoft.com/",
    api_key="API_KEY_HERE",
    api_version="2024-12-01-preview"
)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

JUDGE_MODEL = "gpt-4o"

def extract_statements(answer):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]

def faithfulness_score(question, contexts, answer):
    statements = extract_statements(answer)
    if not statements:
        return 1.0

    supports = []
    for stmt in statements:
        prompt = f"""
        Context:
        {chr(10).join(contexts)}

        Statement: "{stmt}"
        Question: {question}

        Does the context entail or support this statement? Answer with YES or NO only.
        """
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        verdict = response.choices[0].message.content.strip().upper()
        supports.append(1 if "YES" in verdict else 0)

    return sum(supports) / len(supports) if supports else 1.0

def hallucination_probability(answer, contexts):
    prompt = f"""
    Given the retrieved context and generated answer, assess if the answer contains hallucinated (fabricated) information.

    Context:
    {chr(10).join(contexts)}

    Answer: {answer}

    Respond in JSON:
    {{ "hallucinated": true/false, "explanation": "brief reason" }}
    """
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    try:
        result = json.loads(response.choices[0].message.content)
        return 1.0 if result.get("hallucinated", False) else 0.0
    except:
        return 0.5

def citation_grounding_score(contexts, answer):
    ctx_emb = embedder.encode(contexts, convert_to_tensor=True)
    ans_emb = embedder.encode(answer, convert_to_tensor=True)
    similarities = util.cos_sim(ans_emb, ctx_emb)[0]
    return float(similarities.max().item())

def answer_relevance(question, answer):
    q_emb = embedder.encode(question, convert_to_tensor=True)
    a_emb = embedder.encode(answer, convert_to_tensor=True)
    return float(util.cos_sim(q_emb, a_emb)[0][0].item())

def evaluate_rag_sample(sample):
    q = sample["question"]
    ctxs = sample["contexts"]
    ans = sample["answer"]

    return {
        "question": q,
        "faithfulness": round(faithfulness_score(q, ctxs, ans), 3),
        "hallucination_prob": round(hallucination_probability(ans, ctxs), 3),
        "citation_grounding": round(citation_grounding_score(ctxs, ans), 3),
        "answer_relevance": round(answer_relevance(q, ans), 3),
        "overall_score": 0
    }

def evaluate_dataset(dataset_path="sample_rag_data.jsonl"):
    with open(dataset_path) as f:
        data = [json.loads(line) for line in f]

    results = []
    for sample in tqdm(data, desc="Evaluating RAG outputs"):
        scores = evaluate_rag_sample(sample)
        # Weighted overall score
        scores["overall_score"] = round((
            0.4 * scores["faithfulness"] +
            0.3 * (1 - scores["hallucination_prob"]) +
            0.2 * scores["citation_grounding"] +
            0.1 * scores["answer_relevance"]
        ), 3)
        results.append(scores)

    return results