
import os

import re

import faiss

import numpy as np

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# =========================================================
# RAG CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

KNOWLEDGE_FILE = os.path.join(
    BASE_DIR,
    "knowledge_base",
    "financial_knowledge.txt"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

GENERATION_MODEL = (
    "google/flan-t5-base"
)

TOP_K = 3


# =========================================================
# MODEL LOADING
# =========================================================

_embedding_model = None

_generator_tokenizer = None

_generator_model = None

_documents = None

_index = None


def load_models():

    global _embedding_model

    global _generator_tokenizer

    global _generator_model

    if _embedding_model is None:

        _embedding_model = (
            SentenceTransformer(
                EMBEDDING_MODEL
            )
        )

    if _generator_model is None:

        _generator_tokenizer = (
            AutoTokenizer.from_pretrained(
                GENERATION_MODEL
            )
        )

        _generator_model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                GENERATION_MODEL
            )
        )


# =========================================================
# KNOWLEDGE BASE
# =========================================================

def load_documents():

    global _documents

    if _documents is not None:

        return _documents

    if not os.path.exists(
        KNOWLEDGE_FILE
    ):

        return []

    with open(
        KNOWLEDGE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    # Split knowledge into paragraphs.

    paragraphs = re.split(
         r"\n\s*\n",
        text
    )

    _documents = [

        paragraph.strip()

        for paragraph in paragraphs

        if paragraph.strip()

    ]

    return _documents


# =========================================================
# BUILD VECTOR INDEX
# =========================================================

def build_index():

    global _index

    load_models()

    documents = load_documents()

    if not documents:

        return None

    embeddings = (
        _embedding_model.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    )

    dimension = embeddings.shape[1]

    _index = faiss.IndexFlatIP(
        dimension
    )

    _index.add(
        embeddings.astype(
            "float32"
        )
    )

    return _index


# =========================================================
# RETRIEVE RELEVANT KNOWLEDGE
# =========================================================

def retrieve_context(
    question,
    top_k=TOP_K
):

    global _index

    documents = load_documents()

    if not documents:

        return []

    if _index is None:

        build_index()

    if _index is None:

        return []

    question_embedding = (
        _embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    )

    scores, indices = (
        _index.search(
            question_embedding.astype(
                "float32"
            ),
            min(
                top_k,
                len(documents)
            )
        )
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index < 0:

            continue

        results.append(
            {
                "text": documents[index],
                "score": float(score),
            }
        )

    return results


# =========================================================
# RAG ANSWER GENERATION
# =========================================================

def rag_answer(
    question
):

    load_models()

    retrieved = retrieve_context(
        question
    )

    if not retrieved:

        return None

    context = "\n\n".join(
        item["text"]
        for item in retrieved
    )

    prompt = (
        "Answer the user's financial education "
        "question using only the provided context. "
        "Do not invent facts. "
        "If the context does not contain enough "
        "information, say that the information is "
        "not available in the SterliFlux knowledge base. "
        "Keep the answer clear and concise.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    inputs = _generator_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = _generator_model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False
    )

    answer = _generator_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    return answer or None

