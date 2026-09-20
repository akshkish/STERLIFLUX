
import sys
import os

# Allow importing project files
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from rag_engine import (
    load_documents,
    retrieve_context,
    rag_answer
)


# =========================================================
# TEST QUESTIONS
# =========================================================

TEST_CASES = [
    {
        "question": "What is inflation?",
        "expected": [
            "inflation",
            "prices"
        ]
    },
    {
        "question": "How can I save money?",
        "expected": [
            "save",
            "savings"
        ]
    },
    {
        "question": "What is a budget?",
        "expected": [
            "budget",
            "income",
            "expense"
        ]
    },
    {
        "question": "What is an emergency fund?",
        "expected": [
            "emergency",
            "fund"
        ]
    },
    {
        "question": "Why is diversification important?",
        "expected": [
            "diversification",
            "investment"
        ]
    },
    {
        "question": "How can I reduce unnecessary expenses?",
        "expected": [
            "expenses",
            "spending"
        ]
    },
    {
        "question": "What are the benefits of saving money?",
        "expected": [
            "saving",
            "money"
        ]
    },
    {
        "question": "What is debt management?",
        "expected": [
            "debt"
        ]
    }
]


# =========================================================
# HEADER
# =========================================================

print("\n========================================")
print("       STERLIFLUX RAG MODEL TEST")
print("========================================")


# =========================================================
# KNOWLEDGE BASE TEST
# =========================================================

try:
    documents = load_documents()

    print("\nKnowledge base documents :", len(documents))

    if not documents:
        print("ERROR: Knowledge base is empty.")
        sys.exit()

except Exception as e:
    print("\nERROR loading knowledge base:")
    print(type(e).__name__, "-", e)
    sys.exit()


# =========================================================
# RAG TESTING
# =========================================================

total = len(TEST_CASES)
retrieval_pass = 0
answer_pass = 0
grounded_pass = 0


for number, test in enumerate(TEST_CASES, start=1):

    question = test["question"]
    expected_words = test["expected"]

    print("\n----------------------------------------")
    print(f"Test {number}")
    print("----------------------------------------")

    print("Question:", question)

    # -------------------------------------
    # Retrieval
    # -------------------------------------

    try:
        retrieved = retrieve_context(question, top_k=3)

        if retrieved:
            retrieval_pass += 1

            print("\nRetrieved Context:")

            for i, item in enumerate(retrieved, start=1):
                print(
                    f"\n{i}. Score: {item['score']:.4f}"
                )
                print(item["text"][:500])

        else:
            print("\nRetrieval: FAILED")

    except Exception as e:
        print("\nRetrieval ERROR:")
        print(type(e).__name__, "-", e)
        continue


    # -------------------------------------
    # Answer Generation
    # -------------------------------------

    try:
        answer = rag_answer(question)

        if answer:
            print("\nGenerated Answer:")
            print(answer)

            answer_lower = answer.lower()

            # Check whether at least one expected
            # concept appears in the answer.
            matches = sum(
                word.lower() in answer_lower
                for word in expected_words
            )

            if matches >= 1:
                answer_pass += 1
                print("\nAnswer Relevance: PASS")
            else:
                print("\nAnswer Relevance: FAIL")

            # ---------------------------------
            # Grounding check
            # ---------------------------------

            context_text = " ".join(
                item["text"].lower()
                for item in retrieved
            )

            context_words = set(
                context_text.split()
            )

            answer_words = set(
                answer_lower.split()
            )

            if answer_words:
                overlap = (
                    len(answer_words & context_words)
                    / len(answer_words)
                )
            else:
                overlap = 0

            print(
                f"Context overlap: {overlap * 100:.2f}%"
            )

            if overlap >= 20:
                grounded_pass += 1
                print("Groundedness: PASS")
            else:
                print("Groundedness: REVIEW")

        else:
            print("\nAnswer Generation: FAILED")

    except Exception as e:
        print("\nGeneration ERROR:")
        print(type(e).__name__, "-", e)


# =========================================================
# FINAL RESULTS
# =========================================================

print("\n========================================")
print("          RAG TEST RESULTS")
print("========================================")

print(
    f"\nRetrieval success : "
    f"{retrieval_pass}/{total} "
    f"({retrieval_pass / total * 100:.2f}%)"
)

print(
    f"Answer relevance : "
    f"{answer_pass}/{total} "
    f"({answer_pass / total * 100:.2f}%)"
)

print(
    f"Grounded answers : "
    f"{grounded_pass}/{total} "
    f"({grounded_pass / total * 100:.2f}%)"
)

overall = (
    retrieval_pass
    + answer_pass
    + grounded_pass
) / (total * 3) * 100

print(
    f"\nOverall RAG score : "
    f"{overall:.2f}%"
)

print("\n========================================")
print("             TEST COMPLETE")
print("========================================")

