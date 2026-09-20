from nlp_model import detect_intent

questions = [
    "How much can I save next month?",
    "Where am I spending the most?",
    "How healthy are my finances?",
    "What will I spend next month?",
    "How much will I spend in six months?",
    "What is my monthly income?",
    "Hi"
]

for question in questions:

    intent, score = detect_intent(question)

    print()
    print("Question :", question)
    print("Intent   :", intent)
    print("Score    :", round(score, 3))