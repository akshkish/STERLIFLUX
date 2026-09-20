from sentence_transformers import SentenceTransformer, util

# Load Hugging Face model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# SterliFlux chatbot intents
INTENTS = {

    "savings": [
        "How much can I save?",
        "How much can I save next month?",
        "What will my savings be?",
        "How much money will I have left?",
        "Can I save money next month?",
        "How much will I have left after expenses?",
        "What will I save this month?",
        "How much can I put into savings?"
    ],

    "next_month_expense": [
        "How much will I spend next month?",
        "What will my expenses be next month?",
        "What is my predicted expense?",
        "How much am I expected to spend?",
        "How much am I likely to spend next month?",
        "What will I spend next month?"
    ],

    "financial_health": [
        "How is my financial health?",
        "Am I financially healthy?",
        "What is my financial health score?",
        "How healthy are my finances?",
        "What is my health score?",
        "What is my financial score?",
        "Am I doing well financially?"
    ],

    "spending_category": [
        "Where am I spending the most?",
        "What is my biggest spending category?",
        "Which category costs me the most?",
        "Where does most of my money go?",
        "What do I spend the most money on?",
        "Which category has the highest spending?"
    ],

    "reduce_category": [
        "Which category should I reduce?",
        "What category should I cut down?",
        "Where should I reduce my spending?",
        "Which expenses should I reduce?",
        "Which spending category should I cut?",
        "Where can I cut my expenses?",
        "What should I spend less on?",
        "Which category can I reduce?"
    ],

    "spending_risk": [
        "Am I spending too much?",
        "Am I spending more than I should?",
        "Is my spending too high?",
        "Am I overspending?",
        "Am I spending beyond my income?",
        "Is my spending under control?",
        "Is my spending healthy?",
        "Do I spend too much money?",
        "Should I reduce my spending?"
    ],

    "affordability": [
        "Can I afford a house?",
        "Can I afford a house next month?",
        "Can I afford to buy a house?",
        "Can I afford a car?",
        "Can I afford a major purchase?",
        "Can I afford this purchase?",
        "Can I afford something expensive?",
        "Can my finances support a big purchase?"
    ],

    "purchase_affordability": [
        "Can I afford a 50000 purchase?",
        "Can I afford a ₹50000 purchase?",
        "Can I afford to spend 50000?",
        "Can I afford this amount?",
        "Can I afford to spend this much?",
        "Can I make a large purchase?",
        "Do I have enough savings for a purchase?",
        "Can I afford a big expense?"
    ],

    "investment": [
        "What can I invest in?",
        "Where should I invest my money?",
        "How can I invest my savings?",
        "What should I invest in?",
        "Where can I put my savings?",
        "How should I use my savings?",
        "What investment options do I have?",
        "Can I invest my savings?"
    ],

    "inflation": [
        "How does inflation affect my expenses?",
        "What is the inflation rate?",
        "Will inflation increase my expenses?",
        "Tell me about inflation",
        "What is inflation?",
        "How will inflation affect my spending?",
        "How does inflation affect my savings?"
    ],

    "six_month_forecast": [
        "How much will I spend in six months?",
        "What are my six month expenses?",
        "Show me my six month forecast",
        "What will my expenses be over six months?",
        "How much am I expected to spend in six months?"
    ],

    "one_year_forecast": [
        "How much will I spend in one year?",
        "What are my yearly expenses?",
        "Show me my one year forecast",
        "What will my expenses look like next year?",
        "How much will I spend next year?",
        "What are my annual expenses?",
        "What will my expenses be in one year?"
    ],

    "income": [
        "What is my income?",
        "How much do I earn?",
        "What monthly income did I enter?",
        "What is my monthly income?",
        "How much money do I make?"
    ],

    "greeting": [
        "Hello",
        "Hi",
        "Hey",
        "Hello chatbot",
        "Hi there",
        "Good morning",
        "Good afternoon",
        "Good evening",
        "Hello SterliFlux"
    ],

    "thanks": [
        "Thank you",
        "Thanks",
        "Thank you so much",
        "Thanks a lot",
        "I appreciate it",
        "That's helpful",
        "That was helpful"
    ],

    "acknowledgement": [
        "Okay",
        "Ok",
        "Alright",
        "Got it",
        "I understand",
        "Understood",
        "Fine"
    ]
}


# Create embeddings for all example sentences
intent_embeddings = {}

for intent, examples in INTENTS.items():
    intent_embeddings[intent] = model.encode(
        examples,
        convert_to_tensor=True
    )


def detect_intent(question):
    """Find the most semantically similar SterliFlux intent."""

    question_embedding = model.encode(
        question,
        convert_to_tensor=True
    )

    best_intent = None
    best_score = -1

    for intent, embeddings in intent_embeddings.items():

        scores = util.cos_sim(
            question_embedding,
            embeddings
        )[0]

        score = float(scores.max())

        if score > best_score:
            best_score = score
            best_intent = intent

    return best_intent, best_score