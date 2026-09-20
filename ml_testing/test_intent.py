import sys
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Allow importing nlp_model.py from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp_model import detect_intent


# ==========================================
# TEST DATA
# New sentences not directly copied from
# the training examples in nlp_model.py
# ==========================================

TEST_DATA = [

    # savings
    ("How much money should I be able to save?", "savings"),
    ("What amount will remain after I pay my expenses?", "savings"),
    ("Will I have money left over this month?", "savings"),

    # next month expense
    ("Give me an estimate of my spending for next month.", "next_month_expense"),
    ("What expense amount should I expect next month?", "next_month_expense"),
    ("How much money am I likely to spend?", "next_month_expense"),

    # financial health
    ("Can you tell me whether my finances are healthy?", "financial_health"),
    ("How would you rate my current financial situation?", "financial_health"),
    ("What does my financial health rating look like?", "financial_health"),

    # spending category
    ("Which type of expense takes the most money?", "spending_category"),
    ("What area of spending is the highest?", "spending_category"),
    ("Which expense category is costing me the most?", "spending_category"),

    # reduce category
    ("Where should I decrease my spending?", "reduce_category"),
    ("Which expenses could I cut back on?", "reduce_category"),
    ("What spending area should I reduce?", "reduce_category"),

    # spending risk
    ("Is my current spending excessive?", "spending_risk"),
    ("Am I using too much of my income?", "spending_risk"),
    ("Should I be worried about how much I spend?", "spending_risk"),

    # affordability
    ("Would buying a vehicle be financially possible for me?", "affordability"),
    ("Could my finances handle buying a house?", "affordability"),
    ("Is a large purchase financially realistic for me?", "affordability"),

    # purchase affordability
    ("Do I have enough money to spend 75000?", "purchase_affordability"),
    ("Can my budget handle a purchase of 60000?", "purchase_affordability"),
    ("Would spending 100000 be affordable for me?", "purchase_affordability"),

    # investment
    ("Where would be a good place to put my savings?", "investment"),
    ("What are some ways I can invest my money?", "investment"),
    ("How could I make use of my savings through investing?", "investment"),

    # inflation
    ("Could rising prices increase what I spend?", "inflation"),
    ("Why are my expenses affected by inflation?", "inflation"),
    ("Will inflation make my future spending higher?", "inflation"),

    # six month forecast
    ("What spending should I expect during the next six months?", "six_month_forecast"),
    ("Can you predict my expenses for half a year?", "six_month_forecast"),
    ("How much might I spend over six months?", "six_month_forecast"),

    # one year forecast
    ("What will my expenses be during the coming year?", "one_year_forecast"),
    ("Can you estimate my spending for the next twelve months?", "one_year_forecast"),
    ("What should I expect to spend annually?", "one_year_forecast"),

    # income
    ("How much money do I receive every month?", "income"),
    ("What earnings did I provide to the system?", "income"),
    ("Tell me the monthly income in my profile.", "income"),

    # greeting
    ("Hi SterliFlux", "greeting"),
    ("Good day", "greeting"),
    ("Hey there", "greeting"),

    # thanks
    ("I really appreciate your help.", "thanks"),
    ("Thanks for explaining that.", "thanks"),
    ("That was very useful.", "thanks"),

    # acknowledgement
    ("Alright, I got it.", "acknowledgement"),
    ("Okay, I understand.", "acknowledgement"),
    ("Fine, understood.", "acknowledgement"),
]


# ==========================================
# RUN TEST
# ==========================================

print("\n========================================")
print("     STERLIFLUX INTENT MODEL TEST")
print("========================================")

y_true = []
y_pred = []

correct = 0

print("\nIndividual Predictions")
print("----------------------------------------")

for question, expected in TEST_DATA:

    predicted, score = detect_intent(question)

    y_true.append(expected)
    y_pred.append(predicted)

    result = "PASS" if predicted == expected else "FAIL"

    if predicted == expected:
        correct += 1

    print(
        f"\n[{result}]"
        f"\nQuestion : {question}"
        f"\nExpected : {expected}"
        f"\nPredicted: {predicted}"
        f"\nScore    : {score:.4f}"
    )


# ==========================================
# METRICS
# ==========================================

accuracy = accuracy_score(y_true, y_pred)

print("\n========================================")
print("          INTENT PERFORMANCE")
print("========================================")

print(f"\nTotal test cases : {len(TEST_DATA)}")
print(f"Correct          : {correct}")
print(f"Incorrect        : {len(TEST_DATA) - correct}")
print(f"Accuracy         : {accuracy:.4f}")
print(f"Accuracy (%)     : {accuracy * 100:.2f}%")


print("\n========================================")
print("       PRECISION / RECALL / F1")
print("========================================")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

labels = sorted(set(y_true))

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

print("\n========================================")
print("          CONFUSION MATRIX")
print("========================================")

print("\nLabels:")
print(labels)

print("\nMatrix:")
print(cm)

print("\n========================================")
print("          INTENT TEST COMPLETE")
print("========================================")