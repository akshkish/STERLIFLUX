# SterliFlux – AI-Engineered Expense Predictor.

**SterliFlux** is an AI-powered personal finance analysis platform that helps users understand their spending behaviour, forecast future expenses, and identify potential savings through Machine Learning, Natural Language Processing (NLP), and Retrieval-Augmented Generation (RAG).

The application transforms raw financial transaction data into meaningful insights using predictive analytics, intelligent processing, and interactive visualizations.

## Key Features

### 1. AI-Powered Expense Analysis
- Upload financial transaction data through CSV files.
- Automatically analyze spending patterns and transaction history.
- Generate category-wise expense insights.
- Identify major spending categories and unusual spending behaviour.

### 2. Machine Learning Predictions
- **Random Forest Regression:** Predicts future expenses and estimates potential savings based on historical spending patterns.
- **Random Forest Classification:** Supports expense category prediction.
- Uses Scikit-learn, Pandas, and NumPy for model development and data processing.
- Generates financial forecasts for different time periods.

### 3. NLP and RAG-Based Intelligence
- Integrates **Hugging Face Transformers** for NLP processing.
- Implements a **Retrieval-Augmented Generation (RAG) engine** for contextual information retrieval and intelligent financial explanations.
- Enables contextual interpretation of financial information.
- Combines retrieved context with language-model-based response generation.

### 4. Interactive Financial Dashboard
- Visualizes spending patterns using Plotly.
- Displays total expenses, average spending, and highest transactions.
- Provides category-wise financial breakdowns.
- Presents future expense and savings projections.

### 5. Virtual Savings Lock
- A dedicated financial savings feature implemented using Flask and MySQL.
- Supports the application's savings management functionality.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | HTML5, CSS3, Bootstrap, JavaScript |
| Backend | Flask |
| Database | MySQL |
| Machine Learning | Scikit-learn |
| ML Algorithms | Random Forest Regression, Random Forest Classification |
| NLP | Hugging Face Transformers |
| AI Architecture | Retrieval-Augmented Generation (RAG) |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Development Tools | VS Code, Git, GitHub |

## System Architecture

```text
              USER
                |
                v
       CSV Transaction Upload
                |
                v
       FLASK BACKEND
                |
       +--------+---------+
       |                  |
       v                  v
  DATA PROCESSING     MYSQL DATABASE
  Pandas / NumPy      Savings Data
       |
       v
   MACHINE LEARNING
       |
  +----+----------------+
  |                     |
  v                     v
Random Forest       Random Forest
Regression          Classification
  |                     |
  v                     v
Expense Forecast    Category Prediction
  |                     |
  +----------+----------+
             |
             v
       NLP + RAG ENGINE
       Hugging Face
       Transformers
             |
             v
     Contextual Insights
             |
             v
     PLOTLY DASHBOARD
             |
             v
       USER INSIGHTS
```

## Project Structure

```text
SterliFlux/
│
├── app.py
├── requirements.txt
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── ...
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── models/
│   └── ...
│
├── uploads/
│
└── README.md
```

*Note: Adjust the structure above to match the actual files and folders in your repository.*

## Installation and Setup

### Prerequisites
- Python 3.10+
- MySQL
- Git
- VS Code or another Python IDE

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd SterliFlux
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure MySQL

- Start your MySQL server.
- Create the required database.
- Configure database connection credentials using environment variables.
- Ensure the required tables are initialized.

### 5. Run the Application

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000/
```

## Machine Learning Approach

SterliFlux uses supervised machine learning techniques to support personal financial analysis.

| Algorithm | Application |
|---|---|
| Random Forest Regressor | Expense forecasting and savings prediction |
| Random Forest Classifier | Expense category prediction |

Historical transaction records are processed and transformed into suitable model inputs to generate predictions.

## Project Objectives

- Simplify personal expense tracking.
- Apply machine learning to financial forecasting.
- Use NLP and RAG for contextual financial insights.
- Convert transaction data into understandable visual reports.
- Encourage better financial planning and savings habits.

## Future Enhancements

- Advanced conversational financial assistant.
- Improved personalized savings recommendations.
- Real-time financial data integration.
- Enhanced anomaly detection.
- Improved model performance through additional financial datasets.
- Secure user authentication and personalized dashboards.

## Author

**Akshaya Kishore**

Master of Computer Applications (MCA)

Interests: Artificial Intelligence, Machine Learning, NLP, Generative AI, and Full-Stack Development.

---

*SterliFlux – Transforming financial data into intelligent financial insights.*
