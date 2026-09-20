from flask import (
    Flask,
    render_template,
    request,
    session,
    jsonify,
    redirect,
    url_for,
    flash
)

import pandas as pd
import os
import re

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)

import plotly.express as px

# =========================================================
# AUTHENTICATION IMPORTS
# =========================================================

from flask_sqlalchemy import SQLAlchemy

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from datetime import datetime


# =========================================================
# APP CONFIGURATION
# =========================================================

app = Flask(__name__)

app.secret_key = "sterliflux_secret_key"

# =========================================================
# DATABASE CONFIGURATION
# =========================================================

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///sterliflux.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# =========================================================
# LOGIN MANAGER
# =========================================================

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"

login_manager.login_message = "Please login to access SterliFlux."


# =========================================================
# UPLOAD CONFIGURATION
# =========================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# =========================================================
# USER MODEL
# =========================================================

class User(UserMixin, db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    username = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    email = db.Column(
        db.String(150),
        unique=True,
        nullable=False
    )

    password_hash = db.Column(
        db.String(255),
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

    def set_password(self, password):

        self.password_hash = generate_password_hash(
            password
        )

    def check_password(self, password):

        return check_password_hash(
            self.password_hash,
            password
        )


# =========================================================
# LOGIN USER LOADER
# =========================================================

@login_manager.user_loader
def load_user(user_id):

    return db.session.get(
        User,
        int(user_id)
    )


# =========================================================
# CREATE DATABASE
# =========================================================

with app.app_context():

    db.create_all()


# =========================================================
# CATEGORY INFLATION RATES
# =========================================================

CATEGORY_INFLATION_RATES = {

    "food": 7.0,
    "groceries": 7.0,
    "transport": 6.0,
    "travel": 6.0,
    "shopping": 5.0,
    "clothing": 5.0,
    "housing": 6.0,
    "rent": 6.0,
    "healthcare": 5.0,
    "medical": 5.0,
    "education": 5.0,
    "entertainment": 4.0,
    "utilities": 5.0,
    "bills": 5.0,
    "electronics": 4.0,
    "other": 5.0
}

DEFAULT_INFLATION_RATE = 5.0


def get_inflation_rate(category):

    """Return the configured annual inflation rate for a category."""

    normalized = str(category).strip().lower()

    return float(
        CATEGORY_INFLATION_RATES.get(
            normalized,
            DEFAULT_INFLATION_RATE
        )
    )


# =========================================================
# REGISTER
# =========================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if current_user.is_authenticated:

        return redirect(
            url_for("home")
        )

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        if not username or not email or not password:

            flash(
                "Please fill in all fields.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        # -------------------------------------------------
        # CHECK USERNAME
        # -------------------------------------------------

        existing_username = User.query.filter_by(
            username=username
        ).first()

        if existing_username:

            flash(
                "Username already exists.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        # -------------------------------------------------
        # CHECK EMAIL
        # -------------------------------------------------

        existing_email = User.query.filter_by(
            email=email
        ).first()

        if existing_email:

            flash(
                "Email is already registered.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        # -------------------------------------------------
        # CREATE USER
        # -------------------------------------------------

        user = User(
            username=username,
            email=email
        )

        user.set_password(
            password
        )

        db.session.add(user)

        db.session.commit()

        flash(
            "Account created successfully. Please login.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# =========================================================
# LOGIN
# =========================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if current_user.is_authenticated:

        return redirect(
            url_for("home")
        )

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        user = User.query.filter_by(
            email=email
        ).first()

        if user and user.check_password(
            password
        ):

            login_user(user)

            next_page = request.args.get(
                "next"
            )

            if next_page:

                return redirect(
                    next_page
                )

            return redirect(
                url_for("home")
            )

        flash(
            "Invalid email or password.",
            "error"
        )

    return render_template(
        "login.html"
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    session.clear()

    return redirect(
        url_for("login")
    )


# =========================================================
# HOME
# =========================================================

@app.route("/")
@login_required
def home():

    return render_template(
        "index.html",
        username=current_user.username
    )


# =========================================================
# CSV UPLOAD + AI ANALYSIS
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
@login_required
def upload():

    # -----------------------------------------------------
    # GET INPUTS
    # -----------------------------------------------------

    file = request.files.get("file")

    try:

        income = float(
            request.form.get(
                "income",
                0
            )
        )

    except:

        income = 0.0

    if not file or file.filename == "":

        return "Please select a CSV file."

    if not file.filename.lower().endswith(".csv"):

        return "Only CSV files are allowed."


    # -----------------------------------------------------
    # SAVE FILE
    # -----------------------------------------------------

    path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(path)


    # =====================================================
    # LOAD CSV
    # =====================================================

    try:

        df = pd.read_csv(path)

    except Exception as e:

        return f"Could not read CSV: {e}"


    # =====================================================
    # CLEAN COLUMN NAMES
    # =====================================================

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    if (
        "date" not in df.columns
        or
        "amount" not in df.columns
    ):

        return (
            "CSV must contain Date and Amount columns."
        )


    # =====================================================
    # DATA CLEANING
    # =====================================================

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "date",
            "amount"
        ]
    )

    df = df.sort_values(
        "date"
    )

    if len(df) < 5:

        return (
            "Please upload at least "
            "5 valid expense records."
        )


    # =====================================================
    # CATEGORY CLEANING
    # =====================================================

    has_category = (
        "category" in df.columns
    )

    if has_category:

        df["category"] = (
            df["category"]
            .fillna("Other")
            .astype(str)
            .str.strip()
        )


    # =====================================================
    # BASIC STATISTICS
    # =====================================================

    total = float(
        df["amount"].sum()
    )

    average = float(
        df["amount"].mean()
    )

    highest = float(
        df["amount"].max()
    )

    highest_row = df.loc[
        df["amount"].idxmax()
    ]

    if has_category:

        highest_category = str(
            highest_row["category"]
        )

    else:

        highest_category = "Unknown"


    # =====================================================
    # DAILY EXPENSE DATA
    # =====================================================

    daily = (
        df.groupby("date")["amount"]
        .sum()
        .reset_index()
    )

    full_dates = pd.date_range(
        daily["date"].min(),
        daily["date"].max()
    )

    daily = (
        daily
        .set_index("date")
        .reindex(
            full_dates,
            fill_value=0
        )
        .rename_axis("date")
        .reset_index()
    )


    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    daily["day"] = (
        daily["date"].dt.day
    )

    daily["month"] = (
        daily["date"].dt.month
    )

    daily["weekday"] = (
        daily["date"].dt.weekday
    )

    daily["week_of_year"] = (
        daily["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    daily["day_number"] = range(
        1,
        len(daily) + 1
    )

    daily["previous_expense"] = (
        daily["amount"].shift(1)
    )

    daily["rolling_7"] = (
        daily["amount"]
        .rolling(7)
        .mean()
    )

    daily = daily.dropna()


    # =====================================================
    # RANDOM FOREST REGRESSION
    # =====================================================

    regression_features = [

        "day",
        "month",
        "weekday",
        "week_of_year",
        "day_number",
        "previous_expense",
        "rolling_7"

    ]


    if len(daily) >= 10:

        X = daily[
            regression_features
        ]

        y = daily["amount"]

        regressor = RandomForestRegressor(

            n_estimators=250,

            max_depth=10,

            min_samples_leaf=2,

            random_state=42

        )

        regressor.fit(
            X,
            y
        )


        def predict_day(
            date,
            previous,
            rolling,
            day_number
        ):

            row = pd.DataFrame([{

                "day":
                    int(date.day),

                "month":
                    int(date.month),

                "weekday":
                    int(date.weekday()),

                "week_of_year":
                    int(
                        date.isocalendar().week
                    ),

                "day_number":
                    int(day_number),

                "previous_expense":
                    float(previous),

                "rolling_7":
                    float(rolling)

            }])

            prediction = (
                regressor.predict(row)[0]
            )

            return float(
                max(
                    0,
                    prediction
                )
            )

    else:

        historical_average = float(
            daily["amount"].mean()
        )


        def predict_day(
            date,
            previous,
            rolling,
            day_number
        ):

            return float(
                max(
                    0,
                    historical_average
                )
            )


    # =====================================================
    # 28-DAY FORECAST
    # =====================================================

    last_date = df["date"].max()

    future_dates = pd.date_range(

        last_date +
        pd.Timedelta(days=1),

        periods=28

    )

    future_predictions = []

    previous = float(
        daily["amount"].iloc[-1]
    )

    rolling = float(
        daily["amount"]
        .tail(7)
        .mean()
    )

    day_number = int(
        len(daily) + 1
    )


    for date in future_dates:

        prediction = predict_day(

            date,

            previous,

            rolling,

            day_number

        )

        future_predictions.append(
            float(prediction)
        )

        previous = float(
            prediction
        )

        rolling = float(
            (
                rolling * 6
                + prediction
            ) / 7
        )

        day_number += 1


    # =====================================================
    # WEEKLY FORECAST
    # =====================================================

    week1 = float(
        sum(
            future_predictions[0:7]
        )
    )

    week2 = float(
        sum(
            future_predictions[7:14]
        )
    )

    week3 = float(
        sum(
            future_predictions[14:21]
        )
    )

    week4 = float(
        sum(
            future_predictions[21:28]
        )
    )

    next_month_expense = float(
        sum(future_predictions)
    )

    next_expense = float(
        future_predictions[0]
    )


    # =====================================================
    # RANDOM FOREST CATEGORY CLASSIFIER
    # =====================================================

    predicted_category = "Unknown"

    classifier_accuracy_note = (
        "Category prediction unavailable."
    )


    if has_category:

        category_df = df.copy()

        category_df["day"] = (
            category_df["date"].dt.day
        )

        category_df["month"] = (
            category_df["date"].dt.month
        )

        category_df["weekday"] = (
            category_df["date"].dt.weekday
        )

        category_df["week_of_year"] = (
            category_df["date"]
            .dt.isocalendar()
            .week
            .astype(int)
        )

        classifier_features = [

            "amount",
            "day",
            "month",
            "weekday",
            "week_of_year"

        ]

        classifier_df = (
            category_df[
                classifier_features
                + ["category"]
            ]
            .dropna()
        )


        if (
            len(classifier_df) >= 8
            and
            classifier_df["category"]
            .nunique() >= 2
        ):

            X_cat = classifier_df[
                classifier_features
            ]

            y_cat = classifier_df[
                "category"
            ]

            classifier = RandomForestClassifier(

                n_estimators=200,

                max_depth=10,

                min_samples_leaf=1,

                random_state=42

            )

            classifier.fit(
                X_cat,
                y_cat
            )

            next_date = (
                last_date
                + pd.Timedelta(days=1)
            )

            category_input = pd.DataFrame([{

                "amount":
                    float(next_expense),

                "day":
                    int(next_date.day),

                "month":
                    int(next_date.month),

                "weekday":
                    int(next_date.weekday()),

                "week_of_year":
                    int(
                        next_date
                        .isocalendar()
                        .week
                    )

            }])

            predicted_category = str(
                classifier.predict(
                    category_input
                )[0]
            )

            classifier_accuracy_note = (
                "Category predicted using "
                "Random Forest Classification."
            )

        else:

            predicted_category = str(
                df["category"]
                .value_counts()
                .idxmax()
            )

            classifier_accuracy_note = (
                "Limited category data detected; "
                "historical dominant category used."
            )


    # =====================================================
    # CATEGORY ANALYSIS + INFLATION
    # =====================================================

    category_predictions = []

    if has_category:

        category_totals = (
            df.groupby("category")["amount"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        category_total = float(
            category_totals.sum()
        )

        for category, amount in category_totals.items():

            amount = float(amount)

            if category_total > 0:

                share = (
                    amount /
                    category_total
                )

            else:

                share = 0.0

            predicted_amount = float(
                next_month_expense *
                share
            )

            inflation_rate = (
                get_inflation_rate(
                    category
                )
            )

            monthly_inflation_rate = (

                (
                    (
                        1 +
                        inflation_rate / 100
                    ) ** (1 / 12)
                ) - 1

            ) * 100

            inflation_impact = (
                predicted_amount *
                monthly_inflation_rate /
                100
            )

            inflation_adjusted_prediction = (
                predicted_amount +
                inflation_impact
            )

            count = int(
                (
                    df["category"] ==
                    category
                ).sum()
            )

            category_predictions.append({

                "category":
                    str(category),

                "prediction":
                    float(
                        round(
                            predicted_amount,
                            2
                        )
                    ),

                "total":
                    float(
                        round(
                            amount,
                            2
                        )
                    ),

                "count":
                    int(count),

                "percentage":
                    float(
                        round(
                            share * 100,
                            2
                        )
                    ),

                "inflation_rate":
                    float(
                        round(
                            inflation_rate,
                            2
                        )
                    ),

                "monthly_inflation_rate":
                    float(
                        round(
                            monthly_inflation_rate,
                            2
                        )
                    ),

                "inflation_impact":
                    float(
                        round(
                            inflation_impact,
                            2
                        )
                    ),

                "inflation_adjusted_prediction":
                    float(
                        round(
                            inflation_adjusted_prediction,
                            2
                        )
                    )

            })


    # =====================================================
    # 365-DAY LONG-TERM FORECAST
    # =====================================================

    long_future_dates = pd.date_range(

        last_date +
        pd.Timedelta(days=1),

        periods=365

    )

    long_predictions = []

    previous_long = float(
        daily["amount"].iloc[-1]
    )

    rolling_long = float(
        daily["amount"]
        .tail(7)
        .mean()
    )

    long_day_number = int(
        len(daily) + 1
    )


    for future_date in long_future_dates:

        prediction = predict_day(

            future_date,

            previous_long,

            rolling_long,

            long_day_number

        )

        prediction = float(
            prediction
        )

        long_predictions.append(
            prediction
        )

        previous_long = prediction

        rolling_long = float(
            (
                rolling_long * 6
                + prediction
            ) / 7
        )

        long_day_number += 1


    # =====================================================
    # 6-MONTH CUMULATIVE FORECAST
    # =====================================================

    six_month_expense = float(
        sum(
            long_predictions[:182]
        )
    )


    # =====================================================
    # 1-YEAR CUMULATIVE FORECAST
    # =====================================================

    one_year_expense = float(
        sum(
            long_predictions[:365]
        )
    )


    # =====================================================
    # MONTH-BY-MONTH FORECAST
    # =====================================================

    forecast_df = pd.DataFrame({

        "date":
            long_future_dates,

        "prediction":
            long_predictions

    })

    forecast_df["month_period"] = (
        forecast_df["date"]
        .dt.to_period("M")
    )

    grouped_months = (
        forecast_df
        .groupby("month_period")["prediction"]
        .sum()
    )

    monthly_forecast = []

    for month, amount in grouped_months.items():

        monthly_forecast.append({

            "month":
                str(month),

            "expense":
                float(
                    round(
                        amount,
                        2
                    )
                )

        })


    # =====================================================
    # LONG-TERM INCOME
    # =====================================================

    six_month_income = float(
        income * 6
    )

    one_year_income = float(
        income * 12
    )


    # =====================================================
    # LONG-TERM SAVINGS
    # =====================================================

    six_month_savings = float(
        six_month_income
        - six_month_expense
    )

    one_year_savings = float(
        one_year_income
        - one_year_expense
    )


    # =====================================================
    # SAVINGS
    # =====================================================

    predicted_savings = float(
        income
        - next_month_expense
    )

    if income > 0:

        savings_percentage = float(
            predicted_savings
            / income
            * 100
        )

    else:

        savings_percentage = 0.0


    # =====================================================
    # FINANCIAL HEALTH
    # =====================================================

    score = 100

    if income > 0:

        expense_ratio = float(
            next_month_expense
            / income
        )

        if expense_ratio > 1:

            score -= 50

        elif expense_ratio > 0.8:

            score -= 25

        elif expense_ratio > 0.6:

            score -= 10

    else:

        expense_ratio = 0.0

    if predicted_savings < 0:

        score -= 20

    score = int(
        max(
            0,
            min(
                100,
                score
            )
        )
    )


    # =====================================================
    # HEALTH MESSAGE
    # =====================================================

    if score >= 80:

        health_status = "Excellent"

        health_message = (
            "Your projected spending is well "
            "within your income level and you "
            "have a healthy expected savings margin."
        )

    elif score >= 60:

        health_status = "Good"

        health_message = (
            "Your finances appear reasonably "
            "balanced, but controlling unnecessary "
            "expenses could improve your savings."
        )

    elif score >= 40:

        health_status = "Needs Attention"

        health_message = (
            "Your projected spending is relatively "
            "high compared with your income. "
            "Reducing discretionary expenses is recommended."
        )

    else:

        health_status = "Critical"

        health_message = (
            "Your projected expenses may create "
            "financial pressure. Significant spending "
            "control is recommended."
        )


    # =====================================================
    # SPENDING WARNING
    # =====================================================

    if (
        income > 0
        and
        next_month_expense > income
    ):

        spending_warning = (
            "Predicted expenses are higher "
            "than your monthly income."
        )

    elif (
        income > 0
        and
        next_month_expense >
        income * 0.8
    ):

        spending_warning = (
            "Predicted expenses may consume "
            "more than 80% of your income."
        )

    else:

        spending_warning = (
            "Predicted spending is currently "
            "within a manageable range."
        )


    # =====================================================
    # AI INSIGHT
    # =====================================================

    insight = (

        "SterliFlux analyzed your historical "
        "expense pattern using machine learning. "

        f"The predicted expense for the next month "
        f"is approximately ₹{next_month_expense:.2f}. "

        f"The next predicted expense is "
        f"₹{next_expense:.2f} under the "
        f"{predicted_category} category."

    )

    if predicted_savings >= 0:

        insight += (

            f" With an income of ₹{income:.2f}, "

            f"your estimated next-month savings "
            f"could be ₹{predicted_savings:.2f}."

        )

    else:

        insight += (

            f" This may result in a deficit of "
            f"₹{abs(predicted_savings):.2f}."

        )


    # =====================================================
    # CHART 1 - HISTORICAL TREND
    # =====================================================

    fig_trend = px.line(

        df,

        x="date",

        y="amount",

        title="Historical Expense Trend"

    )

    trend_chart = fig_trend.to_html(
        full_html=False
    )


    # =====================================================
    # CHART 2 - CATEGORY
    # =====================================================

    if has_category:

        category_data = (
            df.groupby("category")["amount"]
            .sum()
            .reset_index()
        )

        fig_category = px.bar(

            category_data,

            x="category",

            y="amount",

            title="Spending by Category"

        )

        category_chart = (
            fig_category.to_html(
                full_html=False
            )
        )

    else:

        category_chart = ""


    # =====================================================
    # CHART 3 - 12 MONTH FORECAST
    # =====================================================

    forecast_chart_df = pd.DataFrame(
        monthly_forecast
    )

    if not forecast_chart_df.empty:

        fig_forecast = px.bar(

            forecast_chart_df,

            x="month",

            y="expense",

            title="12-Month AI Expense Forecast"

        )

        forecast_chart = (
            fig_forecast.to_html(
                full_html=False
            )
        )

    else:

        forecast_chart = ""


    # =====================================================
    # STORE ANALYSIS IN SESSION
    # =====================================================

    session["analysis"] = {

        "total":
            float(
                round(
                    total,
                    2
                )
            ),

        "average":
            float(
                round(
                    average,
                    2
                )
            ),

        "highest":
            float(
                round(
                    highest,
                    2
                )
            ),

        "highest_category":
            str(highest_category),

        "income":
            float(
                round(
                    income,
                    2
                )
            ),

        "next_month_expense":
            float(
                round(
                    next_month_expense,
                    2
                )
            ),

        "six_month_expense":
            float(
                round(
                    six_month_expense,
                    2
                )
            ),

        "one_year_expense":
            float(
                round(
                    one_year_expense,
                    2
                )
            ),

        "predicted_savings":
            float(
                round(
                    predicted_savings,
                    2
                )
            ),

        "six_month_savings":
            float(
                round(
                    six_month_savings,
                    2
                )
            ),

        "one_year_savings":
            float(
                round(
                    one_year_savings,
                    2
                )
            ),

        "savings_percentage":
            float(
                round(
                    savings_percentage,
                    2
                )
            ),

        "score":
            int(score),

        "health_status":
            str(health_status),

        "health_message":
            str(health_message),

        "expense_ratio":
            float(
                round(
                    expense_ratio * 100,
                    2
                )
            ),

        "spending_warning":
            str(spending_warning),

        "records":
            int(len(df)),

        "predicted_category":
            str(predicted_category),

        "classifier_note":
            str(classifier_accuracy_note),

        "category_predictions":
            category_predictions,

        "monthly_forecast":
            monthly_forecast
    }


    # =====================================================
    # DASHBOARD
    # =====================================================

    return render_template(

        "result.html",

        prediction=
            float(
                round(
                    next_expense,
                    2
                )
            ),

        predicted_category=
            str(predicted_category),

        total=
            float(
                round(
                    total,
                    2
                )
            ),

        average=
            float(
                round(
                    average,
                    2
                )
            ),

        highest=
            float(
                round(
                    highest,
                    2
                )
            ),

        top_category=
            str(predicted_category),

        score=
            int(score),

        records=
            int(len(df)),

        income=
            float(
                round(
                    income,
                    2
                )
            ),

        week1=
            float(
                round(
                    week1,
                    2
                )
            ),

        week2=
            float(
                round(
                    week2,
                    2
                )
            ),

        week3=
            float(
                round(
                    week3,
                    2
                )
            ),

        week4=
            float(
                round(
                    week4,
                    2
                )
            ),

        next_month_expense=
            float(
                round(
                    next_month_expense,
                    2
                )
            ),

        six_month_expense=
            float(
                round(
                    six_month_expense,
                    2
                )
            ),

        one_year_expense=
            float(
                round(
                    one_year_expense,
                    2
                )
            ),

        predicted_savings=
            float(
                round(
                    predicted_savings,
                    2
                )
            ),

        six_month_savings=
            float(
                round(
                    six_month_savings,
                    2
                )
            ),

        one_year_savings=
            float(
                round(
                    one_year_savings,
                    2
                )
            ),

        savings_percentage=
            float(
                round(
                    savings_percentage,
                    2
                )
            ),

        category_predictions=
            category_predictions,

        monthly_forecast=
            monthly_forecast,

        spending_warning=
            str(spending_warning),

        insight=
            str(insight),

        health_status=
            str(health_status),

        classifier_note=
            str(classifier_accuracy_note),

        trend_chart=
            trend_chart,

        category_chart=
            category_chart,

        forecast_chart=
            forecast_chart

    )


# =========================================================
# DETAILS PAGE
# =========================================================

@app.route(
    "/details/<section>"
)
@login_required
def details(section):

    data = session.get(
        "analysis"
    )

    if not data:

        return (
            "Please upload a CSV first."
        )


    # =====================================================
    # CATEGORY
    # =====================================================

    if section.startswith(
        "category-"
    ):

        category = section.replace(
            "category-",
            "",
            1
        )

        selected = None

        for item in (
            data["category_predictions"]
        ):

            if (
                item["category"]
                == category
            ):

                selected = item

                break

        if not selected:

            return (
                "Category not found."
            )

        return render_template(

            "details.html",

            type="category",

            title=category,

            data=data,

            category=selected

        )


    # =====================================================
    # FINANCIAL HEALTH
    # =====================================================

    if section == "financial-health":

        return render_template(

            "details.html",

            type="health",

            title="Financial Health",

            data=data

        )


    # =====================================================
    # TOTAL
    # =====================================================

    if section == "total":

        return render_template(

            "details.html",

            type="total",

            title="Total Spending",

            data=data

        )


    # =====================================================
    # AVERAGE
    # =====================================================

    if section == "average":

        return render_template(

            "details.html",

            type="average",

            title="Average Expense",

            data=data

        )


    # =====================================================
    # HIGHEST
    # =====================================================

    if section == "highest":

        return render_template(

            "details.html",

            type="highest",

            title="Highest Expense",

            data=data

        )


    # =====================================================
    # MONTHLY
    # =====================================================

    if section == "monthly":

        return render_template(

            "details.html",

            type="monthly",

            title="Next Month Forecast",

            data=data

        )


    # =====================================================
    # SIX MONTHS
    # =====================================================

    if section == "six-months":

        return render_template(

            "details.html",

            type="six-months",

            title="6-Month Expense Forecast",

            data=data

        )


    # =====================================================
    # ONE YEAR
    # =====================================================

    if section == "one-year":

        return render_template(

            "details.html",

            type="one-year",

            title="1-Year Expense Forecast",

            data=data

        )


    # =====================================================
    # SAVINGS
    # =====================================================

    if section == "savings":

        return render_template(

            "details.html",

            type="savings",

            title="Expected Savings",

            data=data

        )


    return (
        "Details not found."
    )


# =========================================================
# =========================================================
#                    STERLIFLUX AI CHATBOT
# =========================================================
# =========================================================


def chatbot_response(question, data):

    """
    Rule-based financial AI assistant.

    Uses the actual user's uploaded financial
    analysis rather than giving generic answers.
    """

    q = question.lower().strip()


    # -----------------------------------------------------
    # IMPORTANT VALUES
    # -----------------------------------------------------

    income = float(
        data.get("income", 0)
    )

    next_month = float(
        data.get(
            "next_month_expense",
            0
        )
    )

    six_month = float(
        data.get(
            "six_month_expense",
            0
        )
    )

    one_year = float(
        data.get(
            "one_year_expense",
            0
        )
    )

    savings = float(
        data.get(
            "predicted_savings",
            0
        )
    )

    six_savings = float(
        data.get(
            "six_month_savings",
            0
        )
    )

    year_savings = float(
        data.get(
            "one_year_savings",
            0
        )
    )

    score = int(
        data.get(
            "score",
            0
        )
    )

    predicted_category = str(
        data.get(
            "predicted_category",
            "Unknown"
        )
    )


    # =====================================================
    # HOUSE / HOME PURCHASE
    # =====================================================

    if (
        "house" in q
        or
        "home" in q
    ):

        if (
            "next month" in q
            or
            "buy" in q
        ):

            if income <= 0:

                return (
                    "I can't responsibly judge a house "
                    "purchase yet because your monthly "
                    "income is not available."
                )

            if savings <= 0:

                return (

                    "Based on your current SterliFlux "
                    "forecast, I would not recommend "
                    "planning a house purchase next month. "

                    f"Your projected next-month expenses "
                    f"are about ₹{next_month:,.2f}, "

                    f"compared with income of "
                    f"₹{income:,.2f}. "

                    "Your projected monthly savings are "

                    f"₹{savings:,.2f}."

                )

            return (

                "Based only on your current cash-flow "
                "pattern, your finances appear to have "
                "some room for saving, but buying a house "
                "next month is a much bigger decision than "
                "monthly expense affordability. "

                f"SterliFlux projects about "
                f"₹{savings:,.2f} in monthly savings. "

                "You should also consider your down "
                "payment, emergency fund, loan EMI, "
                "interest rate and other debts before "
                "making that decision."

            )


    # =====================================================
    # NEXT MONTH EXPENSE
    # =====================================================

    if (
        "next month" in q
        and
        (
            "expense" in q
            or
            "spend" in q
            or
            "cost" in q
        )
    ):

        return (

            f"SterliFlux predicts approximately "
            f"₹{next_month:,.2f} in expenses "
            "for your next month."

        )


    # =====================================================
    # SAVINGS
    # =====================================================

    if (
        "save" in q
        or
        "savings" in q
    ):

        if savings >= 0:

            return (

                f"Your predicted next-month savings "
                f"are approximately ₹{savings:,.2f}. "

                f"Your projected savings rate is "

                f"{data.get('savings_percentage', 0):.1f}%."

            )

        return (

            f"Your current forecast shows a projected "
            f"shortfall of ₹{abs(savings):,.2f} next month. "

            "Reducing discretionary spending could help."

        )


    # =====================================================
    # SIX MONTHS
    # =====================================================

    if (
        "6 month" in q
        or
        "six month" in q
        or
        "half year" in q
    ):

        return (

            f"Over the next 6 months, SterliFlux "
            f"projects approximately ₹{six_month:,.2f} "
            f"in cumulative expenses. "

            f"Based on your stated income, projected "
            f"savings are approximately "
            f"₹{six_savings:,.2f}."

        )


    # =====================================================
    # ONE YEAR
    # =====================================================

    if (
        "1 year" in q
        or
        "one year" in q
        or
        "year" in q
        or
        "12 month" in q
    ):

        return (

            f"Your 1-year projected expenses are "
            f"approximately ₹{one_year:,.2f}. "

            f"Your projected 1-year savings are "
            f"approximately ₹{year_savings:,.2f}."

        )


    # =====================================================
    # FINANCIAL HEALTH
    # =====================================================

    if (
        "health" in q
        or
        "financially" in q
        or
        "financial condition" in q
    ):

        return (

            f"Your SterliFlux financial health score "
            f"is {score}/100, classified as "
            f"{data.get('health_status', 'Unknown')}. "

            f"{data.get('health_message', '')}"

        )


    # =====================================================
    # CATEGORY INFLATION
    # =====================================================

    if (
        "inflation" in q
        or
        "price rise" in q
        or
        "price increase" in q
    ):

        categories = data.get(
            "category_predictions",
            []
        )

        for item in categories:

            category_name = str(
                item.get(
                    "category",
                    ""
                )
            )

            if category_name.lower() in q:

                return (

                    f"For {category_name}, SterliFlux uses an "
                    f"estimated annual inflation rate of "
                    f"{item.get('inflation_rate', 0):.2f}%. "

                    f"The approximate monthly equivalent is "
                    f"{item.get('monthly_inflation_rate', 0):.2f}%. "

                    f"On the current predicted spending of "
                    f"₹{item.get('prediction', 0):,.2f}, the estimated "
                    f"one-month inflation impact is about "
                    f"₹{item.get('inflation_impact', 0):,.2f}."

                )

        if categories:

            highest_inflation = max(

                categories,

                key=lambda x: float(
                    x.get(
                        "inflation_rate",
                        0
                    )
                )

            )

            return (

                f"SterliFlux assigns category-specific estimated "
                f"inflation rates. The highest rate in your current "
                f"data is {highest_inflation.get('category', 'Unknown')} "
                f"at {highest_inflation.get('inflation_rate', 0):.2f}% "
                f"annually. These are configured estimates, not live "
                f"official CPI figures."

            )

        return (

            "Inflation information is available after your CSV "
            "contains spending categories."

        )


    # =====================================================
    # SPENDING CATEGORY
    # =====================================================

    if (
        "category" in q
        or
        "spending most" in q
        or
        "spend most" in q
        or
        "where do i spend" in q
    ):

        categories = data.get(
            "category_predictions",
            []
        )

        if categories:

            top = categories[0]

            return (

                f"Your largest historical spending "
                f"category is {top['category']}, "
                f"accounting for about "
                f"{top['percentage']:.1f}% of your "
                f"recorded spending."

            )

        return (

            "Your CSV does not contain enough category "
            "information for me to determine this."

        )


    # =====================================================
    # INCOME
    # =====================================================

    if (
        "income" in q
        or
        "salary" in q
    ):

        return (

            f"Your monthly income entered into "
            f"SterliFlux is ₹{income:,.2f}."

        )


    # =====================================================
    # CAN I AFFORD
    # =====================================================

    if (
        "afford" in q
        or
        "can i buy" in q
    ):

        # Fixed regex from the original code
        numbers = re.findall(
            r"\d+(?:,\d{3})*(?:\.\d+)?",
            question
        )

        if numbers:

            try:

                purchase = float(
                    numbers[-1].replace(
                        ",",
                        ""
                    )
                )

                available_after_expenses = (
                    income - next_month
                )

                if (
                    available_after_expenses
                    >= purchase
                ):

                    return (

                        f"Your projected amount remaining "
                        f"after next month's expenses is "
                        f"about ₹{available_after_expenses:,.2f}. "

                        f"On that narrow cash-flow basis, "
                        f"₹{purchase:,.2f} appears potentially "
                        "affordable. However, you should "
                        "keep an emergency reserve before "
                        "making the purchase."

                    )

                return (

                    f"Your projected amount remaining after "
                    f"next month's expenses is only about "
                    f"₹{available_after_expenses:,.2f}. "

                    f"A ₹{purchase:,.2f} purchase would "
                    "therefore put pressure on your projected "
                    "cash flow."

                )

            except:

                pass


    # =====================================================
    # PREDICTION
    # =====================================================

    if (
        "predict" in q
        or
        "forecast" in q
    ):

        return (

            f"SterliFlux currently forecasts "
            f"₹{next_month:,.2f} for next month's "
            f"expenses, ₹{six_month:,.2f} over "
            f"6 months and ₹{one_year:,.2f} over "
            f"1 year."

        )


    # =====================================================
    # GENERAL HELP
    # =====================================================

    if (
        "hello" in q
        or
        "hi" in q
        or
        "hey" in q
    ):

        return (

            "Hi. I'm SterliFlux AI. Ask me about "
            "your predicted expenses, savings, "
            "financial health, spending categories, "
            "6-month forecast or 1-year forecast."

        )


    # =====================================================
    # DEFAULT RESPONSE
    # =====================================================

    return (

        "I can help you interpret your SterliFlux "
        "financial analysis. Try asking something like: "

        "\"Can I afford a ₹50,000 purchase?\", "

        "\"How much will I spend next month?\", "

        "\"Can I buy a house next month?\", "

        "\"How much can I save in a year?\" or "

        "\"Where am I spending the most?\""

    )


# =========================================================
# CHATBOT API
# =========================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    data = session.get(
        "analysis"
    )

    if not data:

        message = (

            "Please upload your expense CSV first "
            "so I can analyze your finances."

        )

        return jsonify({

            "answer":
                message,

            "reply":
                message

        })


    try:

        payload = request.get_json(
            silent=True
        ) or {}

        question = str(

            payload.get(
                "question",
                payload.get(
                    "message",
                    ""
                )
            )

        ).strip()


        if not question:

            message = (
                "Please type a question."
            )

            return jsonify({

                "answer":
                    message,

                "reply":
                    message

            })


        reply = chatbot_response(

            question,

            data

        )


        return jsonify({

            "answer":
                reply,

            "reply":
                reply

        })


    except Exception as e:

        message = (
            "Sorry, I couldn't process "
            "that question right now."
        )

        return jsonify({

            "answer":
                message,

            "reply":
                message

        }), 500


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(

        debug=False,

        use_reloader=False

    )