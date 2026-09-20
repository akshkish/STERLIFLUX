from flask import (
    Flask, render_template, request, session, jsonify,
    redirect, url_for, flash
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from urllib.parse import unquote

import os
import re
import uuid
from datetime import datetime

import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)
import plotly.express as px

# MoSPI e-Sankhyiki is used for live CPI data.
# Install with:
# pip install esankhyiki
try:
    import esankhyiki
except ImportError:
    esankhyiki = None


# =========================================================
# APP CONFIGURATION
# =========================================================

app = Flask(__name__)

app.secret_key = os.getenv(
    "STERLIFLUX_SECRET_KEY",
    "sterliflux-dev-secret-key-change-me"
)

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    "sqlite:///sterliflux.db"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)  # 10 MB

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = (
    "Please login to access SterliFlux."
)

UPLOAD_FOLDER = os.path.join(
    app.root_path,
    "uploads"
)

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
        self.password_hash = (
            generate_password_hash(password)
        )

    def check_password(self, password):
        return check_password_hash(
            self.password_hash,
            password
        )


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(
            User,
            int(user_id)
        )
    except (TypeError, ValueError):
        return None


with app.app_context():
    db.create_all()


# =========================================================
# LOGIN / REGISTER / LOGOUT
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

        if not username or not email or not password:
            flash(
                "Please fill in all fields.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        if not re.fullmatch(
            r"[^@\s]+@[^@\s]+\.[^@\s]+",
            email
        ):
            flash(
                "Please enter a valid email address.",
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

        if User.query.filter_by(
            username=username
        ).first():

            flash(
                "Username already exists.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        if User.query.filter_by(
            email=email
        ).first():

            flash(
                "Email is already registered.",
                "error"
            )

            return redirect(
                url_for("register")
            )

        user = User(
            username=username,
            email=email
        )

        user.set_password(password)

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

        if user and user.check_password(password):

            login_user(user)

            session["username"] = (
                user.username
            )

            session["email"] = (
                user.email
            )

            session["account_created"] = (
                user.created_at.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if user.created_at
                else "Not available"
            )

            next_page = request.args.get(
                "next"
            )

            if (
                next_page
                and next_page.startswith("/")
                and not next_page.startswith("//")
            ):
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


@app.route("/logout")
@login_required
def logout():

    logout_user()
    session.clear()

    return redirect(
        url_for("login")
    )



# =========================================================
# MoSPI CPI / INFLATION ENGINE
# =========================================================
#
# All CPI fetching, division matching, category mapping, and
# the "assumed rate" fallback logic live in inflation.py.
# This keeps a single source of truth instead of two CPI
# engines drifting apart.
#

from inflation import (
    CATEGORY_MAP,
    CATEGORY_DIVISIONS,
    DIVISION_TO_CATEGORY,
    ASSUMED_CATEGORY_INFLATION,
    get_cpi_inflation as _fetch_cpi_inflation,
    get_category_inflation,
    get_category_rate,
    annual_to_monthly,
    apply_inflation,
    safe_float,
)


INFLATION_CACHE = {
    "data": None,
    "fetched_at": None,
}


def get_cpi_inflation():
    """
    Thin caching wrapper around inflation.get_cpi_inflation().

    The MoSPI API call is relatively slow, so we cache the
    normalized result for 30 minutes instead of hitting the
    API on every CSV upload.
    """

    now = datetime.utcnow()

    cached_at = INFLATION_CACHE.get("fetched_at")

    if (
        INFLATION_CACHE.get("data")
        and cached_at
        and (now - cached_at).total_seconds() < 1800
    ):

        return INFLATION_CACHE["data"]

    data = _fetch_cpi_inflation()

    INFLATION_CACHE["data"] = data
    INFLATION_CACHE["fetched_at"] = now

    return data


def inflation_source_label(inflation_data):

    if inflation_data and inflation_data.get("available"):

        return (
            "MoSPI e-Sankhyiki CPI — "
            f"latest available data "
            f"({inflation_data.get('month', 'N/A')} "
            f"{inflation_data.get('year', 'N/A')})"
        )

    return (
        "MoSPI CPI unavailable — category and overall "
        "rates below fall back to presumed estimates "
        "(see inflation.py ASSUMED_CATEGORY_INFLATION)"
    )


# =========================================================
# HOME
# =========================================================

@app.route("/")
@login_required
def home():

    return render_template(
        "index.html"
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

    file = request.files.get(
        "file"
    )

    try:

        income = float(
            request.form.get(
                "income",
                0
            )
        )

    except (
        TypeError,
        ValueError
    ):

        income = 0.0

    if income < 0:
        income = 0.0

    if (
        not file
        or not file.filename
    ):

        return (
            "Please select a CSV file."
        )

    if not file.filename.lower().endswith(
        ".csv"
    ):

        return (
            "Only CSV files are allowed."
        )

    safe_name = secure_filename(
        file.filename
    )

    if not safe_name:

        return (
            "Invalid CSV filename."
        )

    # Unique filename prevents different uploads
    # from overwriting each other.

    unique_name = (
        f"{uuid.uuid4().hex}_"
        f"{safe_name}"
    )

    path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )

    try:

        file.save(
            path
        )

    except Exception as exc:

        return (
            f"Could not save CSV: {exc}"
        )

    try:

        df = pd.read_csv(
            path
        )

    except Exception as exc:

        return (
            f"Could not read CSV: {exc}"
        )

    finally:

        # The file has been read into memory;
        # we don't need to keep it on disk.
        try:
            os.remove(path)
        except OSError:
            pass

    if df.empty:

        return (
            "The uploaded CSV is empty."
        )

    # =====================================================
    # CLEAN COLUMN NAMES
    # =====================================================

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if (
        "date" not in df.columns
        or "amount" not in df.columns
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

    df = df[
        df["amount"] >= 0
    ]

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

        df.loc[
            df["category"].eq(""),
            "category"
        ] = "Other"

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

    highest_category = (
        str(
            highest_row["category"]
        )
        if has_category
        else "Unknown"
    )

    # =====================================================
    # DAILY EXPENSE DATA
    # =====================================================

    daily = (
        df.groupby("date")[
            "amount"
        ]
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
        .rolling(7, min_periods=1)
        .mean()
    )

    daily_model = (
        daily
        .dropna()
        .copy()
    )

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
        "rolling_7",
    ]

    if len(daily_model) >= 10:

        X = daily_model[
            regression_features
        ]

        y = daily_model[
            "amount"
        ]

        regressor = RandomForestRegressor(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        regressor.fit(
            X,
            y
        )

        def predict_day(
            date,
            previous,
            rolling,
            day_number,
        ):

            row = pd.DataFrame([
                {
                    "day": int(
                        date.day
                    ),

                    "month": int(
                        date.month
                    ),

                    "weekday": int(
                        date.weekday()
                    ),

                    "week_of_year": int(
                        date.isocalendar().week
                    ),

                    "day_number": int(
                        day_number
                    ),

                    "previous_expense": float(
                        previous
                    ),

                    "rolling_7": float(
                        rolling
                    ),
                }
            ])

            prediction = (
                regressor
                .predict(row)[0]
            )

            return float(
                max(
                    0.0,
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
            day_number,
        ):

            return float(
                max(
                    0.0,
                    historical_average
                )
            )

    # =====================================================
    # 28-DAY FORECAST
    # =====================================================

    last_date = df["date"].max()

    future_dates = pd.date_range(
        last_date
        + pd.Timedelta(days=1),
        periods=28,
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
            day_number,
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
        sum(
            future_predictions
        )
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
            "week_of_year",
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
            and classifier_df[
                "category"
            ].nunique() >= 2
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
                random_state=42,
                n_jobs=-1,
            )

            classifier.fit(
                X_cat,
                y_cat
            )

            next_date = (
                last_date
                + pd.Timedelta(days=1)
            )

            category_input = pd.DataFrame([
                {
                    "amount": float(
                        next_expense
                    ),

                    "day": int(
                        next_date.day
                    ),

                    "month": int(
                        next_date.month
                    ),

                    "weekday": int(
                        next_date.weekday()
                    ),

                    "week_of_year": int(
                        next_date
                        .isocalendar()
                        .week
                    ),
                }
            ])

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
    # CATEGORY ANALYSIS + LIVE INFLATION
    # =====================================================

    inflation_data = (
        get_cpi_inflation()
    )

    mospi_source = (
        inflation_source_label(
            inflation_data
        )
    )

    category_predictions = []

    top_spending_category = (
        "Unknown"
    )

    category_totals = pd.Series(
        dtype=float
    )

    if has_category:

        category_totals = (
            df.groupby("category")[
                "amount"
            ]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        category_total = float(
            category_totals.sum()
        )

        for category, amount in (
            category_totals.items()
        ):

            amount = float(
                amount
            )

            share = (
                amount / category_total
                if category_total > 0
                else 0.0
            )

            predicted_amount = float(
                next_month_expense
                * share
            )

            inflation_info = (
                get_category_inflation(
                    category,
                    inflation_data
                )
            )

            inflation_rate = (
                inflation_info.get(
                    "rate"
                )
            )

            # -------------------------------------------------
            # SAFE INFLATION CALCULATION
            # -------------------------------------------------

            if inflation_rate is not None:

                monthly_inflation_rate = (
                    annual_to_monthly(
                        inflation_rate
                    )
                )

                inflation_impact = (
                    predicted_amount
                    * monthly_inflation_rate
                    / 100
                )

                inflation_adjusted_prediction = (
                    predicted_amount
                    + inflation_impact
                )

            else:

                monthly_inflation_rate = None

                inflation_impact = 0.0

                inflation_adjusted_prediction = (
                    predicted_amount
                )

            count = int(
                (
                    df["category"]
                    == category
                ).sum()
            )

            category_predictions.append(
                {
                    "category": str(
                        category
                    ),

                    "prediction": round(
                        predicted_amount,
                        2
                    ),

                    "total": round(
                        amount,
                        2
                    ),

                    "count": count,

                    "percentage": round(
                        share * 100,
                        2
                    ),

                    "inflation_rate": (
                        round(
                            inflation_rate,
                            2
                        )
                        if inflation_rate is not None
                        else None
                    ),

                    "monthly_inflation_rate": (
                        round(
                            monthly_inflation_rate,
                            4
                        )
                        if monthly_inflation_rate
                        is not None
                        else None
                    ),

                    "inflation_impact": round(
                        inflation_impact,
                        2
                    ),

                    "inflation_adjusted_prediction": round(
                        inflation_adjusted_prediction,
                        2
                    ),

                    "inflation_month": (
                        inflation_info.get(
                            "month",
                            "Unavailable"
                        )
                    ),

                    "inflation_year": (
                        inflation_info.get(
                            "year",
                            "Unavailable"
                        )
                    ),

                    "inflation_source": (
                        inflation_info.get(
                            "source",
                            "MoSPI data unavailable"
                        )
                    ),

                    "inflation_estimated": bool(
                        inflation_info.get(
                            "estimated",
                            False
                        )
                    ),
                }
            )

    if (
        has_category
        and not category_totals.empty
    ):

        top_spending_category = str(
            category_totals.idxmax()
        )

    # =====================================================
    # WEIGHTED INFLATION
    # =====================================================

    weighted_rates = [
        (
            item["inflation_rate"],
            item["percentage"]
        )
        for item in category_predictions
        if item["inflation_rate"] is not None
    ]

    weighted_denominator = sum(
        percentage
        for _, percentage
        in weighted_rates
    )

    if (
        weighted_rates
        and weighted_denominator > 0
    ):

        weighted_inflation_rate = (
            sum(
                rate * percentage
                for rate, percentage
                in weighted_rates
            )
            / weighted_denominator
        )

    elif inflation_data.get("available"):

        # No per-category rates matched at all;
        # fall back to overall CPI instead of
        # silently reporting "no inflation".
        weighted_inflation_rate = safe_float(
            inflation_data.get("overall"),
            None
        )

    else:

        weighted_inflation_rate = None

    # =====================================================
    # 365-DAY LONG-TERM FORECAST
    # =====================================================

    long_future_dates = pd.date_range(
        last_date
        + pd.Timedelta(days=1),
        periods=365,
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
            long_day_number,
        )

        prediction = float(
            max(
                0.0,
                prediction
            )
        )

        long_predictions.append(
            prediction
        )

        previous_long = (
            prediction
        )

        rolling_long = float(
            (
                rolling_long * 6
                + prediction
            ) / 7
        )

        long_day_number += 1

    six_month_expense = float(
        sum(
            long_predictions[:182]
        )
    )

    one_year_expense = float(
        sum(
            long_predictions[:365]
        )
    )

    # =====================================================
    # INFLATION-ADJUSTED FORECASTS
    # =====================================================

    if (
        weighted_inflation_rate
        is not None
    ):

        monthly_weighted_inflation = (
            annual_to_monthly(
                weighted_inflation_rate
            ) / 100
        )

        if category_predictions:

            inflation_adjusted_next_month = (
                sum(
                    item[
                        "inflation_adjusted_prediction"
                    ]
                    for item
                    in category_predictions
                )
            )

        else:

            inflation_adjusted_next_month = (
                next_month_expense
                * (
                    1
                    + monthly_weighted_inflation
                )
            )

        inflation_adjusted_6_month = (
            six_month_expense
            * (
                1
                + weighted_inflation_rate / 100
            )
            ** 0.5
        )

        inflation_adjusted_1_year = (
            one_year_expense
            * (
                1
                + weighted_inflation_rate / 100
            )
        )

        inflation_adjusted_next_year = (
            one_year_expense
            * (
                1
                + weighted_inflation_rate / 100
            )
            ** 2
        )

        inflation_impact = (
            inflation_adjusted_1_year
            - one_year_expense
        )

    else:

        inflation_adjusted_next_month = (
            next_month_expense
        )

        inflation_adjusted_6_month = (
            six_month_expense
        )

        inflation_adjusted_1_year = (
            one_year_expense
        )

        inflation_adjusted_next_year = (
            one_year_expense
        )

        inflation_impact = 0.0

    # =====================================================
    # MONTH-BY-MONTH FORECAST
    # =====================================================

    forecast_df = pd.DataFrame(
        {
            "date": long_future_dates,
            "prediction": long_predictions,
        }
    )

    forecast_df["month_period"] = (
        forecast_df["date"]
        .dt.to_period("M")
    )

    grouped_months = (
        forecast_df
        .groupby("month_period")[
            "prediction"
        ]
        .sum()
    )

    monthly_forecast = [
        {
            "month": str(
                month
            ),

            "expense": round(
                float(amount),
                2
            ),
        }

        for month, amount
        in grouped_months.items()
    ]

    # =====================================================
    # LONG-TERM INCOME
    # =====================================================

    six_month_income = float(
        income * 6
    )

    one_year_income = float(
        income * 12
    )

    six_month_savings = float(
        six_month_income
        - six_month_expense
    )

    one_year_savings = float(
        one_year_income
        - one_year_expense
    )

    predicted_savings = float(
        income
        - next_month_expense
    )

    savings_percentage = (
        float(
            predicted_savings
            / income
            * 100
        )
        if income > 0
        else 0.0
    )

    # =====================================================
    # FINANCIAL HEALTH
    # =====================================================

    score = 100

    if income > 0:

        expense_ratio = (
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

    if score >= 80:

        health_status = (
            "Excellent"
        )

        health_message = (
            "Your projected spending is well "
            "within your income level and you "
            "have a healthy expected savings margin."
        )

    elif score >= 60:

        health_status = (
            "Good"
        )

        health_message = (
            "Your finances appear reasonably "
            "balanced, but controlling unnecessary "
            "expenses could improve your savings."
        )

    elif score >= 40:

        health_status = (
            "Needs Attention"
        )

        health_message = (
            "Your projected spending is relatively "
            "high compared with your income. "
            "Reducing discretionary expenses is recommended."
        )

    else:

        health_status = (
            "Critical"
        )

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
        and next_month_expense > income
    ):

        spending_warning = (
            "Predicted expenses are higher "
            "than your monthly income."
        )

    elif (
        income > 0
        and next_month_expense > income * 0.8
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
    # CHART 1
    # =====================================================

    fig_trend = px.line(
        df,
        x="date",
        y="amount",
        title="Historical Expense Trend",
    )

    trend_chart = (
        fig_trend.to_html(
            full_html=False,
            include_plotlyjs="cdn",
        )
    )

    # =====================================================
    # CHART 2
    # =====================================================

    if has_category:

        category_data = (
            df.groupby("category")[
                "amount"
            ]
            .sum()
            .reset_index()
        )

        fig_category = px.bar(
            category_data,
            x="category",
            y="amount",
            title="Spending by Category",
        )

        category_chart = (
            fig_category.to_html(
                full_html=False,
                include_plotlyjs=False,
            )
        )

    else:

        category_chart = ""

    # =====================================================
    # CHART 3
    # =====================================================

    forecast_chart_df = pd.DataFrame(
        monthly_forecast
    )

    if not forecast_chart_df.empty:

        fig_forecast = px.bar(
            forecast_chart_df,
            x="month",
            y="expense",
            title="12-Month AI Expense Forecast",
        )

        forecast_chart = (
            fig_forecast.to_html(
                full_html=False,
                include_plotlyjs=False,
            )
        )

    else:

        forecast_chart = ""

    # =====================================================
    # STORE ANALYSIS
    # =====================================================
    # NOTE: Flask's default session is a signed client-side
    # cookie with a ~4KB practical size limit. If you expect
    # many categories or a long monthly_forecast, switch to a
    # server-side session backend (e.g. Flask-Session with
    # filesystem/Redis storage) so this payload doesn't get
    # silently truncated or rejected by the browser.

    session["analysis"] = {

        "total": round(
            total,
            2
        ),

        "average": round(
            average,
            2
        ),

        "highest": round(
            highest,
            2
        ),

        "highest_category": str(
            highest_category
        ),

        "income": round(
            income,
            2
        ),

        "next_month_expense": round(
            next_month_expense,
            2
        ),

        "six_month_expense": round(
            six_month_expense,
            2
        ),

        "one_year_expense": round(
            one_year_expense,
            2
        ),

        "predicted_savings": round(
            predicted_savings,
            2
        ),

        "six_month_savings": round(
            six_month_savings,
            2
        ),

        "one_year_savings": round(
            one_year_savings,
            2
        ),

        "savings_percentage": round(
            savings_percentage,
            2
        ),

        "score": score,

        "health_status": (
            health_status
        ),

        "health_message": (
            health_message
        ),

        "expense_ratio": round(
            expense_ratio * 100,
            2
        ),

        "spending_warning": (
            spending_warning
        ),

        "records": int(
            len(df)
        ),

        "predicted_category": str(
            predicted_category
        ),

        "top_category": str(
            top_spending_category
        ),

        "mospi_source": str(
            mospi_source
        ),

        "inflation_rate": (
            round(
                weighted_inflation_rate,
                2
            )
            if weighted_inflation_rate
            is not None
            else None
        ),

        "inflation_adjusted_next_month": round(
            inflation_adjusted_next_month,
            2
        ),

        "inflation_adjusted_6_month": round(
            inflation_adjusted_6_month,
            2
        ),

        "inflation_adjusted_1_year": round(
            inflation_adjusted_1_year,
            2
        ),

        "inflation_adjusted_next_year": round(
            inflation_adjusted_next_year,
            2
        ),

        "inflation_impact": round(
            inflation_impact,
            2
        ),

        "classifier_note": (
            classifier_accuracy_note
        ),

        "category_predictions": (
            category_predictions
        ),

        "monthly_forecast": (
            monthly_forecast
        ),
    }

    # =====================================================
    # DASHBOARD
    # =====================================================

    return render_template(
        "result.html",

        prediction=round(
            next_expense,
            2
        ),

        predicted_category=str(
            predicted_category
        ),

        total=round(
            total,
            2
        ),

        average=round(
            average,
            2
        ),

        highest=round(
            highest,
            2
        ),

        top_category=str(
            top_spending_category
        ),

        score=int(
            score
        ),

        records=int(
            len(df)
        ),

        income=round(
            income,
            2
        ),

        week1=round(
            week1,
            2
        ),

        week2=round(
            week2,
            2
        ),

        week3=round(
            week3,
            2
        ),

        week4=round(
            week4,
            2
        ),

        next_month_expense=round(
            next_month_expense,
            2
        ),

        six_month_expense=round(
            six_month_expense,
            2
        ),

        one_year_expense=round(
            one_year_expense,
            2
        ),

        predicted_savings=round(
            predicted_savings,
            2
        ),

        six_month_savings=round(
            six_month_savings,
            2
        ),

        one_year_savings=round(
            one_year_savings,
            2
        ),

        savings_percentage=round(
            savings_percentage,
            2
        ),

        category_predictions=(
            category_predictions
        ),

        monthly_forecast=(
            monthly_forecast
        ),

        spending_warning=str(
            spending_warning
        ),

        insight=str(
            insight
        ),

        health_status=str(
            health_status
        ),

        classifier_note=str(
            classifier_accuracy_note
        ),

        trend_chart=(
            trend_chart
        ),

        category_chart=(
            category_chart
        ),

        forecast_chart=(
            forecast_chart
        ),

        health_message=str(
            health_message
        ),

        inflation_rate=(
            round(
                weighted_inflation_rate,
                2
            )
            if weighted_inflation_rate
            is not None
            else None
        ),

        inflation_adjusted_next_month=round(
            inflation_adjusted_next_month,
            2
        ),

        inflation_adjusted_6_month=round(
            inflation_adjusted_6_month,
            2
        ),

        inflation_adjusted_1_year=round(
            inflation_adjusted_1_year,
            2
        ),

        inflation_adjusted_next_year=round(
            inflation_adjusted_next_year,
            2
        ),

        inflation_impact=round(
            inflation_impact,
            2
        ),

        mospi_source=str(
            mospi_source
        )
    )


# =========================================================
# DETAILS PAGE
# =========================================================

@app.route(
    "/details/<path:section>"
)
@login_required
def details(
    section
):

    data = session.get(
        "analysis"
    )

    if not data:

        return (
            "Please upload a CSV first."
        )

    if section.startswith(
        "category-"
    ):

        category = section.replace(
            "category-",
            "",
            1
        )

        category = unquote(
            category
        )

        selected = None

        for item in data.get(
            "category_predictions",
            []
        ):

            if (
                str(
                    item.get(
                        "category",
                        ""
                    )
                )
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

            category=selected,
        )

    section_map = {

        "financial-health": (
            "health",
            "Financial Health"
        ),

        "total": (
            "total",
            "Total Spending"
        ),

        "average": (
            "average",
            "Average Expense"
        ),

        "highest": (
            "highest",
            "Highest Expense"
        ),

        "monthly": (
            "monthly",
            "Next Month Forecast"
        ),

        "six-months": (
            "six-months",
            "6-Month Expense Forecast"
        ),

        "one-year": (
            "one-year",
            "1-Year Expense Forecast"
        ),

        "savings": (
            "savings",
            "Expected Savings"
        ),
    }

    if section in section_map:

        detail_type, title = (
            section_map[
                section
            ]
        )

        return render_template(
            "details.html",

            type=detail_type,

            title=title,

            data=data,
        )

    return (
        "Details not found."
    )


# =========================================================
# STERLIFLUX AI CHATBOT
# =========================================================

def _find_category_item(
    data,
    query_text
):
    """
    Try to find a category_predictions entry whose
    category name appears in the user's question.
    """

    query_text = query_text.lower()

    for item in data.get(
        "category_predictions",
        []
    ):

        category_name = str(
            item.get(
                "category",
                ""
            )
        ).lower()

        if (
            category_name
            and category_name in query_text
        ):

            return item

    # Fall back to matching against CATEGORY_MAP keywords,
    # e.g. "travel" -> "Transport".
    for keyword, division in CATEGORY_MAP.items():

        if keyword in query_text:

            for item in data.get(
                "category_predictions",
                []
            ):

                if str(
                    item.get(
                        "category",
                        ""
                    )
                ).strip().lower() == keyword:

                    return item

    return None


def chatbot_response(
    question,
    data
):

    q = str(
        question
    ).lower().strip()

    if not q:

        return (
            "Ask me about your spending, savings, "
            "forecasts, financial health, or "
            "inflation for a specific category."
        )

    if not data:

        return (
            "I don't have any analysis yet. "
            "Please upload a CSV first so I can "
            "answer questions about your finances."
        )

    income = float(
        data.get(
            "income",
            0
        ) or 0
    )

    # -----------------------------------------------------
    # GREETINGS
    # -----------------------------------------------------

    if q in (
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
    ):

        return (
            "Hello! I'm the SterliFlux assistant. "
            "You can ask me about your total spending, "
            "average expense, savings, financial health "
            "score, forecasts, or inflation impact on "
            "any category."
        )

    # -----------------------------------------------------
    # CATEGORY-SPECIFIC (including inflation) QUESTIONS
    # -----------------------------------------------------

    category_item = _find_category_item(
        data,
        q
    )

    if category_item and (
        "inflation" in q
        or "category" in q
        or category_item.get(
            "category",
            ""
        ).lower() in q
    ):

        rate = category_item.get(
            "inflation_rate"
        )

        if rate is not None:

            return (
                f"For {category_item.get('category')}, "
                f"the latest MoSPI CPI inflation rate is "
                f"{rate}% "
                f"({category_item.get('inflation_month')} "
                f"{category_item.get('inflation_year')}). "
                f"Your predicted next-month spend in this "
                f"category is ₹"
                f"{category_item.get('prediction')}, "
                f"adjusted for inflation to approximately ₹"
                f"{category_item.get('inflation_adjusted_prediction')}."
            )

        return (
            f"For {category_item.get('category')}, "
            f"a specific MoSPI category inflation rate "
            f"isn't currently available, so I've used your "
            f"predicted spend of ₹"
            f"{category_item.get('prediction')} "
            f"without an inflation adjustment."
        )

    # -----------------------------------------------------
    # OVERALL INFLATION
    # -----------------------------------------------------

    if "inflation" in q:

        rate = data.get(
            "inflation_rate"
        )

        if rate is not None:

            return (
                f"Your weighted inflation rate, based on "
                f"your spending mix and live MoSPI CPI data, "
                f"is approximately {rate}% per year. This "
                f"adds roughly ₹{data.get('inflation_impact')} "
                f"to your projected 1-year expenses "
                f"(₹{data.get('inflation_adjusted_1_year')} "
                f"vs ₹{data.get('one_year_expense')} without "
                f"inflation)."
            )

        return (
            "Live MoSPI inflation data isn't currently "
            "available, so forecasts don't include an "
            "inflation adjustment right now. "
            f"({data.get('mospi_source')})"
        )

    # -----------------------------------------------------
    # SAVINGS
    # -----------------------------------------------------

    if "saving" in q or "savings" in q:

        return (
            f"Based on an income of ₹{income:.2f}, your "
            f"predicted savings for next month are ₹"
            f"{data.get('predicted_savings')} "
            f"({data.get('savings_percentage')}% of income). "
            f"Over 6 months you could save approximately ₹"
            f"{data.get('six_month_savings')}, and over "
            f"1 year approximately ₹"
            f"{data.get('one_year_savings')}."
        )

    # -----------------------------------------------------
    # FINANCIAL HEALTH
    # -----------------------------------------------------

    if (
        "health" in q
        or "score" in q
    ):

        return (
            f"Your financial health score is "
            f"{data.get('score')}/100, rated as "
            f"'{data.get('health_status')}'. "
            f"{data.get('health_message')}"
        )

    # -----------------------------------------------------
    # FORECASTS
    # -----------------------------------------------------

    if (
        "6 month" in q
        or "six month" in q
        or "six-month" in q
    ):

        return (
            f"Your projected 6-month expense is ₹"
            f"{data.get('six_month_expense')}, or "
            f"approximately ₹"
            f"{data.get('inflation_adjusted_6_month')} "
            f"after adjusting for inflation."
        )

    if (
        "year" in q
        or "annual" in q
        or "yearly" in q
    ):

        return (
            f"Your projected 1-year expense is ₹"
            f"{data.get('one_year_expense')}, or "
            f"approximately ₹"
            f"{data.get('inflation_adjusted_1_year')} "
            f"after adjusting for inflation."
        )

    if (
        "next month" in q
        or "forecast" in q
        or "predict" in q
    ):

        return (
            f"Your predicted expense for next month is ₹"
            f"{data.get('next_month_expense')}, or "
            f"approximately ₹"
            f"{data.get('inflation_adjusted_next_month')} "
            f"after adjusting for inflation."
        )

    # -----------------------------------------------------
    # BASIC STATISTICS
    # -----------------------------------------------------

    if "total" in q:

        return (
            f"Your total recorded spending is ₹"
            f"{data.get('total')} across "
            f"{data.get('records')} records."
        )

    if "average" in q or "avg" in q:

        return (
            f"Your average expense per record is ₹"
            f"{data.get('average')}."
        )

    if "highest" in q or "biggest" in q:

        return (
            f"Your highest single expense was ₹"
            f"{data.get('highest')}, in the "
            f"{data.get('highest_category')} category."
        )

    if (
        "top category" in q
        or "top spending" in q
        or "most spent" in q
    ):

        return (
            f"Your top spending category is "
            f"{data.get('top_category')}."
        )

    # -----------------------------------------------------
    # WARNING / RISK
    # -----------------------------------------------------

    if (
        "warning" in q
        or "risk" in q
        or "deficit" in q
    ):

        return str(
            data.get(
                "spending_warning"
            )
        )

    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    return (
        "I can help with total spending, average expense, "
        "highest expense, top category, savings, financial "
        "health, monthly/6-month/1-year forecasts, or "
        "inflation for a specific category (e.g. "
        "'what's the inflation on travel?'). Try rephrasing "
        "your question."
    )


@app.route(
    "/chat",
    methods=["POST"]
)
@login_required
def chat():

    payload = request.get_json(
        silent=True
    ) or {}

    question = str(
        payload.get(
            "message",
            payload.get(
                "question",
                ""
            )
        )
    ).strip()

    data = session.get(
        "analysis"
    )

    answer = chatbot_response(
        question,
        data
    )

    return jsonify(
        {
            "reply": answer,
        }
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    debug_mode = (
        os.getenv(
            "FLASK_DEBUG",
            "0"
        )
        == "1"
    )

    app.run(
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                5000
            )
        ),
        debug=debug_mode,
    )