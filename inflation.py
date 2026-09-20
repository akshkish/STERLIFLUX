# ============================================================
# STERLIFLUX - MoSPI CPI INFLATION ENGINE
# ============================================================
#
# Uses:
#   MoSPI e-Sankhyiki CPI
#
# Purpose:
#   1. Fetch latest CPI data from MoSPI
#   2. Use All India + Combined
#   3. Extract REAL division-level inflation
#   4. Map SterliFlux categories to MoSPI divisions
#   5. Calculate equivalent monthly inflation
#   6. Apply inflation to predicted expenses
#   7. If MoSPI has no data for a category's division,
#      fall back to a configured "assumed" inflation rate
#      instead of silently reporting None/0.
#
# MoSPI CPI:
#   Base year : 2024
#   Series    : Current
#
# ============================================================

import pandas as pd
from datetime import datetime

try:
    import esankhyiki
except ImportError:
    esankhyiki = None


# ============================================================
# CONFIGURATION
# ============================================================

BASE_YEAR = "2024"
SERIES = "Current"

# MoSPI API maximum allowed limit is 100.
API_LIMIT = 100


# ============================================================
# ACTUAL MoSPI DIVISION CODES
# ============================================================
#
# These are the division codes returned by the 2024-base-year
# CPI API.
#
# IMPORTANT:
# Do not replace these with the old CPI classification names.
#
# ============================================================

MOSPI_DIVISION_CODES = {

    "Food and beverages": "01",

    "Paan, tobacco and intoxicants": "02",

    "Clothing and footwear": "03",

    "Housing, water, electricity, gas and other fuels": "04",

    "Furnishings, household equipment and routine household maintenance": "05",

    "Health": "06",

    "Transport": "07",

    "Information and communication": "08",

    "Recreation, sport and culture": "09",

    "Education services": "10",

    "Restaurants and accommodation services": "11",

    "Personal care, social protection and miscellaneous goods and services": "13",
}


# ============================================================
# STERLIFLUX CATEGORY -> MoSPI DIVISION
# ============================================================
#
# IMPORTANT:
#
# The new MoSPI classification combines:
#
#   Housing
#   Water
#   Electricity
#   Gas
#   Other fuels
#
# into division 04.
#
# Therefore Housing and Fuel will intentionally use the same
# MoSPI inflation rate.
#
# ============================================================

CATEGORY_MAP = {

    # --------------------------------------------------------
    # FOOD
    # --------------------------------------------------------

    "food": "Food and beverages",
    "foods": "Food and beverages",
    "grocery": "Food and beverages",
    "groceries": "Food and beverages",

    # --------------------------------------------------------
    # CLOTHING
    # --------------------------------------------------------

    "clothing": "Clothing and footwear",
    "clothes": "Clothing and footwear",
    "cloth": "Clothing and footwear",
    "fashion": "Clothing and footwear",
    "footwear": "Clothing and footwear",
    "shoes": "Clothing and footwear",

    # --------------------------------------------------------
    # HOUSING
    # --------------------------------------------------------

    "housing":
        "Housing, water, electricity, gas and other fuels",

    "house":
        "Housing, water, electricity, gas and other fuels",

    "household rent":
        "Housing, water, electricity, gas and other fuels",

    "rent":
        "Housing, water, electricity, gas and other fuels",

    "home":
        "Housing, water, electricity, gas and other fuels",

    # --------------------------------------------------------
    # FUEL / ELECTRICITY
    # --------------------------------------------------------
    #
    # MoSPI 2024 classification places these under division 04.
    #

    "fuel":
        "Housing, water, electricity, gas and other fuels",

    "gas":
        "Housing, water, electricity, gas and other fuels",

    "lpg":
        "Housing, water, electricity, gas and other fuels",

    "electricity":
        "Housing, water, electricity, gas and other fuels",

    "power":
        "Housing, water, electricity, gas and other fuels",

    # --------------------------------------------------------
    # HEALTH
    # --------------------------------------------------------

    "health": "Health",
    "medical": "Health",
    "medicine": "Health",
    "medicines": "Health",
    "healthcare": "Health",
    "hospital": "Health",
    "doctor": "Health",
    "pharmacy": "Health",

    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    "education": "Education services",
    "school": "Education services",
    "college": "Education services",
    "tuition": "Education services",
    "university": "Education services",
    "course": "Education services",
    "courses": "Education services",

    # --------------------------------------------------------
    # TRANSPORT
    # --------------------------------------------------------

    "transport": "Transport",
    "transportation": "Transport",
    "travel": "Transport",
    "taxi": "Transport",
    "cab": "Transport",
    "bus": "Transport",
    "train": "Transport",
    "flight": "Transport",
    "flights": "Transport",
    "petrol": "Transport",
    "diesel": "Transport",

    # --------------------------------------------------------
    # COMMUNICATION
    # --------------------------------------------------------

    "communication": "Information and communication",
    "mobile": "Information and communication",
    "internet": "Information and communication",
    "phone": "Information and communication",
    "telephone": "Information and communication",
    "data": "Information and communication",

    # --------------------------------------------------------
    # ENTERTAINMENT / RECREATION
    # --------------------------------------------------------

    "entertainment": "Recreation, sport and culture",
    "recreation": "Recreation, sport and culture",
    "movies": "Recreation, sport and culture",
    "movie": "Recreation, sport and culture",
    "sports": "Recreation, sport and culture",
    "sport": "Recreation, sport and culture",
    "culture": "Recreation, sport and culture",
    "games": "Recreation, sport and culture",

    # --------------------------------------------------------
    # PERSONAL CARE
    # --------------------------------------------------------

    "personal care":
        "Personal care, social protection and miscellaneous goods and services",

    "personal":
        "Personal care, social protection and miscellaneous goods and services",

    "beauty":
        "Personal care, social protection and miscellaneous goods and services",

    "salon":
        "Personal care, social protection and miscellaneous goods and services",

    "cosmetics":
        "Personal care, social protection and miscellaneous goods and services",

    # --------------------------------------------------------
    # HOUSEHOLD
    # --------------------------------------------------------

    "household":
        "Furnishings, household equipment and routine household maintenance",

    "shopping":
        "Furnishings, household equipment and routine household maintenance",

    "home goods":
        "Furnishings, household equipment and routine household maintenance",

    "furniture":
        "Furnishings, household equipment and routine household maintenance",

    "appliances":
        "Furnishings, household equipment and routine household maintenance",

    # --------------------------------------------------------
    # TOBACCO
    # --------------------------------------------------------

    "tobacco": "Paan, tobacco and intoxicants",
    "smoking": "Paan, tobacco and intoxicants",
    "cigarette": "Paan, tobacco and intoxicants",
    "cigarettes": "Paan, tobacco and intoxicants",

    # --------------------------------------------------------
    # RESTAURANTS / ACCOMMODATION
    # --------------------------------------------------------

    "restaurant":
        "Restaurants and accommodation services",

    "restaurants":
        "Restaurants and accommodation services",

    "food service":
        "Restaurants and accommodation services",

    "hotel":
        "Restaurants and accommodation services",

    "hotels":
        "Restaurants and accommodation services",

    "accommodation":
        "Restaurants and accommodation services",

    # --------------------------------------------------------
    # UNKNOWN / OTHER
    # --------------------------------------------------------

    "other": "CPI (General)",
}


# ============================================================
# MoSPI DIVISION -> STERLIFLUX DISPLAY CATEGORY
# ============================================================

DIVISION_TO_CATEGORY = {

    "Food and beverages":
        "Food",

    "Clothing and footwear":
        "Clothing",

    "Housing, water, electricity, gas and other fuels":
        "Housing",

    "Furnishings, household equipment and routine household maintenance":
        "Household",

    "Health":
        "Health",

    "Transport":
        "Transport",

    "Information and communication":
        "Communication",

    "Recreation, sport and culture":
        "Entertainment",

    "Education services":
        "Education",

    "Restaurants and accommodation services":
        "Restaurants",

    "Personal care, social protection and miscellaneous goods and services":
        "Personal Care",

    "Paan, tobacco and intoxicants":
        "Tobacco",
}


# ============================================================
# CATEGORY LIST
# ============================================================

CATEGORY_DIVISIONS = [

    "Food and beverages",

    "Clothing and footwear",

    "Housing, water, electricity, gas and other fuels",

    "Furnishings, household equipment and routine household maintenance",

    "Health",

    "Transport",

    "Information and communication",

    "Recreation, sport and culture",

    "Education services",

    "Restaurants and accommodation services",

    "Personal care, social protection and miscellaneous goods and services",

    "Paan, tobacco and intoxicants",
]


# ============================================================
# ASSUMED / FALLBACK CATEGORY INFLATION
# ============================================================
#
# These are NOT official MoSPI figures. They are used ONLY as
# a last-resort presumed annual inflation rate when MoSPI's
# live API does not return a usable value for that division
# (e.g. the division was missing from the current response,
# was filtered out, or the API itself is unreachable).
#
# Any rate served from this table is always labeled with
# "estimated": True and a source string that says so, so the
# UI/chatbot never claims it's live MoSPI data.
#
# Tune these to whatever long-run averages you're comfortable
# presenting to users, or load them from an env var / config
# file if you want them adjustable without a redeploy.
#
# ============================================================

ASSUMED_CATEGORY_INFLATION = {

    "Food": 6.5,
    "Clothing": 3.5,
    "Housing": 4.5,
    "Household": 4.0,
    "Health": 6.0,
    "Transport": 5.0,
    "Communication": 2.0,
    "Entertainment": 4.0,
    "Education": 7.0,
    "Restaurants": 5.5,
    "Personal Care": 4.5,
    "Tobacco": 8.0,
}

# Used when a category can't even be mapped to a division and
# the overall MoSPI CPI itself is unavailable.
ASSUMED_GENERAL_INFLATION = 5.5


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value, default=None):

    try:

        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except (ValueError, TypeError):

        return default


# ============================================================
# GET CURRENT YEAR
# ============================================================

def get_current_year():

    return str(
        datetime.now().year
    )


# ============================================================
# NORMALIZE DATAFRAME
# ============================================================

def normalize_dataframe(df):

    if df is None:

        return pd.DataFrame()

    if not isinstance(
        df,
        pd.DataFrame
    ):

        try:

            df = pd.DataFrame(df)

        except Exception:

            return pd.DataFrame()

    if df.empty:

        return df

    df = df.copy()

    df.columns = [

        str(column).strip()

        for column in df.columns

    ]

    return df


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    df,
    possible_names
):

    if df is None or df.empty:

        return None

    lookup = {

        str(column)
        .strip()
        .lower():

        column

        for column in df.columns

    }

    for name in possible_names:

        found = lookup.get(

            str(name)
            .strip()
            .lower()

        )

        if found is not None:

            return found

    return None


# ============================================================
# FETCH CPI DATA
# ============================================================
#
# The MoSPI API supports a maximum limit of 100.
# We request limit=100, page=1 so we get the full set of
# division-level Combined rows we need in one call.
#
# ============================================================

def fetch_cpi_data(year=None):

    if esankhyiki is None:

        print(
            "esankhyiki package not installed; "
            "skipping live MoSPI fetch."
        )

        return pd.DataFrame()

    if year is None:

        year = get_current_year()

    year = str(year)

    print(
        f"\nFetching MoSPI CPI data for {year}..."
    )

    try:

        params = {

            "base_year":
                BASE_YEAR,

            "year":
                year,

            "series":
                SERIES,

            "limit":
                API_LIMIT,

            "page":
                1,
        }

        print(
            "MoSPI API parameters:",
            params
        )

        df = esankhyiki.get_data(

            "CPI",

            params,

            format="df"

        )

        df = normalize_dataframe(df)

        if df.empty:

            print(
                "MoSPI returned an empty CPI dataframe."
            )

            return pd.DataFrame()

        print(
            f"MoSPI CPI records received: "
            f"{len(df)}"
        )

        print(
            "CPI columns:",
            list(df.columns)
        )

        # ----------------------------------------------------
        # REQUIRED COLUMNS
        # ----------------------------------------------------

        required_columns = [

            "state",
            "sector",
            "division",
            "group",
            "class",
            "sub_class",
            "item",
            "inflation",
            "month",
            "year",
        ]

        missing_columns = [

            column

            for column in required_columns

            if column not in df.columns

        ]

        if missing_columns:

            print(
                "Missing CPI columns:",
                missing_columns
            )

            return pd.DataFrame()

        # ----------------------------------------------------
        # ALL INDIA
        # ----------------------------------------------------

        df = df[
            df["state"]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            "all india"
        ].copy()

        # ----------------------------------------------------
        # COMBINED
        # ----------------------------------------------------

        df = df[
            df["sector"]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            "combined"
        ].copy()

        # ----------------------------------------------------
        # DIVISION LEVEL ONLY
        # ----------------------------------------------------
        #
        # At division level:
        #
        # group      = null
        # class      = null
        # sub_class  = null
        # item       = null
        #
        # This removes all lower-level CPI records.
        #

        df = df[
            df["group"].isna()
            &
            df["class"].isna()
            &
            df["sub_class"].isna()
            &
            df["item"].isna()
        ].copy()

        # ----------------------------------------------------
        # NUMERIC INFLATION / INDEX
        # ----------------------------------------------------

        df["inflation"] = pd.to_numeric(

            df["inflation"],

            errors="coerce"

        )

        if "index" in df.columns:

            df["index"] = pd.to_numeric(

                df["index"],

                errors="coerce"

            )

        # ----------------------------------------------------
        # REMOVE INVALID RECORDS
        # ----------------------------------------------------

        df = df.dropna(

            subset=[
                "division",
                "inflation"
            ]

        )

        # ----------------------------------------------------
        # CLEAN DIVISION NAMES
        # ----------------------------------------------------

        df["division"] = (

            df["division"]

            .astype(str)

            .str.strip()

        )

        # ----------------------------------------------------
        # DISPLAY WHAT WE GOT
        # ----------------------------------------------------

        print(
            "\nMoSPI division-level CPI data:"
        )

        display_columns = [

            "division",
            "code",
            "inflation",
            "month",
            "year",
        ]

        available_display_columns = [

            column

            for column in display_columns

            if column in df.columns

        ]

        print(

            df[
                available_display_columns
            ].to_string(
                index=False
            )

        )

        print(
            f"\nDivision records retained: "
            f"{len(df)}"
        )

        return df.reset_index(
            drop=True
        )

    except Exception as e:

        print(
            "\nMoSPI CPI error:"
        )

        print(
            repr(e)
        )

        return pd.DataFrame()


# ============================================================
# GET LATEST ROW
# ============================================================

def get_latest_row(rows):

    if rows is None or rows.empty:

        return None

    rows = rows.copy()

    # --------------------------------------------------------
    # INFLATION
    # --------------------------------------------------------

    inflation_column = find_column(

        rows,

        [
            "inflation",
            "Inflation"
        ]

    )

    if inflation_column:

        rows[inflation_column] = pd.to_numeric(

            rows[inflation_column],

            errors="coerce"

        )

        rows = rows.dropna(

            subset=[
                inflation_column
            ]

        )

    if rows.empty:

        return None

    # --------------------------------------------------------
    # YEAR
    # --------------------------------------------------------

    year_column = find_column(

        rows,

        [
            "year",
            "Year"
        ]

    )

    if year_column:

        rows["_sort_year"] = pd.to_numeric(

            rows[year_column],

            errors="coerce"

        ).fillna(0)

    else:

        rows["_sort_year"] = 0

    # --------------------------------------------------------
    # MONTH
    # --------------------------------------------------------

    month_column = find_column(

        rows,

        [
            "month",
            "Month"
        ]

    )

    if month_column:

        month_order = {

            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,

        }

        month_text = (

            rows[month_column]

            .astype(str)

            .str.strip()

            .str.lower()

        )

        rows["_sort_month"] = (

            month_text

            .map(month_order)

        )

        numeric_month = pd.to_numeric(

            rows[month_column],

            errors="coerce"

        )

        rows["_sort_month"] = (

            rows["_sort_month"]

            .fillna(numeric_month)

            .fillna(0)

        )

    else:

        rows["_sort_month"] = 0

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    rows = rows.sort_values(

        [
            "_sort_year",
            "_sort_month"
        ]

    )

    return rows.iloc[-1]


# ============================================================
# GET CPI INFLATION
# ============================================================

def get_cpi_inflation():
    """
    Fetch and normalize the latest MoSPI CPI data.

    Returns a dict with:
      available   - bool
      overall     - overall CPI (General) annual inflation %
      month/year  - latest period the overall figure covers
      categories  - {display_category_name: rate_or_None, ...}
      source      - human-readable source label
    """

    df = fetch_cpi_data()

    if df.empty:

        return {

            "available": False,

            "overall": 0.0,

            "month": "Unavailable",

            "year": "Unavailable",

            "categories": {},

            "source":
                "MoSPI CPI data unavailable",

        }

    # ========================================================
    # FIND COLUMNS
    # ========================================================

    division_column = find_column(df, ["division"])
    inflation_column = find_column(df, ["inflation"])
    month_column = find_column(df, ["month"])
    year_column = find_column(df, ["year"])

    if not division_column:

        print(
            "MoSPI CPI dataframe does not "
            "contain division column."
        )

        return {

            "available": False,
            "overall": 0.0,
            "month": "Unavailable",
            "year": "Unavailable",
            "categories": {},
            "source":
                "MoSPI CPI division data unavailable",

        }

    if not inflation_column:

        print(
            "MoSPI CPI dataframe does not "
            "contain inflation column."
        )

        return {

            "available": False,
            "overall": 0.0,
            "month": "Unavailable",
            "year": "Unavailable",
            "categories": {},
            "source":
                "MoSPI CPI inflation data unavailable",

        }

    # ========================================================
    # CLEAN
    # ========================================================

    df[division_column] = (
        df[division_column]
        .astype(str)
        .str.strip()
    )

    df[inflation_column] = pd.to_numeric(
        df[inflation_column],
        errors="coerce"
    )

    # ========================================================
    # OVERALL CPI
    # ========================================================

    general_rows = df[

        df[division_column]
        .str.strip()
        .str.lower()
        ==
        "cpi (general)"

    ]

    general_latest = get_latest_row(general_rows)

    if general_latest is None:

        overall = 0.0
        latest_month = "Unavailable"
        latest_year = "Unavailable"

    else:

        overall = safe_float(
            general_latest[inflation_column],
            0.0
        )

        latest_month = (
            str(general_latest[month_column])
            if month_column
            else "Unavailable"
        )

        latest_year = (
            str(general_latest[year_column])
            if year_column
            else "Unavailable"
        )

    # ========================================================
    # CATEGORY CPI
    # ========================================================

    categories = {}

    for division in CATEGORY_DIVISIONS:

        rows = df[

            df[division_column]
            .str.strip()
            .str.lower()
            ==
            division.lower()

        ]

        latest = get_latest_row(rows)

        display_name = DIVISION_TO_CATEGORY.get(division)

        if display_name is None:
            continue

        if latest is None:

            categories[display_name] = None

        else:

            categories[display_name] = safe_float(
                latest[inflation_column],
                None
            )

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "available": True,
        "overall": round(overall, 2),
        "month": latest_month,
        "year": latest_year,
        "categories": categories,
        "source": "MoSPI e-Sankhyiki CPI",

    }


# ============================================================
# ANNUAL -> MONTHLY
# ============================================================

def annual_to_monthly(annual_inflation):

    annual_inflation = safe_float(annual_inflation, 0.0)

    if annual_inflation <= -100:
        return 0.0

    monthly_rate = (
        (1 + annual_inflation / 100) ** (1 / 12) - 1
    ) * 100

    return monthly_rate


# ============================================================
# GET CATEGORY INFLATION
# ============================================================

def _rate_result(
    rate,
    month,
    year,
    source,
    estimated
):

    return {

        "rate": round(rate, 2),

        "monthly_rate": round(
            annual_to_monthly(rate),
            4
        ),

        "month": month,
        "year": year,
        "source": source,
        "estimated": estimated,

    }


def get_category_inflation(
    category,
    inflation_data
):
    """
    Return the inflation rate for a SterliFlux spending
    category.

    Resolution order:
      1. Real MoSPI rate for the mapped division.
      2. If MoSPI has no data for that division, an assumed/
         presumed rate from ASSUMED_CATEGORY_INFLATION.
      3. If the category can't be mapped at all ("Other" or
         unrecognized), the overall MoSPI CPI.
      4. If even overall CPI is unavailable, a general assumed
         fallback rate.

    Every result includes "estimated": True/False so callers
    can label assumed values differently from live MoSPI data.
    """

    month = "Unavailable"
    year = "Unavailable"

    if inflation_data:

        month = inflation_data.get("month", "Unavailable")
        year = inflation_data.get("year", "Unavailable")

    category_clean = str(category or "").strip().lower()

    mapped_division = CATEGORY_MAP.get(category_clean)

    overall_available = bool(
        inflation_data
        and inflation_data.get("available")
    )

    overall_rate = (
        safe_float(inflation_data.get("overall"), None)
        if inflation_data
        else None
    )

    # ========================================================
    # UNKNOWN CATEGORY OR EXPLICIT "OTHER" -> OVERALL CPI
    # ========================================================

    if mapped_division is None or mapped_division == "CPI (General)":

        if overall_available and overall_rate is not None:

            return _rate_result(
                overall_rate,
                month,
                year,
                inflation_data.get(
                    "source",
                    "MoSPI e-Sankhyiki CPI"
                ),
                estimated=False,
            )

        # Overall CPI itself unavailable — use the general
        # assumed fallback so the UI never shows a bare None.
        return _rate_result(
            ASSUMED_GENERAL_INFLATION,
            month,
            year,
            "Estimated (MoSPI overall CPI unavailable — "
            "using a presumed general inflation rate)",
            estimated=True,
        )

    # ========================================================
    # MAPPED TO A REAL MoSPI DIVISION
    # ========================================================

    display_name = DIVISION_TO_CATEGORY.get(mapped_division)

    categories = (
        inflation_data.get("categories", {})
        if inflation_data
        else {}
    )

    rate = categories.get(display_name)
    rate = safe_float(rate, None)

    if rate is not None:

        return _rate_result(
            rate,
            month,
            year,
            inflation_data.get(
                "source",
                "MoSPI e-Sankhyiki CPI"
            ),
            estimated=False,
        )

    # --------------------------------------------------------
    # MoSPI HAS NO DATA FOR THIS DIVISION.
    # Fall back to a presumed/assumed rate for the category,
    # then to overall CPI, then to the general assumption.
    # --------------------------------------------------------

    assumed_rate = ASSUMED_CATEGORY_INFLATION.get(display_name)

    if assumed_rate is not None:

        return _rate_result(
            assumed_rate,
            month,
            year,
            f"Estimated (MoSPI has no current data for "
            f"{display_name}; using a presumed rate)",
            estimated=True,
        )

    if overall_available and overall_rate is not None:

        return _rate_result(
            overall_rate,
            month,
            year,
            "Estimated (MoSPI has no category-specific data; "
            "using overall CPI)",
            estimated=True,
        )

    return _rate_result(
        ASSUMED_GENERAL_INFLATION,
        month,
        year,
        "Estimated (no MoSPI data available at all; using a "
        "presumed general inflation rate)",
        estimated=True,
    )


# ============================================================
# SIMPLE CATEGORY RATE
# ============================================================

def get_category_rate(category, inflation_data):

    result = get_category_inflation(category, inflation_data)

    return result.get("rate")


# ============================================================
# APPLY INFLATION
# ============================================================

def apply_inflation(amount, annual_inflation, months=1):

    amount = safe_float(amount, 0.0)
    annual_inflation = safe_float(annual_inflation, 0.0)
    months = safe_float(months, 1.0)

    monthly_rate = (
        (1 + annual_inflation / 100) ** (1 / 12) - 1
    )

    adjusted = amount * ((1 + monthly_rate) ** months)

    return round(adjusted, 2)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("STERLIFLUX - MoSPI CPI INFLATION ENGINE")
    print("=" * 70)

    print("\nFetching MoSPI CPI data...")

    inflation_data = get_cpi_inflation()

    print("\n" + "=" * 70)
    print("MoSPI Available:", inflation_data.get("available"))
    print("Overall CPI Inflation:", inflation_data.get("overall"), "%")
    print("Latest Month:", inflation_data.get("month"))
    print("Latest Year:", inflation_data.get("year"))

    print("\nCategory lookup test (rate / estimated / source)")
    print("=" * 70)

    test_categories = [
        "Food",
        "Groceries",
        "Shopping",
        "Transport",
        "Communication",
        "Health",
        "Education",
        "Housing",
        "Fuel",
        "Petrol",
        "Electricity",
        "Entertainment",
        "Personal Care",
        "Household",
        "Tobacco",
        "Unknown",
    ]

    for category in test_categories:

        result = get_category_inflation(category, inflation_data)

        print(f"\n{category}:")
        print("  Rate:", result.get("rate"))
        print("  Estimated:", result.get("estimated"))
        print("  Source:", result.get("source"))

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)