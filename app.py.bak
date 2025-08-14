import streamlit as st
import pandas as pd
import os
import re

# -----------------------------
# Load and clean category rules
# -----------------------------
@st.cache_data
def load_category_rules():
    df = pd.read_csv('data/category_rules.csv')
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df['category'] = df['category'].astype(str).str.strip().str.upper()

    df['min_marks'] = pd.to_numeric(df['min_marks'], errors='coerce')
    df['max_marks'] = pd.to_numeric(df['max_marks'], errors='coerce')

    # Extract rank ranges (handle commas, hyphen/en-dash)
    df['low_rank'] = df['rank_range'].str.extract(r'([\d,]+)')[0].str.replace(",", "").astype(float)
    df['high_rank'] = df['rank_range'].str.extract(r'[\–\-]\s*([\d,]+)')[0].str.replace(",", "").astype(float)

    df.dropna(subset=['low_rank', 'high_rank', 'min_marks', 'max_marks'], inplace=True)
    df['low_rank'] = df['low_rank'].astype(int)
    df['high_rank'] = df['high_rank'].astype(int)

    return df

# -----------------------------
# Load and clean college master
# -----------------------------
@st.cache_data
def load_college_master():
    df = pd.read_csv('data/college_master.csv')
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df['category'] = df['category'].astype(str).str.strip().str.upper()

    df['low_rank'] = df['rank_range'].str.extract(r'([\d,]+)')[0].str.replace(",", "").astype(float)
    df['high_rank'] = df['rank_range'].str.extract(r'[\–\-]\s*([\d,]+)')[0].str.replace(",", "").astype(float)

    df.dropna(subset=['low_rank', 'high_rank'], inplace=True)
    df['low_rank'] = df['low_rank'].astype(int)
    df['high_rank'] = df['high_rank'].astype(int)

    return df

# -----------------------------
# Save form submission to CSV
# -----------------------------
def save_submission(data):
    filepath = 'data/form_submissions.csv'
    columns = [
        "Name", "Email", "Contact", "Marks", "Category",
        "predicted_low_rank", "predicted_high_rank", "eligible_colleges"
    ]
    df = pd.DataFrame([data])
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False, columns=columns)
    else:
        df.to_csv(filepath, mode='a', header=False, index=False, columns=columns)

# -----------------------------
# Main Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="College Predictor", layout="centered")
    st.title("🎓 College Predictor")

    category_df = load_category_rules()
    college_df = load_college_master()
    available_categories = sorted(category_df['category'].dropna().unique().tolist())

    with st.form("predict_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        contact = st.text_input("Contact")
        marks = st.number_input("Marks", min_value=0.0, max_value=720.0, step=0.5)
        category = st.selectbox("Category", available_categories)

        submitted = st.form_submit_button("Predict")

    if submitted:
        if not name or not email or not contact:
            st.warning("Please fill in all the fields.")
            return

        matched_rule = category_df[
            (category_df['category'] == category) &
            (category_df['min_marks'] <= marks) &
            (category_df['max_marks'] >= marks)
        ]

        if matched_rule.empty:
            st.error("❌ No matching rank rule found for this category and marks.")
            save_submission({
                "Name": name,
                "Email": email,
                "Contact": contact,
                "Marks": marks,
                "Category": category,
                "predicted_low_rank": "",
                "predicted_high_rank": "",
                "eligible_colleges": ""
            })
            return

        low_rank = int(matched_rule.iloc[0]['low_rank'])
        high_rank = int(matched_rule.iloc[0]['high_rank'])
        st.success(f"✅ Predicted Rank Range: {low_rank:,} – {high_rank:,}")

        eligible = college_df[
            (college_df['category'] == category) &
            (college_df['low_rank'] <= high_rank) &
            (college_df['high_rank'] >= low_rank)
        ]

        eligible_colleges = eligible['institute_name'].dropna().unique().tolist()

        if eligible_colleges:
            st.subheader("🎯 Eligible Colleges")
            for college in eligible_colleges:
                st.markdown(f"- {college}")
        else:
            st.warning("No eligible colleges found for your predicted rank.")

        save_submission({
            "Name": name,
            "Email": email,
            "Contact": contact,
            "Marks": marks,
            "Category": category,
            "predicted_low_rank": low_rank,
            "predicted_high_rank": high_rank,
            "eligible_colleges": ", ".join(eligible_colleges)
        })

# -----------------------------
if __name__ == "__main__":
    main()
