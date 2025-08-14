import streamlit as st
import pandas as pd
import os
import re
import math
from io import StringIO

# -----------------------------
# Config / file paths
# -----------------------------
# Keep these relative to the project root where `data/` exists
DATA_DIR = "data"
CATEGORY_RULES_PATH = os.path.join(DATA_DIR, "category_rules.csv")
COLLEGE_MASTER_PATH = os.path.join(DATA_DIR, "college_master.csv")
SUBMISSIONS_PATH = os.path.join(DATA_DIR, "form_submissions.csv")  # optional - appended to

# -----------------------------
# Utilities: canonicalization
# -----------------------------
def canonicalize_cat(s):
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r'[\s\-_]+', ' ', s)
    alias_map = {
        'open': 'ur',
        'general': 'ur',
        'unreserved': 'ur',
        'o': 'ur',
        'ur': 'ur',
        'sc': 'sc',
        'st': 'st',
        'obc': 'obc',
        'obc-ncl': 'obc',
        'ews': 'ews',
        'pwd': 'pwd',
        'nri': 'nri',
    }
    return alias_map.get(s, s)

# -----------------------------
# Load & normalize CSVs
# -----------------------------
@st.cache_data
def load_csvs():
    # load with safe defaults
    def safe_read(p):
        if os.path.exists(p):
            return pd.read_csv(p, dtype=str).fillna('')
        return pd.DataFrame()
    cat_rules = safe_read(CATEGORY_RULES_PATH)
    college = safe_read(COLLEGE_MASTER_PATH)
    submissions = safe_read(SUBMISSIONS_PATH)
    # normalize column names
    def norm_cols(df):
        df = df.copy()
        if df.empty:
            return df
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df
    cat_rules = norm_cols(cat_rules)
    college = norm_cols(college)
    submissions = norm_cols(submissions)
    # add canonical category columns if present
    if 'category' in cat_rules.columns:
        cat_rules['_cat_canonical'] = cat_rules['category'].apply(canonicalize_cat)
    if 'category' in college.columns:
        college['_cat_canonical'] = college['category'].apply(canonicalize_cat)
    return cat_rules, college, submissions

# -----------------------------
# Rank parsing and mapping
# -----------------------------
def parse_rank_range(rank_str):
    if not isinstance(rank_str, str) or rank_str.strip() == "":
        return (math.nan, math.nan)
    s = rank_str.replace(',', '').strip()
    # handle trailing '+'
    if '+' in s:
        digits = re.findall(r'(\d+)', s)
        if digits:
            return (int(digits[0]), float('inf'))
    # handle hyphen range
    m = re.match(r'^\s*(\d+)\s*[-–]\s*(\d+)\s*$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    # single number
    digits = re.findall(r'(\d+)', s)
    if len(digits) == 1:
        v = int(digits[0])
        return (v, v)
    return (math.nan, math.nan)

def add_parsed_rank_cols(df, rank_col=None):
    """
    Parse rank ranges into numeric 'rank_low' and 'rank_high'.
    If rank_col is provided we use it; otherwise detect a sensible column
    (e.g., 'closing_rank', 'rank range', 'Rank Range', 'cutoff', etc.).
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # If explicit rank_col provided and exists, use it
    if rank_col and rank_col in df.columns:
        col_to_use = rank_col
    else:
        # Detect rank-like column names
        candidates = [c for c in df.columns if re.search(r'rank|range|cutoff|closing|merit', c, re.I)]
        # prefer exact patterns
        preferred = [c for c in candidates if re.search(r'closing|closing_rank|rank_range|rank range|closing rank', c, re.I)]
        if preferred:
            col_to_use = preferred[0]
        elif candidates:
            col_to_use = candidates[0]
        else:
            col_to_use = None

    # If no rank column found, ensure rank_low/rank_high exist and return
    if not col_to_use:
        if 'rank_low' not in df.columns:
            df['rank_low'] = float('nan')
        if 'rank_high' not in df.columns:
            df['rank_high'] = float('nan')
        return df

    # Parse the chosen column into numeric low/high
    def _parse_cell(x):
        s = str(x).strip()
        # reuse parse_rank_range logic (inline to avoid import issues)
        if s == "" or s.lower() in ("na", "n/a", "-"):
            return (math.nan, math.nan)
        t = s.replace(',', '')
        if '+' in t:
            m = re.findall(r'(\d+)', t)
            return (int(m[0]), float('inf')) if m else (math.nan, math.nan)
        m = re.match(r'^\s*(\d+)\s*[-–]\s*(\d+)\s*$', t)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        digits = re.findall(r'(\d+)', t)
        if len(digits) == 1:
            v = int(digits[0])
            return (v, v)
        return (math.nan, math.nan)

    parsed = df[col_to_use].astype(str).apply(_parse_cell)
    df['rank_low'] = parsed.apply(lambda t: t[0])
    df['rank_high'] = parsed.apply(lambda t: t[1])
    # keep a helpful column to know what we parsed
    df['_parsed_rank_column'] = col_to_use
    return df


def marks_to_rank_range(marks):
    m = int(marks)
    if m >= 675:
        return "1-100"
    elif 665 <= m <= 674:
        return "101-500"
    elif 645 <= m <= 664:
        return "501-1000"
    elif 618 <= m <= 644:
        return "1001-5000"
    elif 589 <= m <= 617:
        return "5001-10000"
    elif 560 <= m <= 588:
        return "10001-20000"
    elif 532 <= m <= 559:
        return "20001-35000"
    elif 503 <= m <= 531:
        return "35001-50000"
    elif 475 <= m <= 502:
        return "50001-75000"
    elif 445 <= m <= 474:
        return "75001-100000"
    elif 415 <= m <= 444:
        return "100001-150000"
    else:
        return "150001+"

def marks_to_rank_tuple(marks):
    rng = marks_to_rank_range(marks)
    if '+' in rng:
        low = int(rng.replace(',', '').replace('+', ''))
        return (low, float('inf'))
    low, high = [int(x.replace(',', '')) for x in rng.split('-')]
    return (low, high)

# -----------------------------
# Category resolution & matching
# -----------------------------
def resolve_category_rule(cat, cat_rules_df, fallback_to_ur=True):
    cat_can = canonicalize_cat(cat)
    if cat_rules_df is None or cat_rules_df.empty:
        return 'ur' if fallback_to_ur else None
    available = set(cat_rules_df['_cat_canonical'].unique())
    if cat_can in available:
        return cat_can
    if cat_can == 'nri':
        return 'ur' if fallback_to_ur else None
    return 'ur' if fallback_to_ur else None

def find_matching_colleges(candidate_marks, candidate_category, college_df):
    """
    Return dataframe of matching colleges for candidate_marks and candidate_category.
    Uses add_parsed_rank_cols which auto-detects the rank column.
    """
    if college_df is None or college_df.empty:
        return pd.DataFrame()

    cand_low, cand_high = marks_to_rank_tuple(candidate_marks)

    # Ensure rank_low/rank_high exist by parsing available rank-like column
    df = add_parsed_rank_cols(college_df)  # auto-detects rank column

    # Guarantee rank_low / rank_high columns exist
    if 'rank_low' not in df.columns:
        df['rank_low'] = float('nan')
    if 'rank_high' not in df.columns:
        df['rank_high'] = float('nan')

    # Ensure there is a canonical category column if category exists in data
    if 'category' in df.columns and '_cat_canonical' not in df.columns:
        df['_cat_canonical'] = df['category'].apply(canonicalize_cat)

    cat = canonicalize_cat(candidate_category)

    # Filter rows that belong to the candidate category (if category info exists)
    if '_cat_canonical' in df.columns:
        df_cat = df[df['_cat_canonical'] == cat].copy()
    else:
        df_cat = df.copy()

    # Overlap check between candidate range and college range
    def overlap(row):
        try:
            rl = row.get('rank_low', float('nan'))
            rh = row.get('rank_high', float('nan'))
            rl = float(rl) if rl not in ("", None) and not pd.isna(rl) else float('nan')
            rh = float(rh) if rh not in ("", None) and not pd.isna(rh) else float('nan')
            if math.isnan(rl) or math.isnan(rh):
                return False
            return not (cand_high < rl or cand_low > rh)
        except Exception:
            return False

    matches = df_cat[df_cat.apply(overlap, axis=1)].copy()

    # If no category-specific matches, try fallback to UR rows (if present)
    if matches.empty and '_cat_canonical' in df.columns:
        ur_df = df[df['_cat_canonical'] == 'ur'].copy()
        matches = ur_df[ur_df.apply(overlap, axis=1)].copy()
        if not matches.empty:
            matches['match_confidence'] = 'fallback_ur'

    # If still empty, try matching across all rows using parsed rank (ignore category)
    if matches.empty:
        all_df = df.copy()
        matches = all_df[all_df.apply(overlap, axis=1)].copy()
        if not matches.empty:
            matches['match_confidence'] = 'ignore_category'

    # Default confidence label for exact category matches
    if not matches.empty and 'match_confidence' not in matches.columns:
        matches['match_confidence'] = 'exact'

    # Ensure expected columns exist for display/sorting
    if 'rank_low' not in matches.columns:
        matches['rank_low'] = float('nan')
    if 'rank_high' not in matches.columns:
        matches['rank_high'] = float('nan')
    if 'match_confidence' not in matches.columns:
        matches['match_confidence'] = ''

    # Try to sort by rank_low; if that fails just return unsorted frame
    try:
        return matches.sort_values(by=['rank_low'])
    except Exception:
        return matches

# -----------------------------
# Reconciliation helpers
# -----------------------------
def find_category_mismatches(cat_rules_df, college_df):
    if (cat_rules_df is None or cat_rules_df.empty) and (college_df is None or college_df.empty):
        return [], []
    cr = cat_rules_df.copy() if cat_rules_df is not None else pd.DataFrame()
    cm = college_df.copy() if college_df is not None else pd.DataFrame()
    if not cr.empty and 'category' in cr.columns:
        cr['_cat_canonical'] = cr['category'].apply(canonicalize_cat)
    if not cm.empty and 'category' in cm.columns:
        cm['_cat_canonical'] = cm['category'].apply(canonicalize_cat)
    rules_set = set(cr['_cat_canonical'].unique()) if not cr.empty else set()
    college_set = set(cm['_cat_canonical'].unique()) if not cm.empty else set()
    only_in_rules = sorted(list(rules_set - college_set))
    only_in_colleges = sorted(list(college_set - rules_set))
    return only_in_rules, only_in_colleges

def reconcile_and_save_reports(cat_rules_df, college_df, out_dir="reconcile_reports"):
    os.makedirs(out_dir, exist_ok=True)
    only_in_rules, only_in_colleges = find_category_mismatches(cat_rules_df, college_df)
    pd.DataFrame({'only_in_rules': only_in_rules}).to_csv(os.path.join(out_dir, 'only_in_rules.csv'), index=False)
    pd.DataFrame({'only_in_colleges': only_in_colleges}).to_csv(os.path.join(out_dir, 'only_in_colleges.csv'), index=False)
    # unparsable ranks
    unparsable = []
    if college_df is not None and not college_df.empty and 'closing_rank' in college_df.columns:
        for idx, row in college_df.iterrows():
            low, high = parse_rank_range(str(row.get('closing_rank', '')))
            if math.isnan(low) and math.isnan(high):
                unr = row.to_dict()
                unr['_row_index'] = idx
                unparsable.append(unr)
        if unparsable:
            pd.DataFrame(unparsable).to_csv(os.path.join(out_dir, 'unparsable_college_ranks.csv'), index=False)
    return os.path.join(out_dir, 'only_in_rules.csv'), os.path.join(out_dir, 'only_in_colleges.csv')

# -----------------------------
# Persist submissions (append)
# -----------------------------
def save_submission(record):
    # create directory if not exists
    os.makedirs(DATA_DIR, exist_ok=True)
    path = SUBMISSIONS_PATH
    df_row = pd.DataFrame([record])
    if os.path.exists(path):
        try:
            df_existing = pd.read_csv(path, dtype=str)
            df_out = pd.concat([df_existing, df_row], ignore_index=True)
        except Exception:
            df_out = df_row
    else:
        df_out = df_row
    df_out.to_csv(path, index=False)

# -----------------------------
# Streamlit UI: main
# -----------------------------
def main():
    st.set_page_config(page_title="Medical College Predictor", layout="wide")
    st.title("Medical College Predictor")

    cat_rules, college, submissions = load_csvs()

    # Admin controls
    st.sidebar.header("Admin / Debug")
    admin_strict = st.sidebar.checkbox("Enforce strict category rules (no fallback)", value=False)
    show_reconcile = st.sidebar.button("Generate reconciliation report")

    if show_reconcile:
        r1, r2 = reconcile_and_save_reports(cat_rules, college)
        st.sidebar.success("Reconciliation reports saved:")
        st.sidebar.write(r1)
        st.sidebar.write(r2)

    # Input form
    with st.form("predict_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        contact = st.text_input("Contact")
        marks = st.number_input("Enter NEET Marks (numeric)", min_value=0, max_value=800, step=1, value=0)
        category = st.text_input("Category (e.g., Open, UR, OBC, SC, ST, NRI)", value="Open")
        submitted = st.form_submit_button("Predict")

    if submitted:
        resolved_cat = resolve_category_rule(category, cat_rules, fallback_to_ur=not admin_strict)
        if resolved_cat is None:
            st.error("No category rule found and strict mode is enabled. Please ask admin to provide rules for this category.")
            return
        st.info(f"Using category: {resolved_cat} (input: {category})")
        rng = marks_to_rank_range(marks)
        st.write("Estimated Rank Range:", rng)
        low_rank, high_rank = marks_to_rank_tuple(marks)
        matches = find_matching_colleges(marks, resolved_cat, college)
        if matches.empty:
            st.warning("No matches found for this category and rank range. If you used a non-UR category, consider allowing fallback to UR in Admin panel.")
        else:
            display_cols = [c for c in ['college_name','closing_rank','rank_low','rank_high','match_confidence'] if c in matches.columns]
            st.write(f"Top {min(20, len(matches))} matching colleges:")
            st.dataframe(matches[display_cols].head(50))

        # Save submission
        eligible_colleges = matches['college_name'].tolist() if 'college_name' in matches.columns else []
        save_submission({
            "name": name,
            "email": email,
            "contact": contact,
            "marks": marks,
            "category": category,
            "resolved_category": resolved_cat,
            "predicted_low_rank": low_rank,
            "predicted_high_rank": high_rank,
            "eligible_colleges": ";".join(eligible_colleges)
        })
        st.success("Submission saved.")

if __name__ == "__main__":
    main()
