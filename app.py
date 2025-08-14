import streamlit as st
import pandas as pd
import os
import re
import math

# -----------------------------
# Config / file paths
# -----------------------------
DATA_DIR = "data"
CATEGORY_RULES_PATH = os.path.join(DATA_DIR, "category_rules.csv")
COLLEGE_MASTER_PATH = os.path.join(DATA_DIR, "college_master.csv")
SUBMISSIONS_PATH = os.path.join(DATA_DIR, "form_submissions.csv")  # appended to

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
    def safe_read(p):
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, dtype=str).fillna('')
            except Exception:
                # if CSV malformed, return empty DataFrame
                df = pd.DataFrame()
            return df
        return pd.DataFrame()
    cat_rules = safe_read(CATEGORY_RULES_PATH)
    college = safe_read(COLLEGE_MASTER_PATH)
    submissions = safe_read(SUBMISSIONS_PATH)

    def norm_cols(df):
        if df is None or df.empty:
            return df
        df = df.copy()
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    cat_rules = norm_cols(cat_rules)
    college = norm_cols(college)
    submissions = norm_cols(submissions)

    if cat_rules is not None and not cat_rules.empty and 'category' in cat_rules.columns:
        cat_rules['_cat_canonical'] = cat_rules['category'].apply(canonicalize_cat)
    if college is not None and not college.empty and 'category' in college.columns:
        college['_cat_canonical'] = college['category'].apply(canonicalize_cat)

    return cat_rules, college, submissions

# -----------------------------
# Rank parsing and mapping
# -----------------------------
def parse_rank_range(rank_str):
    """
    Parse strings like '1-100', '150,001+', '100000' into (low, high).
    Returns (nan, nan) if unparsable.
    """
    if not isinstance(rank_str, str) or rank_str.strip() == "":
        return (math.nan, math.nan)
    s = rank_str.replace(',', '').strip()
    if s.lower() in ("na", "n/a", "-", ""):
        return (math.nan, math.nan)
    # trailing plus: "150001+"
    if '+' in s:
        digits = re.findall(r'(\d+)', s)
        if digits:
            return (int(digits[0]), float('inf'))
    # hyphenated range
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
    If rank_col is provided and exists use it; otherwise detect a sensible column
    (e.g., 'closing_rank', 'rank_range', 'rank range', 'cutoff', etc.).
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
        # prefer patterns that look explicit
        preferred = [c for c in candidates if re.search(r'closing|closing_rank|rank_range|rank_range|rank range|closing rank|rank_range', c, re.I)]
        if preferred:
            col_to_use = preferred[0]
        elif candidates:
            col_to_use = candidates[0]
        else:
            col_to_use = None

    if not col_to_use:
        # ensure columns exist so later code won't crash
        if 'rank_low' not in df.columns:
            df['rank_low'] = float('nan')
        if 'rank_high' not in df.columns:
            df['rank_high'] = float('nan')
        return df

    # Parse the chosen column into numeric low/high
    def _parse_cell(x):
        s = str(x).strip()
        if s == "" or s.lower() in ("na", "n/a", "-"):
            return (math.nan, math.nan)
        return parse_rank_range(s)

    parsed = df[col_to_use].astype(str).apply(_parse_cell)
    df['rank_low'] = parsed.apply(lambda t: t[0])
    df['rank_high'] = parsed.apply(lambda t: t[1])
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
    df = add_parsed_rank_cols(college_df)

    # Guarantee rank_low/rank_high exist
    if 'rank_low' not in df.columns:
        df['rank_low'] = float('nan')
    if 'rank_high' not in df.columns:
        df['rank_high'] = float('nan')

    # Ensure canonical category column exists if raw category exists
    if 'category' in df.columns and '_cat_canonical' not in df.columns:
        df['_cat_canonical'] = df['category'].apply(canonicalize_cat)

    cat = canonicalize_cat(candidate_category)

    # Filter by category where possible
    if '_cat_canonical' in df.columns:
        df_cat = df[df['_cat_canonical'] == cat].copy()
    else:
        df_cat = df.copy()

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

    # Fallback to UR rows if category-specific rows empty
    if matches.empty and '_cat_canonical' in df.columns:
        ur_df = df[df['_cat_canonical'] == 'ur'].copy()
        matches = ur_df[ur_df.apply(overlap, axis=1)].copy()
        if not matches.empty:
            matches['match_confidence'] = 'fallback_ur'

    # If still empty, try matching ignoring category (best-effort)
    if matches.empty:
        all_df = df.copy()
        matches = all_df[all_df.apply(overlap, axis=1)].copy()
        if not matches.empty:
            matches['match_confidence'] = 'ignore_category'

    # Default confidence label for exact category matches
    if not matches.empty and 'match_confidence' not in matches.columns:
        matches['match_confidence'] = 'exact'

    # Ensure expected columns exist
    for c in ('rank_low', 'rank_high', 'match_confidence'):
        if c not in matches.columns:
            matches[c] = float('nan') if 'rank' in c else ''

    # Try to sort by rank_low; otherwise return unsorted matches
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
    unparsable = []
    if college_df is not None and not college_df.empty:
        # try to detect parsed rank column and collect unparsable rows
        parsed_col = None
        for c in college_df.columns:
            if '_parsed_rank_column' in c or re.search(r'rank|range|cutoff|closing|merit', c, re.I):
                # if created previously by add_parsed_rank_cols, use it; else try 'rank_range' or 'closing_rank'
                if c == '_parsed_rank_column':
                    parsed_col = c
                    break
        # fallback: run parser on detected rank-like column(s)
        rank_candidates = [c for c in college_df.columns if re.search(r'rank|range|cutoff|closing|merit', c, re.I)]
        if rank_candidates:
            for idx, row in college_df.iterrows():
                parsed = parse_rank_range(str(row.get(rank_candidates[0], "")))
                if math.isnan(parsed[0]) and math.isnan(parsed[1]):
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
    #st.sidebar.header("Admin / Debug")
    #admin_strict = st.sidebar.checkbox("Enforce strict category rules (no fallback)", value=False)
    #show_reconcile = st.sidebar.button("Generate reconciliation report")

    #if show_reconcile:
        #r1, r2 = reconcile_and_save_reports(cat_rules, college)
        #st.sidebar.success("Reconciliation reports saved:")
        #st.sidebar.write(r1)
        #st.sidebar.write(r2)

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
            # Pick sensible display columns (normalized column names)
            name_candidates = ['institute_name', 'institute', 'college_name', 'college', 'name']
            rank_candidates = ['closing_rank', 'rank_range', 'rank', 'range', 'cutoff']

            name_col = next((c for c in matches.columns if c in name_candidates), None)
            rank_col = next((c for c in matches.columns if c in rank_candidates), None)

            # fuzzy lookup if exact not found (e.g., 'institute name' -> 'institute_name' already normalized)
            if name_col is None:
                for c in matches.columns:
                    if c.lower().replace(' ', '_') in name_candidates:
                        name_col = c
                        break
            if rank_col is None:
                for c in matches.columns:
                    lc = c.lower().replace(' ', '_')
                    if 'rank' in lc or 'range' in lc or 'cutoff' in lc:
                        rank_col = c
                        break

            display_cols = []
            if name_col:
                display_cols.append(name_col)
            if rank_col and rank_col not in display_cols:
                display_cols.append(rank_col)

            for c in ['rank_low', 'rank_high', 'match_confidence', '_parsed_rank_column']:
                if c in matches.columns and c not in display_cols:
                    display_cols.append(c)

            if not display_cols:
                st.write(f"Top {min(20, len(matches))} matching colleges (raw rows):")
                st.dataframe(matches.head(50))
            else:
                st.write(f"Top {min(20, len(matches))} matching colleges:")
                st.dataframe(matches[display_cols].head(50))

        # Save submission
        eligible_colleges = []
        # try to append the name_col values to the submission if present
        try:
            if not matches.empty:
                # choose first sensible name column for storing list
                name_candidates_store = ['institute_name', 'institute', 'college_name', 'college', 'name']
                store_name_col = next((c for c in matches.columns if c in name_candidates_store), None)
                if store_name_col:
                    eligible_colleges = matches[store_name_col].astype(str).tolist()
                else:
                    eligible_colleges = matches.index.astype(str).tolist()
        except Exception:
            eligible_colleges = []

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
