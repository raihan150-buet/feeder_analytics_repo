import streamlit as st
import pandas as pd
import re

def format_consumption_column_regex(col):
    """Clean column names to YYYY-MM format."""
    if 'Consumption' not in col:
        return col

    pattern = r'^([A-Za-z]+)(\d{2})_Consumption'
    match = re.match(pattern, col)

    if match:
        month_str, year_part = match.groups()
        month_str = month_str.upper()

        MONTH_MAP = {
            'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04', 'MAY': '05', 'JUN': '06',
            'JUL': '07', 'AUG': '08', 'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12',
            'JANUARY': '01', 'FEBRUARY': '02', 'MARCH': '03', 'APRIL': '04',
            'MAY': '05', 'JUNE': '06', 'JULY': '07', 'AUGUST': '08',
            'SEPTEMBER': '09', 'OCTOBER': '10', 'NOVEMBER': '11', 'DECEMBER': '12'
        }

        for month_name, month_num in MONTH_MAP.items():
            if month_str.startswith(month_name):
                return f'20{year_part}-{month_num}'

    return col

@st.cache_data(show_spinner="Loading and cleaning raw data...")
def load_data(uploaded_file):
    """
    Loads, cleans, and pre-processes the Excel data.
    Returns (DataFrame, list of date columns).
    """
    if uploaded_file is None:
        return None, []

    try:
        df = pd.read_excel(uploaded_file, sheet_name="Linked_11KV", usecols="A:AF")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, []

    df.columns = df.columns.str.strip()
    
    exclusion_feeder = ['AT-1','AT-2','AT-3','AT-4']
    exclusion_nocs = [0,'Jigatola OLD 33/11KV S/S']
    df = df[(~df['Feeder_Name'].isin(exclusion_feeder)) & (~df['NOCS'].isin(exclusion_nocs))].copy()

    if 'SL' in df.columns:
        df = df.drop('SL', axis=1)

    rename_map = {
        'Feeder_Name': 'feeder_name',
        'NOCS': 'nocs',
        'Substation_Name': 'substation_name'
    }
    df = df.rename(columns=rename_map)

    df.columns = [format_consumption_column_regex(col) for col in df.columns]

    date_cols = sorted([
        col for col in df.columns
        if len(col) == 7 and col[4] == '-' and col.replace('-', '').isdigit()
    ])

    for col in date_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[date_cols] = df[date_cols].fillna(0)

    # Return the DataFrame and the list separately
    return df, date_cols
