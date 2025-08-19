"""
dashboard.py
============

Streamlit application for exploring data collected by the social media agent.

To run the dashboard locally install Streamlit (e.g. `pip install streamlit`)
and execute the following command:

    streamlit run dashboard.py -- --db sqlite:///data.db

The dashboard connects to the same database used by the agent and displays
recent posts along with their classification and fact‑checking results.  You
can filter by platform, keyword, label and date range.  A set of summary
charts provide a quick overview of the distribution of hoax vs non‑hoax
posts.
"""

import datetime as dt
import argparse

import pandas as pd
import streamlit as st

from social_media_agent import Database, PostModel


def load_data(db: Database, limit: int = 1000) -> pd.DataFrame:
    posts = db.get_posts(limit=limit)
    records = []
    for p in posts:
        records.append({
            "platform": p.platform,
            "keyword": p.keyword,
            "content": p.content,
            "url": p.url,
            "created_at": p.created_at,
            "author": p.author,
            "predicted_label": p.predicted_label,
            "prediction_score": p.prediction_score,
            "fact_check_url": p.fact_check_url,
            "fact_check_rating": p.fact_check_rating,
            "fact_check_publisher": p.fact_check_publisher,
            "inserted_at": p.inserted_at,
        })
    df = pd.DataFrame.from_records(records)
    return df

def make_fact_link(url: str) -> str:
    if pd.notna(url) and url.strip():
        return f"[Lihat Fakta]({url})"
    return "-"


def main(db_url: str) -> None:
    st.set_page_config(page_title="Social Media Hoax Detector", layout="wide")
    st.title("Social Media Hoax Detector Dashboard")
    db = Database(db_url=db_url)
    df = load_data(db)
    if df.empty:
        st.warning("No data available. Please run the scraper first.")
        return
    # Filters
    with st.sidebar:
        st.header("Filters")
        platforms = st.multiselect("Platform", options=sorted(df['platform'].unique()), default=list(df['platform'].unique()))
        keywords = st.multiselect("Keyword", options=sorted(df['keyword'].unique()), default=list(df['keyword'].unique()))
        labels = st.multiselect("Predicted label", options=sorted(df['predicted_label'].dropna().unique()), default=list(df['predicted_label'].dropna().unique()))
        date_min = df['created_at'].min().date() if not df.empty else dt.date.today()
        date_max = df['created_at'].max().date() if not df.empty else dt.date.today()
        date_range = st.date_input("Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
    # Apply filters
    filtered = df.copy()
    if platforms:
        filtered = filtered[filtered['platform'].isin(platforms)]
    if keywords:
        filtered = filtered[filtered['keyword'].isin(keywords)]
    if labels:
        filtered = filtered[filtered['predicted_label'].isin(labels)]
    if date_range and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[(filtered['created_at'].dt.date >= start) & (filtered['created_at'].dt.date <= end)]
    # Summary
    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total posts", len(filtered))
    hoax_count = (filtered['predicted_label'] == 'hoax').sum()
    col2.metric("Hoax", hoax_count)
    non_hoax_count = (filtered['predicted_label'] == 'not_hoax').sum()
    col3.metric("Not hoax", non_hoax_count)
    # Display data
    
    st.subheader("Posts")
    # Salin dataframe untuk ditampilkan
    df_display = filtered.copy()

    # Tambahkan kolom 'Lihat Fakta'
    df_display["Lihat Fakta"] = df_display["fact_check_url"].apply(make_fact_link)

    # Pilih kolom-kolom yang ingin ditampilkan
    show_cols = [
        "created_at", "author", "keyword", "predicted_label", "prediction_score", "Lihat Fakta"
    ]

    # Urutkan dan reset index (biar rapi)
    df_display = df_display[show_cols].sort_values(by="created_at", ascending=False).reset_index(drop=True)

    # Tampilkan sebagai markdown agar link bisa diklik
    st.markdown(df_display.to_markdown(index=False), unsafe_allow_html=True)


    # Chart: distribution of labels over time
    st.subheader("Label distribution over time")
    if not filtered.empty:
        chart_df = filtered.groupby([filtered['created_at'].dt.date, 'predicted_label']).size().unstack(fill_value=0)
        st.bar_chart(chart_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Streamlit dashboard")
    parser.add_argument("--db", default="sqlite:///data.db", help="Database URL")
    args = parser.parse_args()
    main(args.db)