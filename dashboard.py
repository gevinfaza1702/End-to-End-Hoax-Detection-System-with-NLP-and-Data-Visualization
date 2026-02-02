"""
dashboard.py
============

Streamlit application for exploring data collected by the social media agent.
Updated with enhanced visualizations, filtering, and export capabilities.
"""

import datetime as dt
import argparse
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except ImportError:
    px = None

from database import Database, PostModel


def load_data(db: Database, limit: int = 2000) -> pd.DataFrame:
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
    if pd.notna(url) and str(url).strip():
        return f"[Lihat Fakta]({url})"
    return "-"

def main(db_url: str) -> None:
    st.set_page_config(page_title="Social Media Hoax Detector", layout="wide", page_icon="🕵️")
    st.title("🕵️ Social Media Hoax Detector Dashboard")

    db = Database(db_url=db_url)
    df = load_data(db)

    if df.empty:
        st.warning("⚠️ Data belum tersedia. Silakan jalankan agent scraping terlebih dahulu.")
        st.code("python social_media_agent.py --once --source google")
        return

    # Sidebar Filters
    with st.sidebar:
        st.header("🔍 Filter Data")
        platforms = st.multiselect("Platform", options=sorted(df['platform'].unique()), default=list(df['platform'].unique()))
        keywords = st.multiselect("Topik / Keyword", options=sorted(df['keyword'].unique()), default=list(df['keyword'].unique()))
        labels = st.multiselect("Label Prediksi", options=sorted(df['predicted_label'].dropna().unique()), default=list(df['predicted_label'].dropna().unique()))

        # Date Filter
        min_date = df['created_at'].min().date()
        max_date = df['created_at'].max().date()
        date_range = st.date_input("Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    # Apply Filters
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

    # Tabs Layout
    tab1, tab2, tab3 = st.tabs(["📊 Ringkasan", "🔎 Eksplorasi Data", "📈 Analisis Mendalam"])

    # --- TAB 1: RINGKASAN ---
    with tab1:
        st.subheader("Statistik Utama")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Artikel", len(filtered))
        c2.metric("Terdeteksi Hoaks", len(filtered[filtered['predicted_label'] == 'hoax']), delta_color="inverse")
        c3.metric("Bukan Hoaks (Fakta)", len(filtered[filtered['predicted_label'] == 'not_hoax']))
        c4.metric("Terverifikasi Fakta (Google)", len(filtered[filtered['fact_check_url'].notna()]))

        st.divider()

        # Simple Charts (Streamlit Native)
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("Tren Hoaks per Hari")
            if not filtered.empty:
                daily_counts = filtered[filtered['predicted_label'] == 'hoax'].groupby(filtered['created_at'].dt.date).size()
                st.line_chart(daily_counts)

        with col_chart2:
            st.subheader("Sebaran Platform")
            platform_counts = filtered['platform'].value_counts()
            st.bar_chart(platform_counts)

    # --- TAB 2: EKSPLORASI DATA ---
    with tab2:
        st.subheader("Daftar Artikel & Postingan")

        # Search Box
        search_query = st.text_input("Cari judul atau isi konten...", placeholder="Ketik kata kunci...")
        if search_query:
            filtered = filtered[filtered['content'].str.contains(search_query, case=False, na=False)]

        # Display Table
        df_display = filtered.copy()
        df_display["Lihat Fakta"] = df_display["fact_check_url"].apply(make_fact_link)

        # Select and rename columns for display
        # Note: We keep 'fact_check_url' for the LinkColumn config to work
        cols_to_show = ["created_at", "platform", "keyword", "predicted_label", "prediction_score", "fact_check_url", "content"]
        df_display = df_display[cols_to_show].sort_values(by="created_at", ascending=False).reset_index(drop=True)

        # Style the dataframe (highlight hoax)
        def highlight_hoax(row):
            if row.predicted_label == 'hoax':
                return ['background-color: #ffe6e6; color: black'] * len(row)
            return [''] * len(row)

        st.dataframe(
            df_display.style.apply(highlight_hoax, axis=1),
            column_config={
                "fact_check_url": st.column_config.LinkColumn("Link Fakta", display_text="Buka Link"),
                "prediction_score": st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0, max_value=1),
            },
            use_container_width=True
        )

        # Download Button
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data (CSV)",
            data=csv,
            file_name='laporan_hoaks.csv',
            mime='text/csv',
        )

    # --- TAB 3: ANALISIS MENDALAM ---
    with tab3:
        if px is None:
            st.warning("Library 'plotly' tidak terinstall. Jalankan `pip install plotly` untuk melihat grafik interaktif.")
        elif not filtered.empty:
            st.subheader("Visualisasi Interaktif")

            c_a, c_b = st.columns(2)

            with c_a:
                st.markdown("#### Proporsi Hoaks vs Fakta")
                fig_pie = px.pie(filtered, names='predicted_label', title='Persentase Label',
                                 color='predicted_label',
                                 color_discrete_map={'hoax':'red', 'not_hoax':'green'})
                st.plotly_chart(fig_pie, use_container_width=True)

            with c_b:
                st.markdown("#### Topik Paling Banyak Hoaks")
                hoax_only = filtered[filtered['predicted_label'] == 'hoax']
                if not hoax_only.empty:
                    topic_counts = hoax_only['keyword'].value_counts().reset_index()
                    topic_counts.columns = ['Topik', 'Jumlah Hoaks']
                    fig_bar = px.bar(topic_counts, x='Topik', y='Jumlah Hoaks', color='Topik')
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("Belum ada data hoaks untuk ditampilkan.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the Streamlit dashboard")
    parser.add_argument("--db", default="sqlite:///data.db", help="Database URL")
    args = parser.parse_args()
    main(args.db)
