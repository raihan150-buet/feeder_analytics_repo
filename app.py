import streamlit as st
import pandas as pd
from modules.data_loader import load_data
from modules.analyzer import StreamlitFeederAnalysis
from modules.ui_components import (
    executive_summary_page,
    setup_sidebar,
    render_basic_stats,
    render_trend_analysis,
    render_seasonality_analysis,
    render_region_analysis,
    render_anomaly_analysis,
    render_clustering_analysis,
    render_forecasting,
    render_bulk_forecasting,
    render_comparison,
    render_correlation
)

def main():
    st.title("⚡ DPDC Feeder Consumption Analytics Dashboard ")
    st.markdown("---")

    uploaded_file = st.sidebar.file_uploader(
        "Upload the 'Analysis.xlsx' file (Must contain 'Linked_11KV' sheet)",
        type=['xlsx']
    )

    if uploaded_file is None:
        st.info("Please upload your data file in the sidebar to begin the analysis.")
        return

    df, date_cols = load_data(uploaded_file)
    if df is None or df.empty:
        if uploaded_file:
            st.error("Data loading failed. Please ensure the file is correct and contains the 'Linked_11KV' sheet.")
        return

    analyzer = StreamlitFeederAnalysis(df, date_cols)
    
    # Setup sidebar and get selected module
    selected_module = setup_sidebar(analyzer, df, date_cols)
    
    # Route to appropriate module
    module_handlers = {
        "executive_summary": lambda: executive_summary_page(analyzer),
        "basic_stats": lambda: render_basic_stats(analyzer),
        "trend_analysis": lambda: render_trend_analysis(analyzer),
        "seasonality_analysis": lambda: render_seasonality_analysis(analyzer),
        "region_analysis": lambda: render_region_analysis(analyzer),
        "anomaly_analysis": lambda: render_anomaly_analysis(analyzer),
        "clustering_analysis": lambda: render_clustering_analysis(analyzer),
        "forecasting": lambda: render_forecasting(analyzer),
        "bulk_forecasting": lambda: render_bulk_forecasting(analyzer),
        "comparison": lambda: render_comparison(analyzer),
        "correlation": lambda: render_correlation(analyzer)
    }
    
    if selected_module in module_handlers:
        module_handlers[selected_module]()

if __name__ == '__main__':
    main()
