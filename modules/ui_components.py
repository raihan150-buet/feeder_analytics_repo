# modules/ui_components.py
import streamlit as st
import pandas as pd
from datetime import datetime
from modules.reporting import create_excel_report, generate_html_report
from modules.visualization import (
    plot_basic_statistics, 
    plot_trend_analysis, 
    plot_seasonality_analysis,
    plot_clustering_analysis,
    compare_feeders
)
from modules.forecasting import forecast_feeder, bulk_forecasting

def format_number(num):
    """Formats large numbers for display in metrics."""
    if num >= 1e9:
        return f'{num / 1e9:,.2f} B'
    if num >= 1e6:
        return f'{num / 1e6:,.2f} M'
    return f'{num:,.0f}'

def executive_summary_page(analyzer):
    """Displays the Executive Summary page."""
    
    st.subheader("Key Operational Insights ⚡")

    stats_df = analyzer.stats_df
    trend_df = analyzer.trend_df
    load_factor_df = analyzer.load_factor_df
    
    total_feeders = len(analyzer.df)
    total_consumption = stats_df['total_consumption'].sum()
    avg_monthly_consumption = stats_df['avg_monthly'].mean()
    
    if len(analyzer.date_columns) >= 3:
        recent_3_months = analyzer.date_columns[-3:]
        inactive_mask = (analyzer.df[recent_3_months] == 0).all(axis=1)
        inactive_count = inactive_mask.sum()
    else:
        inactive_count = 0

    high_cv_count = (stats_df['cv'] > 1.0).sum()
    low_lf_count = (load_factor_df['load_factor'] < 50).sum()
    
    pos_growth_count = (trend_df['growth_rate_%'] > 0).sum()
    neg_growth_count = (trend_df['growth_rate_%'] < 0).sum()
    
    fastest_growing = trend_df.nlargest(1, 'growth_rate_%')
    fastest_growing_name = fastest_growing['feeder_name'].values[0] if len(fastest_growing) > 0 else "N/A"
    fastest_growing_rate = fastest_growing['growth_rate_%'].values[0] if len(fastest_growing) > 0 else 0

    st.markdown("#### Core Dataset Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Feeders Analyzed", f"{total_feeders:,}")
    col2.metric("Total Consumption (Period)", format_number(total_consumption), delta="Full historical period")
    col3.metric("Avg Monthly Consump/Feeder", format_number(avg_monthly_consumption))
    col4.metric("Analysis Period", f"{analyzer.date_columns[0]} to {analyzer.date_columns[-1]}")

    st.markdown("---")
    
    st.markdown("#### Operational and Trend Indicators")
    colA, colB, colC = st.columns(3)
    
    colA.metric("Inactive Feeders (Last 3M)", f"{inactive_count:,}", help="Feeders with zero consumption in the three most recent months.")
    colB.metric("High Variability Feeders (CV > 1)", f"{high_cv_count:,}", help="Feeders whose consumption varies significantly month-to-month.")
    colC.metric("Low Load Factor Feeders (<50%)", f"{low_lf_count:,}", help="Feeders that are potentially underutilized or have highly fluctuating peak demand.")
    
    st.markdown("---")
    
    st.markdown("#### Growth Summary (Recent 6M vs Initial 6M)")
    colD, colE, colF = st.columns(3)

    colD.metric("Feeders Showing Positive Growth", f"{pos_growth_count:,}")
    colE.metric("Feeders Showing Negative Growth", f"{neg_growth_count:,}")
    colF.metric("Fastest Growing Feeder", f"{fastest_growing_name}", delta=f"{fastest_growing_rate:.2f}% Growth")

    st.markdown("---")
    
    st.markdown("#### Top 10 Feeders by Total Consumption")
    top_10 = stats_df.nlargest(10, 'total_consumption')[[
        'feeder_name', 'substation', 'nocs', 'total_consumption'
    ]]
    st.dataframe(top_10.style.format({'total_consumption': '{:,.0f}'}), width='stretch')
    
    st.markdown("#### Recommendations")
    st.info("""
    * **Investigate Inactive Feeders:** Determine if they are decommissioned or require reconnection/metering checks.
    * **Monitor High-Growth/Peak Feeders:** Ensure capacity planning is aligned with future demand.
    * **Review Low Load Factor Feeders:** Potential for demand-side management or load balancing optimization.
    """)

def setup_sidebar(analyzer, df, date_cols):
    """Setup sidebar and return selected module."""
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Select Analysis Module")
    
    analysis_options = {
        "🏠 Executive Summary": "executive_summary",
        "📈 Basic Statistics & Distribution": "basic_stats",
        "📊 Trend & Growth Analysis": "trend_analysis",
        "🌡️ Seasonality & Time Series": "seasonality_analysis",
        "🏢 Substation & NOCS Performance": "region_analysis",
        "⚠️ Anomaly & Load Factor Analysis": "anomaly_analysis",
        "🎯 Clustering of Feeder Patterns": "clustering_analysis",
        "🔮 Individual Feeder Forecasting": "forecasting",
        "⚙️ Bulk Feeder Forecasting": "bulk_forecasting",
        "🔍 Feeder Comparison Tool": "comparison",
        "🔗 Correlation Analysis": "correlation"
    }
    
    selected_analysis = st.sidebar.radio("Modules", list(analysis_options.keys()))
    module_key = analysis_options[selected_analysis]

    st.sidebar.markdown("---")
    
    # Export Reports Section
    st.sidebar.header("📥 Export Reports")
    
    col_excel, col_html = st.sidebar.columns(2)
    
    with col_excel:
        excel_data = create_excel_report(analyzer)
        st.download_button(
            label="📊 Excel",
            data=excel_data,
            file_name=f'DPDC_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key='download-excel'
        )
    
    with col_html:
        html_report = generate_html_report(analyzer)
        st.download_button(
            label="📄 HTML",
            data=html_report,
            file_name=f'DPDC_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html',
            mime='text/html',
            key='download-html'
        )
    
    st.sidebar.info("💡 **Tip:** Download the HTML report and open it in a browser, then use 'Print to PDF' for a PDF version.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Data Loaded:** {len(df):,} Feeders")
    st.sidebar.markdown(f"**Months:** {len(date_cols)}")
    st.sidebar.markdown(f"**Period:** {date_cols[0]} to {date_cols[-1]}")
    
    return module_key

def render_basic_stats(analyzer):
    st.header("📈 Basic Statistics & Consumption Distribution")
    
    figs, data_summary = plot_basic_statistics(analyzer)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(figs['hist'], use_container_width=True, config={'displayModeBar': False})
        st.plotly_chart(figs['cv'], use_container_width=True, config={'displayModeBar': False})
    with col2:
        st.plotly_chart(figs['top'], use_container_width=True, config={'displayModeBar': False})
        st.plotly_chart(figs['avg'], use_container_width=True, config={'displayModeBar': False})
    
    # Additional interactive hierarchical visuals
    if figs.get('treemap') is not None:
        st.markdown("---")
        st.markdown("#### Consumption Treemap (explore by drilling into levels)")
        st.plotly_chart(figs['treemap'], use_container_width=True, config={'displayModeBar': False})
    if figs.get('sunburst') is not None:
        st.markdown("---")
        st.markdown("#### Sunburst (Substation → Feeder)")
        st.plotly_chart(figs['sunburst'], use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("#### Data Summary (Top 20 by Total Consumption)")
    st.dataframe(data_summary.nlargest(20, 'total_consumption').style.format({
        'total_consumption': '{:,.0f}',
        'avg_monthly': '{:,.0f}',
        'cv': '{:.2f}'
    }), width='stretch')
    
    st.download_button(
        label="📥 Download Basic Statistics Data (CSV)",
        data=data_summary.to_csv(index=False).encode('utf-8'),
        file_name='basic_statistics.csv',
        mime='text/csv',
        key='download-basic-stats'
    )

def render_trend_analysis(analyzer):
    st.header("📊 Trend & Growth Analysis")
    
    figs, trend_data = plot_trend_analysis(analyzer)
    if figs is not None:
        col_hist, col_top = st.columns([1, 1])
        with col_hist:
            st.plotly_chart(figs['growth_hist'], use_container_width=True, config={'displayModeBar': False})
            st.plotly_chart(figs['slope'], use_container_width=True, config={'displayModeBar': False})
        with col_top:
            st.plotly_chart(figs['top_growth'], use_container_width=True, config={'displayModeBar': False})
        
        st.plotly_chart(figs['scatter'], use_container_width=True, config={'displayModeBar': False})
        
        # Additional heat/summary chart
        # Optional heatmap of growth bins (safe render)
        heat_fig = figs.get('heat')
        if heat_fig is not None and hasattr(heat_fig, "to_plotly_json"):
            st.markdown("---")
            st.markdown("#### Growth Rate Bins — Average Absolute Change")
            st.plotly_chart(heat_fig, use_container_width=True, config={'displayModeBar': False})

        
        col_grow, col_decline = st.columns(2)
        
        with col_grow:
            st.markdown("#### Top 20 Fastest Growing Feeders")
            top_growing = trend_data[trend_data['growth_rate_%'] > 0].nlargest(20, 'growth_rate_%')
            st.dataframe(top_growing.style.format({
                'growth_rate_%': '{:,.2f}%',
                'monthly_slope': '{:,.0f}',
                'absolute_change': '{:,.0f}'
            }), width='stretch')
        
        with col_decline:
            st.markdown("#### Top 20 Fastest Declining Feeders")
            top_declining = trend_data[trend_data['growth_rate_%'] < 0].nsmallest(20, 'growth_rate_%')
            st.dataframe(top_declining.style.format({
                'growth_rate_%': '{:,.2f}%',
                'monthly_slope': '{:,.0f}',
                'absolute_change': '{:,.0f}'
            }), width='stretch')

def render_seasonality_analysis(analyzer):
    st.header("🌡️ Seasonality & Time Series Analysis")
    
    figs, monthly_stats = plot_seasonality_analysis(analyzer)
    
    st.plotly_chart(figs['ts'], use_container_width=True, config={'displayModeBar': False})

    col_avg, col_yoy = st.columns([1, 1])
    with col_avg:
        st.plotly_chart(figs['avg_month'], use_container_width=True, config={'displayModeBar': False})
    with col_yoy:
        st.plotly_chart(figs['yoy'], use_container_width=True, config={'displayModeBar': False})
    
    # Month treemap (if present)
    if figs.get('month_treemap') is not None:
        st.markdown("---")
        st.markdown("#### Monthly Contribution Treemap (drill down by year → month)")
        st.plotly_chart(figs['month_treemap'], use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("#### Monthly Average Consumption Summary")
    st.dataframe(monthly_stats.style.format({
        'Avg_Total_Consumption': '{:,.0f}',
        'Std_Dev_Total_Consumption': '{:,.0f}'
    }), width='stretch')

def render_region_analysis(analyzer):
    st.header("🏢 Substation & NOCS Performance")
    
    sub_stats, nocs_stats = analyzer.get_nocs_substation_summary()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top 15 Substations by Total Consumption")
        st.dataframe(sub_stats.head(15).style.format({
            'Total Consumption': '{:,.0f}',
            'Avg Consump/Feeder': '{:,.0f}',
            'Avg CV': '{:.2f}',
            'Avg Growth Rate (%)': '{:,.2f}'
        }), width='stretch')
        
    with col2:
        st.markdown("#### Top 15 NOCS by Total Consumption")
        st.dataframe(nocs_stats.head(15).style.format({
            'Total Consumption': '{:,.0f}',
            'Avg Consump/Feeder': '{:,.0f}',
            'Avg CV': '{:.2f}',
            'Avg Growth Rate (%)': '{:,.2f}'
        }), width='stretch')

def render_anomaly_analysis(analyzer):
    st.header("⚠️ Anomaly & Load Factor Analysis")
    
    high_cv, inactive, low_lf = analyzer.get_anomaly_tables()
    
    st.markdown("#### High Variability Feeders (CV ≥ 1.0)")
    st.info(f"Total High Variability Feeders: **{len(high_cv)}**")
    st.dataframe(high_cv.head(15).style.format({
        'cv': '{:.2f}',
        'Avg Monthly Consumption': '{:,.0f}'
    }), width='stretch')

    st.markdown("#### Inactive Feeders (Zero consumption in last 3 months)")
    st.warning(f"Total Inactive Feeders: **{len(inactive)}**")
    st.dataframe(inactive.head(15), width='stretch')
    
    st.markdown("#### Low Load Factor Feeders (Load Factor < 50%)")
    st.error(f"Total Low Load Factor Feeders: **{len(low_lf)}**")
    st.dataframe(low_lf.head(15).style.format({
        'load_factor': '{:.2f}%',
        'avg_load': '{:,.0f}',
        'peak_load': '{:,.0f}'
    }), width='stretch')

def render_clustering_analysis(analyzer):
    st.header("🎯 Clustering of Feeder Patterns")
    
    n_clusters = st.slider("Select Number of Clusters (K)", min_value=3, max_value=10, value=5, step=1)
    
    figs, cluster_summary = plot_clustering_analysis(analyzer, n_clusters)
    if figs is not None:
        col_pca, col_pattern = st.columns([1, 1])
        with col_pca:
            st.plotly_chart(figs['pca'], use_container_width=True, config={'displayModeBar': False})
            # Show cluster composition sunburst under PCA for quick drilldown
            if figs.get('sunburst_cluster') is not None:
                st.markdown("#### Cluster Composition (drill into cluster → substation → feeder)")
                st.plotly_chart(figs['sunburst_cluster'], use_container_width=True, config={'displayModeBar': False})
        with col_pattern:
            st.plotly_chart(figs['pattern'], use_container_width=True, config={'displayModeBar': False})
        
        st.markdown(f"#### Cluster Summary (K={n_clusters})")
        st.dataframe(cluster_summary.style.format({
            'Total_Consumption': '{:,.0f}'
        }), width='stretch')

def render_forecasting(analyzer):
    st.header("🔮 Individual Feeder Forecasting")
    
    feeder_names = sorted(analyzer.df['feeder_name'].unique())
    
    col_select, col_settings = st.columns([2, 1])
    with col_select:
        selected_feeder = st.selectbox("Select Feeder for Forecast:", feeder_names)
    with col_settings:
        selected_method = st.selectbox("Select Forecasting Method:", 
                                      ['Exponential Smoothing', 'Linear Regression', 'Polynomial (2nd Order)'], 
                                      index=0)
    
    if selected_feeder:
        fig_forecast, forecast_data = forecast_feeder(analyzer, selected_feeder, selected_method, forecast_months=6)
        
        if fig_forecast is None:
            st.error(forecast_data)
        else:
            st.plotly_chart(fig_forecast, use_container_width=True, config={'displayModeBar': False})
            
            st.markdown("#### 6-Month Forecast Values")
            st.dataframe(forecast_data.style.format({'Forecast Consumption': '{:,.0f}'}), width='stretch')

def render_bulk_forecasting(analyzer):
    st.header("⚙️ Bulk Feeder Forecasting")
    st.info("Run a single model across all feeders to quickly generate aggregated and feeder-wise consumption predictions for capacity planning.")

    col_method, col_months = st.columns(2)
    with col_method:
        bulk_method = st.selectbox("Select Bulk Forecasting Method:", 
                                 ['Exponential Smoothing', 'Linear Regression', 'Polynomial (2nd Order)'], 
                                 index=0, key='bulk_method')
    with col_months:
        bulk_months = st.number_input("Forecast Months Ahead:", min_value=1, max_value=24, value=6, step=1, key='bulk_months')

    if st.button(f"Run Bulk Forecast for {bulk_months} Months", type="primary"):
        
        with st.spinner(f"Running {bulk_method} forecast for {len(analyzer.df)} feeders..."):
            bulk_report_data, count = bulk_forecasting(analyzer, bulk_method, bulk_months)
        
        if bulk_report_data is None:
            st.error(f"Bulk forecast failed: {count}")
        else:
            st.success(f"Bulk forecast completed for {count} feeders. Download the detailed report below.")
            st.download_button(
                label="📥 Download Bulk Forecast Report (Excel)",
                data=bulk_report_data,
                file_name=f'Bulk_Forecast_{bulk_method.replace(" ", "_")}_{bulk_months}M_{datetime.now().strftime("%Y%m%d")}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key='download-bulk-forecast'
            )

def render_comparison(analyzer):
    st.header("🔍 Feeder Comparison Tool")
    
    st.info("Compare consumption patterns of multiple feeders side-by-side")
    
    feeder_names = sorted(analyzer.df['feeder_name'].unique())
    selected_feeders = st.multiselect(
        "Select Feeders to Compare (2-5 recommended):",
        feeder_names,
        max_selections=10
    )
    
    if len(selected_feeders) >= 2:
        fig_comparison = compare_feeders(analyzer, selected_feeders)
        st.plotly_chart(fig_comparison, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("#### Comparison Statistics")
        comparison_stats = analyzer.stats_df[analyzer.stats_df['feeder_name'].isin(selected_feeders)][
            ['feeder_name', 'substation', 'total_consumption', 'avg_monthly', 'cv']
        ].sort_values('total_consumption', ascending=False)
        
        st.dataframe(comparison_stats.style.format({
            'total_consumption': '{:,.0f}',
            'avg_monthly': '{:,.0f}',
            'cv': '{:.2f}'
        }), width='stretch')
    else:
        st.warning("Please select at least 2 feeders to compare")

def render_correlation(analyzer):
    st.header("🔗 Correlation Analysis")
    
    st.info("Analyze consumption correlation between feeders in the same substation")
    
    corr_df = analyzer.get_correlation_analysis()
    
    st.markdown("#### Substations Ranked by Average Feeder Correlation")
    st.dataframe(corr_df.style.format({
        'Avg_Correlation': '{:.3f}'
    }), width='stretch') 
    
    st.markdown("""
    **Interpretation:**
    - **High Correlation (>0.7):** Feeders show similar consumption patterns, suggesting coordinated demand or shared characteristics
    - **Moderate Correlation (0.3-0.7):** Some relationship exists but feeders have distinct patterns
    - **Low Correlation (<0.3):** Feeders operate independently with different consumption behaviors
    """)
