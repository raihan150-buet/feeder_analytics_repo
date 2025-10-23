# modules/visualization.py

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def _standard_layout(fig, xaxis_tickangle=-45, yaxis_tickformat='~s'):
    """Apply consistent interactive styling to all figures."""
    fig.update_layout(
        template="plotly_white",
        hovermode='closest',
        margin=dict(l=40, r=40, t=60, b=40),
        autosize=True,
        legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    if hasattr(fig.layout, 'xaxis'):
        fig.update_xaxes(tickangle=xaxis_tickangle, rangeslider_visible=True, automargin=True)
    if hasattr(fig.layout, 'yaxis'):
        fig.update_yaxes(tickformat=yaxis_tickformat, automargin=True)
    return fig


# ==========================================================
# 📊 BASIC STATISTICS VISUALIZATIONS
# ==========================================================

def plot_basic_statistics(self):
    """Generates enhanced interactive figures for basic statistics."""
    stats_df = self.stats_df.copy()

    # Histogram of total consumption
    fig_hist = px.histogram(
        stats_df, x='total_consumption', nbins=50,
        title='Distribution of Total Consumption',
        labels={'total_consumption': 'Total Consumption (units)'}
    )
    fig_hist = _standard_layout(fig_hist)

    # Top consumers
    top_consumers = stats_df.nlargest(15, 'total_consumption').sort_values('total_consumption', ascending=True)
    fig_top = px.bar(
        top_consumers, x='total_consumption', y='feeder_name', orientation='h',
        title='Top 15 Consumers by Total Consumption',
        labels={'total_consumption': 'Total Consumption', 'feeder_name': 'Feeder Name'}
    )
    fig_top = _standard_layout(fig_top)

    # CV distribution
    fig_cv = px.histogram(
        stats_df, x=stats_df['cv'].clip(0, 5), nbins=50,
        title='Distribution of Consumption Variability (CV)',
        labels={'x': 'Coefficient of Variation (Clipped 0–5)', 'count': 'Number of Feeders'}
    )
    fig_cv = _standard_layout(fig_cv, xaxis_tickangle=0)

    # Average monthly consumption
    fig_avg = px.histogram(
        stats_df, x='avg_monthly', nbins=50,
        title='Distribution of Average Monthly Consumption',
        labels={'avg_monthly': 'Average Monthly Consumption (units)'}
    )
    fig_avg = _standard_layout(fig_avg)

    # Treemap: Substation → NOCS → Feeder
    treemap_df = stats_df.copy()
    treemap_df['substation'] = treemap_df['substation'].fillna('Unknown')
    treemap_df['nocs'] = treemap_df['nocs'].fillna('Unknown')
    fig_treemap = px.treemap(
        treemap_df,
        path=['substation', 'nocs', 'feeder_name'],
        values='total_consumption',
        title='Consumption Treemap: Substation → NOCS → Feeder',
        hover_data={'total_consumption': ':,.0f'}
    )
    fig_treemap = _standard_layout(fig_treemap, xaxis_tickangle=0)

    # Sunburst: Substation → Feeder
    sunburst_df = stats_df.copy()
    sunburst_df['substation'] = sunburst_df['substation'].fillna('Unknown')
    fig_sunburst = px.sunburst(
        sunburst_df,
        path=['substation', 'feeder_name'],
        values='total_consumption',
        title='Sunburst: Substation → Feeder Consumption'
    )
    fig_sunburst = _standard_layout(fig_sunburst, xaxis_tickangle=0)

    figs = {
        'hist': fig_hist,
        'top': fig_top,
        'cv': fig_cv,
        'avg': fig_avg,
        'treemap': fig_treemap,
        'sunburst': fig_sunburst
    }

    summary_table = stats_df[
        ['feeder_name', 'substation', 'nocs', 'total_consumption', 'avg_monthly', 'cv']
    ].copy()

    return figs, summary_table


# ==========================================================
# 📈 TREND ANALYSIS VISUALIZATIONS
# ==========================================================

def plot_trend_analysis(self):
    """Interactive trend analysis with additional visual summaries."""
    trend_df = self.trend_df.copy()
    if len(trend_df) == 0 or 'growth_rate_%' not in trend_df.columns:
        st.warning("Not enough data points (less than 12 months) to perform Trend Analysis.")
        return None, pd.DataFrame()

    # Distribution of growth rates
    fig_growth_hist = px.histogram(
        trend_df, x=trend_df['growth_rate_%'].clip(-200, 200), nbins=50,
        title='Distribution of Growth Rates (Recent 6M vs Initial 6M)',
        labels={'x': 'Growth Rate (%) (Clipped -200% to 200%)'}
    )
    fig_growth_hist = _standard_layout(fig_growth_hist, xaxis_tickangle=0)
    fig_growth_hist.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")

    # Top growing feeders
    top_growth = trend_df[trend_df['growth_rate_%'] > 0].nlargest(15, 'growth_rate_%').sort_values('growth_rate_%', ascending=True)
    fig_top_growth = px.bar(
        top_growth, x='growth_rate_%', y='feeder_name', orientation='h',
        title='Top 15 Fastest Growing Feeders',
        labels={'growth_rate_%': 'Growth Rate (%)', 'feeder_name': 'Feeder Name'}
    )
    fig_top_growth = _standard_layout(fig_top_growth)

    # Scatter (Old vs Recent)
    fig_scatter = px.scatter(
        trend_df, x='old_avg', y='recent_avg',
        hover_data=['feeder_name', 'growth_rate_%', 'substation', 'nocs'],
        title='Consumption Change: Old vs Recent',
        labels={'old_avg': 'Avg (First 6M)', 'recent_avg': 'Avg (Recent 6M)'}
    )
    max_val = max(trend_df['old_avg'].max(), trend_df['recent_avg'].max()) * 1.05
    fig_scatter.add_shape(
        type='line', x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color='Red', dash='dash')
    )
    fig_scatter = _standard_layout(fig_scatter)

    # Slope distribution
    fig_slope = px.histogram(
        trend_df, x='monthly_slope', nbins=50,
        title='Distribution of Monthly Consumption Trends (Linear Slope)',
        labels={'monthly_slope': 'Monthly Trend Slope (units/month)'}
    )
    fig_slope = _standard_layout(fig_slope)

    # New: heat summary — average absolute change by growth bin
    heat_df = trend_df.copy()
    heat_df['growth_bin'] = pd.cut(heat_df['growth_rate_%'].clip(-500, 500), bins=10)
    pivot = heat_df.pivot_table(index='growth_bin', values='absolute_change', aggfunc='mean').reset_index().dropna()

    fig_heat = px.bar(
        pivot, x='growth_bin', y='absolute_change',
        title='Average Absolute Change by Growth Rate Bin',
        labels={'absolute_change': 'Mean Δ Consumption'}
    ) if not pivot.empty else None
    if fig_heat:
        fig_heat = _standard_layout(fig_heat, xaxis_tickangle=45)

    figs = {
        'growth_hist': fig_growth_hist,
        'top_growth': fig_top_growth,
        'scatter': fig_scatter,
        'slope': fig_slope,
        'heat': fig_heat
    }

    return figs, trend_df[['feeder_name', 'growth_rate_%', 'monthly_slope', 'absolute_change']]


# ==========================================================
# 🌡️ SEASONALITY ANALYSIS VISUALIZATIONS
# ==========================================================

def plot_seasonality_analysis(self):
    """Interactive seasonality and time-series visualizations."""
    consumption_data = self.df[self.date_columns]
    monthly_total = consumption_data.sum(axis=0)

    monthly_df = pd.DataFrame({
        'date': self.date_columns,
        'total': monthly_total.values
    })
    monthly_df['dt'] = pd.to_datetime(monthly_df['date'])
    monthly_df['year'] = monthly_df['dt'].dt.year
    monthly_df['month'] = monthly_df['dt'].dt.month
    monthly_df['month_name'] = monthly_df['dt'].dt.strftime('%b')

    # Overall time-series
    fig_ts = px.line(
        monthly_df, x='date', y='total', markers=True,
        title='Total Consumption Over Time (All Feeders)',
        labels={'total': 'Total Consumption (units)', 'date': 'Month'}
    )
    fig_ts = _standard_layout(fig_ts)

    # Monthly mean/std summary
    month_stats = monthly_df.groupby('month').agg(
        mean_total=('total', 'mean'),
        std_total=('total', 'std')
    ).reset_index()
    month_stats['month_name'] = pd.to_datetime(month_stats['month'], format='%m').dt.strftime('%b')
    month_stats = month_stats.sort_values('month')

    fig_avg_month = go.Figure()
    fig_avg_month.add_trace(go.Bar(
        x=month_stats['month_name'],
        y=month_stats['mean_total'],
        name='Mean Total Consumption',
        error_y=dict(type='data', array=month_stats['std_total'], visible=True, thickness=1.5, width=3)
    ))
    fig_avg_month.update_layout(title='Average Monthly Total Consumption (± Std Dev)')
    fig_avg_month = _standard_layout(fig_avg_month, xaxis_tickangle=0)

    # Year-over-year comparison
    fig_yoy = px.line(
        monthly_df, x='month_name', y='total', color='year', markers=True,
        title='Year-over-Year Consumption Comparison',
        labels={'total': 'Total Consumption (units)', 'month_name': 'Month', 'year': 'Year'},
        category_orders={"month_name": [pd.to_datetime(str(i), format='%m').strftime('%b') for i in range(1, 13)]}
    )
    fig_yoy = _standard_layout(fig_yoy)

    # Treemap: Year → Month
    treemap_month = px.treemap(
        monthly_df, path=['year', 'month_name'], values='total',
        title='Monthly Contribution Treemap (Year → Month)'
    )
    treemap_month = _standard_layout(treemap_month, xaxis_tickangle=0)

    figs = {'ts': fig_ts, 'avg_month': fig_avg_month, 'yoy': fig_yoy, 'month_treemap': treemap_month}

    return figs, month_stats.rename(columns={
        'mean_total': 'Avg_Total_Consumption',
        'std_total': 'Std_Dev_Total_Consumption'
    })


# ==========================================================
# 🎯 CLUSTERING ANALYSIS VISUALIZATIONS
# ==========================================================

def plot_clustering_analysis(self, n_clusters=5):
    """Performs K-Means clustering and returns interactive plots."""
    consumption_data = self.df[self.date_columns].fillna(0).values

    if consumption_data.shape[0] < n_clusters or consumption_data.shape[1] < 1:
        st.error(f"Cannot perform clustering: Need at least {n_clusters} feeders and monthly data.")
        return None, pd.DataFrame()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(consumption_data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    cluster_df = self.stats_df[['feeder_name', 'substation', 'nocs', 'total_consumption']].copy()
    cluster_df['cluster'] = clusters.astype(str)

    # PCA projection
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_.sum()

    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['cluster'] = cluster_df['cluster']
    pca_df['feeder_name'] = cluster_df['feeder_name']
    pca_df['substation'] = cluster_df['substation']
    pca_df['nocs'] = cluster_df['nocs']
    pca_df['total_consumption'] = cluster_df['total_consumption']

    fig_pca = px.scatter(
        pca_df, x='PC1', y='PC2', color='cluster',
        hover_data=['feeder_name', 'total_consumption', 'substation', 'nocs'],
        title=f'Feeder Clusters (PCA Projection, Var: {explained_variance * 100:.1f}%)'
    )
    fig_pca = _standard_layout(fig_pca)

    # Cluster patterns over time
    time_series_data = []
    for i in range(n_clusters):
        cluster_mean = consumption_data[clusters == i].mean(axis=0)
        cluster_size = len(cluster_df[cluster_df['cluster'] == str(i)])
        df_temp = pd.DataFrame({
            'Month': self.date_columns,
            'Consumption': cluster_mean,
            'Cluster': f'Cluster {i} (N={cluster_size})'
        })
        time_series_data.append(df_temp)
    time_series_df = pd.concat(time_series_data)

    fig_pattern = px.line(
        time_series_df, x='Month', y='Consumption', color='Cluster', markers=False,
        title='Average Consumption Pattern by Cluster',
        labels={'Consumption': 'Average Consumption (units)', 'Month': 'Month'}
    )
    fig_pattern = _standard_layout(fig_pattern)

    # Sunburst: Cluster → Substation → Feeder
    sunburst_df = cluster_df.copy()
    sunburst_df['substation'] = sunburst_df['substation'].fillna('Unknown')
    fig_sunburst_cluster = px.sunburst(
        sunburst_df, path=['cluster', 'substation', 'feeder_name'],
        values='total_consumption',
        title='Cluster Composition (Cluster → Substation → Feeder)'
    )
    fig_sunburst_cluster = _standard_layout(fig_sunburst_cluster, xaxis_tickangle=0)

    cluster_summary = cluster_df.groupby('cluster').agg(
        Feeder_Count=('feeder_name', 'count'),
        Total_Consumption=('total_consumption', 'sum'),
        Mode_Substation=('substation', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'),
        Mode_NOCS=('nocs', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A')
    ).sort_values('Total_Consumption', ascending=False)
    cluster_summary.index.name = 'Cluster ID'

    figs = {'pca': fig_pca, 'pattern': fig_pattern, 'sunburst_cluster': fig_sunburst_cluster}

    return figs, cluster_summary


# ==========================================================
# 🔍 FEEDER COMPARISON
# ==========================================================

def compare_feeders(self, feeder_list):
    """Compare multiple feeders side by side."""
    comparison_data = []
    for feeder in feeder_list:
        feeder_row = self.df[self.df['feeder_name'] == feeder].iloc[0]
        values = feeder_row[self.date_columns].values
        for month, val in zip(self.date_columns, values):
            comparison_data.append({'Month': month, 'Consumption': val, 'Feeder': feeder})

    comp_df = pd.DataFrame(comparison_data)
    fig = px.line(
        comp_df, x='Month', y='Consumption', color='Feeder', markers=True,
        title='Feeder Comparison',
        labels={'Consumption': 'Consumption (units)', 'Month': 'Month'}
    )
    fig = _standard_layout(fig)
    return fig
