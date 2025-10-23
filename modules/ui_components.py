import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def plot_basic_statistics(self):
    """Generates figures for basic statistics using Plotly."""
    stats_df = self.stats_df
    
    fig_hist = px.histogram(stats_df, x='total_consumption', nbins=50,
                            title='Distribution of Total Consumption',
                            labels={'total_consumption': 'Total Consumption (units)'},
                            template="plotly_white")
    fig_hist.update_layout(xaxis_tickformat='~s')
    
    top_consumers = stats_df.nlargest(15, 'total_consumption').sort_values('total_consumption', ascending=True)
    fig_top = px.bar(top_consumers, x='total_consumption', y='feeder_name', orientation='h',
                     title='Top 15 Consumers by Total Consumption',
                     labels={'total_consumption': 'Total Consumption', 'feeder_name': 'Feeder Name'},
                     template="plotly_white")
    fig_top.update_layout(xaxis_tickformat='~s')

    fig_cv = px.histogram(stats_df, x=stats_df['cv'].clip(0, 5), nbins=50,
                          title='Distribution of Consumption Variability (CV)',
                          labels={'x': 'Coefficient of Variation (Clipped 0-5)', 'count': 'Number of Feeders'},
                          template="plotly_white")

    fig_avg = px.histogram(stats_df, x='avg_monthly', nbins=50,
                            title='Distribution of Average Monthly Consumption',
                            labels={'avg_monthly': 'Average Monthly Consumption (units)'},
                            template="plotly_white")
    fig_avg.update_layout(xaxis_tickformat='~s')
    
    return {'hist': fig_hist, 'top': fig_top, 'cv': fig_cv, 'avg': fig_avg}, stats_df[['feeder_name', 'substation', 'nocs', 'total_consumption', 'avg_monthly', 'cv']]

def plot_trend_analysis(self):
    """Generates figures for trend analysis using Plotly."""
    trend_df = self.trend_df
    if len(trend_df) == 0 or 'growth_rate_%' not in trend_df.columns:
        st.warning("Not enough data points (less than 12 months) to perform Trend Analysis.")
        return None, pd.DataFrame()
        
    fig_growth_hist = px.histogram(trend_df, x=trend_df['growth_rate_%'].clip(-200, 200), nbins=50,
                                  title='Distribution of Growth Rates (Recent 6M vs Initial 6M)',
                                  labels={'x': 'Growth Rate (%) (Clipped -200% to 200%)'},
                                  template="plotly_white")
    fig_growth_hist.add_vline(x=0, line_width=2, line_dash="dash", line_color="red", annotation_text="No Change")
    
    top_growth = trend_df[trend_df['growth_rate_%'] > 0].nlargest(15, 'growth_rate_%').sort_values('growth_rate_%', ascending=True)
    fig_top_growth = px.bar(top_growth, x='growth_rate_%', y='feeder_name', orientation='h',
                           title='Top 15 Fastest Growing Feeders',
                           labels={'growth_rate_%': 'Growth Rate (%)', 'feeder_name': 'Feeder Name'},
                           template="plotly_white")

    fig_scatter = px.scatter(trend_df, x='old_avg', y='recent_avg',
                             hover_data=['feeder_name', 'growth_rate_%', 'substation', 'nocs'],
                             title='Consumption Change: Old vs Recent',
                             labels={'old_avg': 'Avg Consumption (First 6 months)', 'recent_avg': 'Avg Consumption (Recent 6 months)'},
                             template="plotly_white")
    
    max_val = max(trend_df['old_avg'].max(), trend_df['recent_avg'].max()) * 1.05
    fig_scatter.add_shape(type='line', x0=0, y0=0, x1=max_val, y1=max_val,
                          line=dict(color='Red', dash='dash'),
                          name='No change line')
    fig_scatter.update_layout(xaxis_tickformat='~s', yaxis_tickformat='~s')

    fig_slope = px.histogram(trend_df, x='monthly_slope', nbins=50,
                             title='Distribution of Monthly Consumption Trends (Linear Slope)',
                             labels={'monthly_slope': 'Monthly Trend Slope (units/month)'},
                             template="plotly_white")
    fig_slope.update_layout(xaxis_tickformat='~s')
    
    return {'growth_hist': fig_growth_hist, 'top_growth': fig_top_growth, 'scatter': fig_scatter, 'slope': fig_slope}, trend_df[['feeder_name', 'growth_rate_%', 'monthly_slope', 'absolute_change']]

def plot_seasonality_analysis(self):
    """Generates figures for seasonality analysis using Plotly."""
    consumption_data = self.df[self.date_columns]
    monthly_total = consumption_data.sum(axis=0)
    
    monthly_df = pd.DataFrame({
        'date': self.date_columns,
        'total': monthly_total.values,
    })
    monthly_df['dt'] = pd.to_datetime(monthly_df['date'])
    monthly_df['year'] = monthly_df['dt'].dt.year
    monthly_df['month'] = monthly_df['dt'].dt.month
    monthly_df['month_name'] = monthly_df['dt'].dt.strftime('%b')

    month_stats = monthly_df.groupby('month').agg(
        mean_total=('total', 'mean'),
        std_total=('total', 'std'),
    ).reset_index()
    month_stats['month_name'] = pd.to_datetime(month_stats['month'], format='%m').dt.strftime('%b')
    month_stats = month_stats.sort_values('month')

    fig_ts = px.line(monthly_df, x='date', y='total', markers=True,
                     title='Total Consumption Over Time (All Feeders)',
                     labels={'total': 'Total Consumption (units)', 'date': 'Month'},
                     template="plotly_white")
    fig_ts.update_layout(yaxis_tickformat='~s', xaxis_tickangle=-45)

    fig_avg_month = go.Figure()
    fig_avg_month.add_trace(go.Bar(
        x=month_stats['month_name'], 
        y=month_stats['mean_total'], 
        name='Mean Total Consumption',
        error_y=dict(type='data', array=month_stats['std_total'], visible=True, thickness=1.5, width=3)
    ))
    fig_avg_month.update_layout(
        title='Average Monthly Total Consumption (with Std Dev)',
        xaxis_title='Month',
        yaxis_title='Mean Total Consumption (units)',
        template="plotly_white",
        yaxis_tickformat='~s'
    )

    fig_yoy = px.line(monthly_df, x='month_name', y='total', color='year', markers=True,
                      title='Year-over-Year Consumption Comparison',
                      labels={'total': 'Total Consumption (units)', 'month_name': 'Month', 'year': 'Year'},
                      template="plotly_white",
                      category_orders={"month_name": [pd.to_datetime(str(i), format='%m').strftime('%b') for i in range(1, 13)]})
    fig_yoy.update_layout(yaxis_tickformat='~s')
    
    return {'ts': fig_ts, 'avg_month': fig_avg_month, 'yoy': fig_yoy}, month_stats.rename(columns={'mean_total': 'Avg_Total_Consumption', 'std_total': 'Std_Dev_Total_Consumption'})

def plot_clustering_analysis(self, n_clusters=5):
    """Performs K-Means clustering and generates visualization using Plotly."""
    consumption_data = self.df[self.date_columns].fillna(0).values

    if consumption_data.shape[0] < n_clusters or consumption_data.shape[1] < 1:
        st.error(f"Cannot perform clustering: Need at least {n_clusters} feeders and monthly data.")
        return None, pd.DataFrame()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(consumption_data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    cluster_df = self.stats_df[['feeder_name', 'substation', 'nocs', 'total_consumption']].copy()
    cluster_df['cluster'] = clusters
    cluster_df['cluster'] = cluster_df['cluster'].astype(str)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    explained_variance = pca.explained_variance_ratio_.sum()

    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['cluster'] = cluster_df['cluster']
    pca_df['feeder_name'] = cluster_df['feeder_name']
    pca_df['substation'] = cluster_df['substation']
    pca_df['nocs'] = cluster_df['nocs']
    pca_df['total_consumption'] = cluster_df['total_consumption']
    
    fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='cluster',
                         hover_data=['feeder_name', 'total_consumption', 'substation', 'nocs'],
                         title=f'Feeder Clusters (PCA Projection, Var: {explained_variance*100:.1f}%)',
                         labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', 
                                 'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'},
                         template="plotly_white")
    
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

    fig_pattern = px.line(time_series_df, x='Month', y='Consumption', color='Cluster', markers=False,
                          title='Average Consumption Pattern by Cluster',
                          labels={'Consumption': 'Average Consumption (units)', 'Month': 'Month'},
                          template="plotly_white")
    fig_pattern.update_layout(yaxis_tickformat='~s', xaxis_tickangle=-45)

    cluster_summary = cluster_df.groupby('cluster').agg(
        Feeder_Count=('feeder_name', 'count'),
        Total_Consumption=('total_consumption', 'sum'),
        Mode_Substation=('substation', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'),
        Mode_NOCS=('nocs', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A')
    ).sort_values('Total_Consumption', ascending=False)
    cluster_summary.index.name = 'Cluster ID'
    
    return {'pca': fig_pca, 'pattern': fig_pattern}, cluster_summary

def compare_feeders(self, feeder_list):
    """Compare multiple feeders side by side."""
    comparison_data = []
    
    for feeder in feeder_list:
        feeder_row = self.df[self.df['feeder_name'] == feeder].iloc[0]
        values = feeder_row[self.date_columns].values
        
        for i, (month, val) in enumerate(zip(self.date_columns, values)):
            comparison_data.append({
                'Month': month,
                'Consumption': val,
                'Feeder': feeder
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    fig = px.line(comp_df, x='Month', y='Consumption', color='Feeder', markers=True,
                  title='Feeder Comparison',
                  labels={'Consumption': 'Consumption (units)', 'Month': 'Month'},
                  template="plotly_white")
    fig.update_layout(yaxis_tickformat='~s', xaxis_tickangle=-45)
    
    return fig
