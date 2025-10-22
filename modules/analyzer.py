import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class StreamlitFeederAnalysis:
    """Enhanced analysis suite with advanced features."""

    def __init__(self, dataframe, date_columns):
        self.df = dataframe.copy()
        self.date_columns = date_columns

        self.stats_df = self._calculate_basic_statistics()
        self.trend_df = self._calculate_trend_analysis()
        self.load_factor_df = self._calculate_load_factor_analysis()

    def _calculate_basic_statistics(self):
        """Helper to calculate basic stats."""
        stats_df = pd.DataFrame()
        stats_df['feeder_name'] = self.df['feeder_name']
        stats_df['substation'] = self.df['substation_name']
        stats_df['nocs'] = self.df['nocs']

        consumption_data = self.df[self.date_columns].values
        
        stats_df['total_consumption'] = np.nansum(consumption_data, axis=1)
        stats_df['avg_monthly'] = np.nanmean(consumption_data, axis=1)
        stats_df['max_consumption'] = np.nanmax(consumption_data, axis=1)
        stats_df['min_consumption'] = np.nanmin(consumption_data, axis=1)
        stats_df['std_dev'] = np.nanstd(consumption_data, axis=1)
        stats_df['cv'] = np.where(stats_df['avg_monthly'] > 0,
                                 stats_df['std_dev'] / stats_df['avg_monthly'], 0)
        
        # Additional metrics
        stats_df['median_consumption'] = np.nanmedian(consumption_data, axis=1)
        stats_df['range'] = stats_df['max_consumption'] - stats_df['min_consumption']
        
        return stats_df
    
    def _calculate_trend_analysis(self):
        """Helper to calculate growth and trend slopes."""
        trend_df = pd.DataFrame({
            'feeder_name': self.df['feeder_name'],
            'substation': self.df['substation_name'],
            'nocs': self.df['nocs']
        })

        if len(self.date_columns) >= 12:
            recent_cols = self.date_columns[-6:]
            old_cols = self.date_columns[:6]

            recent_avg = self.df[recent_cols].mean(axis=1)
            old_avg = self.df[old_cols].mean(axis=1)

            growth_rate = np.where(old_avg > 0, ((recent_avg - old_avg) / old_avg * 100), 0)
            growth_rate = np.nan_to_num(growth_rate, nan=0, posinf=0, neginf=0)

            trend_df['old_avg'] = old_avg
            trend_df['recent_avg'] = recent_avg
            trend_df['growth_rate_%'] = growth_rate
            trend_df['absolute_change'] = recent_avg - old_avg

            slopes = []
            consumption_data = self.df[self.date_columns].values
            for row in consumption_data:
                values = row
                x = np.arange(len(values))
                # Ensure values are float before checking for positive/nan
                values = values.astype(np.float64) 
                mask = (values > 0) & (~np.isnan(values))
                slope = stats.linregress(x[mask], values[mask])[0] if mask.sum() >= 2 else 0
                slopes.append(slope)
            trend_df['monthly_slope'] = slopes
        else:
            trend_df['growth_rate_%'] = 0.0
            trend_df['monthly_slope'] = 0.0

        return trend_df

    def _calculate_load_factor_analysis(self):
        """Helper to calculate load factor."""
        load_factor_df = self.stats_df[['feeder_name', 'substation', 'nocs', 'avg_monthly', 'max_consumption']].rename(
            columns={'avg_monthly': 'avg_load', 'max_consumption': 'peak_load'}
        )
        load_factor_df['load_factor'] = np.where(
            load_factor_df['peak_load'] > 0,
            (load_factor_df['avg_load'] / load_factor_df['peak_load'] * 100),
            0
        )
        return load_factor_df[load_factor_df['peak_load'] > 0]

    def get_nocs_substation_summary(self):
        """Generates tables for NOCS and Substation Summary."""
        
        substation_stats = self.stats_df.groupby('substation').agg({
            'feeder_name': 'count',
            'total_consumption': 'sum',
            'avg_monthly': 'mean',
            'cv': 'mean'
        }).round(2)

        substation_stats.columns = ['Feeder Count', 'Total Consumption', 'Avg Consump/Feeder', 'Avg CV']
        substation_stats = substation_stats.sort_values('Total Consumption', ascending=False)
        substation_stats.index.name = 'Substation'

        nocs_stats = self.stats_df.groupby('nocs').agg({
            'feeder_name': 'count',
            'total_consumption': 'sum',
            'avg_monthly': 'mean',
            'cv': 'mean'
        }).round(2)

        nocs_stats.columns = ['Feeder Count', 'Total Consumption', 'Avg Consump/Feeder', 'Avg CV']
        nocs_stats = nocs_stats.sort_values('Total Consumption', ascending=False)
        nocs_stats.index.name = 'NOCS'

        if 'growth_rate_%' in self.trend_df.columns:
            nocs_growth = self.trend_df.groupby('nocs')['growth_rate_%'].mean().round(2).rename('Avg Growth Rate (%)')
            nocs_stats = nocs_stats.join(nocs_growth).fillna(0)
            substation_growth = self.trend_df.groupby('substation')['growth_rate_%'].mean().round(2).rename('Avg Growth Rate (%)')
            substation_stats = substation_stats.join(substation_growth).fillna(0)

        return substation_stats, nocs_stats

    def get_anomaly_tables(self):
        """Generates tables for high variability and inactive feeders."""

        high_cv = self.stats_df[self.stats_df['cv'] >= 1.0].nlargest(20, 'cv')[
            ['feeder_name', 'substation', 'nocs', 'cv', 'avg_monthly']
        ].sort_values('cv', ascending=False).rename(columns={'avg_monthly': 'Avg Monthly Consumption'})

        if len(self.date_columns) >= 3:
            recent_3_months = self.date_columns[-3:]
            inactive_mask = (self.df[recent_3_months] == 0).all(axis=1)
            inactive_feeders = self.df[inactive_mask][['feeder_name', 'substation_name', 'nocs']].rename(
                columns={'substation_name': 'substation'}
            )
        else:
            inactive_feeders = pd.DataFrame(columns=['feeder_name', 'substation', 'nocs'])

        low_lf = self.load_factor_df[self.load_factor_df['load_factor'] < 50].nsmallest(20, 'load_factor')[
            ['feeder_name', 'substation', 'load_factor', 'avg_load', 'peak_load']
        ].sort_values('load_factor', ascending=True)

        return high_cv, inactive_feeders, low_lf

    def get_correlation_analysis(self):
        """Analyze correlation between feeders in the same substation."""
        correlation_results = []
        
        for substation in self.df['substation_name'].unique():
            sub_feeders = self.df[self.df['substation_name'] == substation]
            
            if len(sub_feeders) >= 2:
                consumption_matrix = sub_feeders[self.date_columns].T
                corr_matrix = consumption_matrix.corr()
                
                # Check for NaNs/non-numeric columns before taking mean
                valid_corrs = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
                valid_corrs = valid_corrs[~np.isnan(valid_corrs)]
                
                avg_corr = valid_corrs.mean() if len(valid_corrs) > 0 else 0
                
                correlation_results.append({
                    'Substation': substation,
                    'Feeder_Count': len(sub_feeders),
                    'Avg_Correlation': avg_corr
                })
        
        corr_df = pd.DataFrame(correlation_results).sort_values('Avg_Correlation', ascending=False)
        return corr_df
