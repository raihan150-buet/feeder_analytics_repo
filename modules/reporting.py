import pandas as pd
import streamlit as st
from io import BytesIO
from datetime import datetime

def create_excel_report(analyzer):
    """Creates a comprehensive Excel report with multiple sheets."""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Format definitions (used for context but mostly decorative in this function)
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1
        })
        
        # 1. Executive Summary Sheet
        # Ensure latest stats are used in summary_data calculation
        inactive_count = (analyzer.df[analyzer.date_columns[-3:]] == 0).all(axis=1).sum() if len(analyzer.date_columns) >= 3 else 0
        low_lf_count = (analyzer.load_factor_df['load_factor'] < 50).sum()
        
        summary_data = {
            'Metric': [
                'Total Feeders Analyzed',
                'Total Consumption (Period)',
                'Average Monthly Consumption per Feeder',
                'Analysis Period Start',
                'Analysis Period End',
                'Inactive Feeders (Last 3M)',
                'High Variability Feeders (CV > 1)',
                'Low Load Factor Feeders (<50%)'
            ],
            'Value': [
                len(analyzer.df),
                analyzer.stats_df['total_consumption'].sum(),
                analyzer.stats_df['avg_monthly'].mean(),
                analyzer.date_columns[0],
                analyzer.date_columns[-1],
                inactive_count,
                (analyzer.stats_df['cv'] > 1.0).sum(),
                low_lf_count
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
        
        # 2. Basic Statistics
        analyzer.stats_df.to_excel(writer, sheet_name='Basic Statistics', index=False)
        
        # 3. Trend Analysis
        if 'growth_rate_%' in analyzer.trend_df.columns:
            analyzer.trend_df.to_excel(writer, sheet_name='Trend Analysis', index=False)
        
        # 4. Load Factor Analysis
        analyzer.load_factor_df.to_excel(writer, sheet_name='Load Factor', index=False)
        
        # 5. Substation & NOCS Summary
        sub_stats, nocs_stats = analyzer.get_nocs_substation_summary()
        sub_stats.to_excel(writer, sheet_name='Substation Summary')
        nocs_stats.to_excel(writer, sheet_name='NOCS Summary')
        
        # 6. Anomalies
        high_cv, inactive, low_lf = analyzer.get_anomaly_tables()
        high_cv.to_excel(writer, sheet_name='High Variability Feeders', index=False)
        inactive.to_excel(writer, sheet_name='Inactive Feeders', index=False)
        low_lf.to_excel(writer, sheet_name='Low Load Factor', index=False)
        
        # 7. Monthly Time Series Data
        monthly_data = analyzer.df[['feeder_name', 'substation_name', 'nocs'] + analyzer.date_columns].copy()
        monthly_data.to_excel(writer, sheet_name='Monthly Consumption Data', index=False)
        
    output.seek(0)
    return output

def generate_html_report(analyzer):
    """Generates an HTML report that can be converted to PDF."""
    
    stats_df = analyzer.stats_df
    
    total_feeders = len(analyzer.df)
    total_consumption = stats_df['total_consumption'].sum()
    avg_monthly = stats_df['avg_monthly'].mean()
    
    inactive_count = (analyzer.df[analyzer.date_columns[-3:]] == 0).all(axis=1).sum() if len(analyzer.date_columns) >= 3 else 0
    high_cv_count = (stats_df['cv'] > 1.0).sum()
    low_lf_count = (analyzer.load_factor_df['load_factor'] < 50).sum()
    
    top_10_feeders = stats_df.nlargest(10, 'total_consumption')[['feeder_name', 'substation', 'total_consumption']]
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>DPDC Feeder Consumption Analytics Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
                border-bottom: 2px solid #95a5a6;
                padding-bottom: 5px;
            }}
            .metric-box {{
                display: inline-block;
                background: #ecf0f1;
                padding: 15px;
                margin: 10px;
                border-radius: 5px;
                min-width: 200px;
            }}
            .metric-title {{
                font-size: 12px;
                color: #7f8c8d;
                text-transform: uppercase;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 20px;
            }}
            th, td {{
                border: 1px solid #bdc3c7;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #ecf0f1;
            }}
            .footer {{
                margin-top: 50px;
                text-align: center;
                color: #95a5a6;
                font-size: 12px;
            }}
            .warning {{
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 10px;
                margin: 20px 0;
            }}
            .info {{
                background-color: #d1ecf1;
                border-left: 4px solid #17a2b8;
                padding: 10px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <h1>DPDC Feeder Consumption Analytics Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Analysis Period:</strong> {analyzer.date_columns[0]} to {analyzer.date_columns[-1]}</p>
        
        <h2>Executive Summary</h2>
        
        <div class="metric-box">
            <div class="metric-title">Total Feeders</div>
            <div class="metric-value">{total_feeders:,}</div>
        </div>
        
        <div class="metric-box">
            <div class="metric-title">Total Consumption</div>
            <div class="metric-value">{total_consumption:,.0f}</div>
        </div>
        
        <div class="metric-box">
            <div class="metric-title">Avg Monthly/Feeder</div>
            <div class="metric-value">{avg_monthly:,.0f}</div>
        </div>
        
        <h2>Key Operational Indicators</h2>
        
        <div class="metric-box">
            <div class="metric-title">Inactive Feeders</div>
            <div class="metric-value">{inactive_count}</div>
        </div>
        
        <div class="metric-box">
            <div class="metric-title">High Variability (CV>1)</div>
            <div class="metric-value">{high_cv_count}</div>
        </div>
        
        <div class="metric-box">
            <div class="metric-title">Low Load Factor (<50%)</div>
            <div class="metric-value">{low_lf_count}</div>
        </div>
        
        <h2>Top 10 Feeders by Total Consumption</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Feeder Name</th>
                    <th>Substation</th>
                    <th>Total Consumption</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for idx, row in enumerate(top_10_feeders.itertuples(), 1):
        html_content += f"""
                <tr>
                    <td>{idx}</td>
                    <td>{row.feeder_name}</td>
                    <td>{row.substation}</td>
                    <td>{row.total_consumption:,.0f}</td>
                </tr>
        """
    
    html_content += """
            </tbody>
        </table>
        
        <h2>Recommendations</h2>
        <div class="info">
            <strong>Priority Actions:</strong>
            <ul>
                <li><strong>Investigate Inactive Feeders:</strong> Determine if feeders are decommissioned or require reconnection/metering checks.</li>
                <li><strong>Monitor High-Growth Feeders:</strong> Ensure capacity planning aligns with future demand projections.</li>
                <li><strong>Optimize Low Load Factor Feeders:</strong> Explore demand-side management and load balancing opportunities.</li>
                <li><strong>Address High Variability:</strong> Review consumption patterns for feeders with CV > 1.0 to identify potential metering issues or irregular usage.</li>
            </ul>
        </div>
        
        <div class="warning">
            <strong>Note:</strong> This report is generated automatically. For detailed analysis and specific feeder investigations, 
            please refer to the interactive dashboard or contact the analytics team.
        </div>
        
        <div class="footer">
            <p>DPDC Feeder Consumption Analytics Dashboard</p>
            <p>Confidential - For Internal Use Only</p>
        </div>
    </body>
    </html>
    """
    
    return html_content
