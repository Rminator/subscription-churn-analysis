import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


class ChurnAnalyzer:
    """
    A class for analyzing customer churn in subscription businesses.
    
    This class provides methods to calculate various churn metrics
    including customer churn rate, revenue churn rate, and
    cohort-based retention analysis.
    """
    
    def __init__(self, customer_data=None, subscription_data=None):
        """
        Initialize the ChurnAnalyzer with customer and subscription data.
        
        Parameters:
        -----------
        customer_data : pandas.DataFrame, optional
            DataFrame containing customer information with at least 'customer_id' column.
        subscription_data : pandas.DataFrame, optional
            DataFrame containing subscription information with columns:
            'customer_id', 'start_date', 'end_date', 'monthly_revenue'
        """
        self.customer_data = customer_data
        self.subscription_data = subscription_data
    
    def load_data(self, customer_file=None, subscription_file=None):
        """
        Load data from CSV files.
        
        Parameters:
        -----------
        customer_file : str, optional
            Path to CSV file containing customer data.
        subscription_file : str, optional
            Path to CSV file containing subscription data.
        """
        if customer_file:
            self.customer_data = pd.read_csv(customer_file)
        if subscription_file:
            self.subscription_data = pd.read_csv(subscription_file)
            
        # Convert date columns to datetime
        if self.subscription_data is not None:
            for col in ['start_date', 'end_date']:
                if col in self.subscription_data.columns:
                    self.subscription_data[col] = pd.to_datetime(self.subscription_data[col])
    
    def calculate_customer_churn_rate(self, period_start, period_end, interval='monthly'):
        """
        Calculate customer churn rate for a specified period.
        
        Parameters:
        -----------
        period_start : str or datetime
            Start date of the analysis period (inclusive).
        period_end : str or datetime
            End date of the analysis period (exclusive).
        interval : str, default 'monthly'
            Time interval for analysis ('monthly', 'quarterly', 'yearly').
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with columns for period, active customers at start,
            churned customers, and churn rate.
        """
        if isinstance(period_start, str):
            period_start = pd.to_datetime(period_start)
        if isinstance(period_end, str):
            period_end = pd.to_datetime(period_end)
        
        # Determine time intervals based on the specified interval
        if interval == 'monthly':
            periods = pd.date_range(start=period_start, end=period_end, freq='MS')
            period_length = pd.DateOffset(months=1)
        elif interval == 'quarterly':
            periods = pd.date_range(start=period_start, end=period_end, freq='QS')
            period_length = pd.DateOffset(months=3)
        elif interval == 'yearly':
            periods = pd.date_range(start=period_start, end=period_end, freq='YS')
            period_length = pd.DateOffset(years=1)
        else:
            raise ValueError("Interval must be 'monthly', 'quarterly', or 'yearly'")
        
        results = []
        
        for i, period_start_date in enumerate(periods[:-1]):
            period_end_date = periods[i+1]
            
            # Customers active at the start of the period
            active_at_start = self.subscription_data[
                (self.subscription_data['start_date'] < period_start_date) & 
                ((self.subscription_data['end_date'] >= period_start_date) | 
                 (self.subscription_data['end_date'].isna()))
            ]['customer_id'].nunique()
            
            # Customers who churned during the period
            churned = self.subscription_data[
                (self.subscription_data['start_date'] < period_start_date) & 
                (self.subscription_data['end_date'] >= period_start_date) & 
                (self.subscription_data['end_date'] < period_end_date)
            ]['customer_id'].nunique()
            
            # Calculate churn rate
            churn_rate = (churned / active_at_start * 100) if active_at_start > 0 else 0
            
            results.append({
                'period_start': period_start_date,
                'period_end': period_end_date,
                'active_customers': active_at_start,
                'churned_customers': churned,
                'churn_rate_percent': round(churn_rate, 2)
            })
        
        return pd.DataFrame(results)
    
    def calculate_revenue_churn_rate(self, period_start, period_end, interval='monthly'):
        """
        Calculate revenue churn rate for a specified period.
        
        Parameters:
        -----------
        period_start : str or datetime
            Start date of the analysis period (inclusive).
        period_end : str or datetime
            End date of the analysis period (exclusive).
        interval : str, default 'monthly'
            Time interval for analysis ('monthly', 'quarterly', 'yearly').
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with columns for period, MRR at start,
            churned MRR, and revenue churn rate.
        """
        if isinstance(period_start, str):
            period_start = pd.to_datetime(period_start)
        if isinstance(period_end, str):
            period_end = pd.to_datetime(period_end)
        
        # Determine time intervals based on the specified interval
        if interval == 'monthly':
            periods = pd.date_range(start=period_start, end=period_end, freq='MS')
        elif interval == 'quarterly':
            periods = pd.date_range(start=period_start, end=period_end, freq='QS')
        elif interval == 'yearly':
            periods = pd.date_range(start=period_start, end=period_end, freq='YS')
        else:
            raise ValueError("Interval must be 'monthly', 'quarterly', or 'yearly'")
        
        results = []
        
        for i, period_start_date in enumerate(periods[:-1]):
            period_end_date = periods[i+1]
            
            # MRR at the start of the period
            active_subs_at_start = self.subscription_data[
                (self.subscription_data['start_date'] < period_start_date) & 
                ((self.subscription_data['end_date'] >= period_start_date) | 
                 (self.subscription_data['end_date'].isna()))
            ]
            mrr_at_start = active_subs_at_start['monthly_revenue'].sum()
            
            # Churned MRR during the period
            churned_subs = self.subscription_data[
                (self.subscription_data['start_date'] < period_start_date) & 
                (self.subscription_data['end_date'] >= period_start_date) & 
                (self.subscription_data['end_date'] < period_end_date)
            ]
            churned_mrr = churned_subs['monthly_revenue'].sum()
            
            # Calculate revenue churn rate
            revenue_churn_rate = (churned_mrr / mrr_at_start * 100) if mrr_at_start > 0 else 0
            
            results.append({
                'period_start': period_start_date,
                'period_end': period_end_date,
                'mrr_at_start': round(mrr_at_start, 2),
                'churned_mrr': round(churned_mrr, 2),
                'revenue_churn_rate_percent': round(revenue_churn_rate, 2)
            })
        
        return pd.DataFrame(results)
    
    def create_cohort_analysis(self, periods=12):
        """
        Perform cohort-based retention analysis.
        
        Parameters:
        -----------
        periods : int, default 12
            Number of periods to analyze after the initial subscription.
        
        Returns:
        --------
        pandas.DataFrame
            Cohort retention matrix where rows represent cohorts (by start month)
            and columns represent periods since first subscription.
        """
        # Ensure we have subscription data
        if self.subscription_data is None:
            raise ValueError("Subscription data is required for cohort analysis")
        
        # Extract the cohort month (first subscription month) for each customer
        customer_cohorts = self.subscription_data.groupby('customer_id')['start_date'].min().reset_index()
        customer_cohorts['cohort_month'] = customer_cohorts['start_date'].dt.to_period('M')
        
        # Determine active periods for each subscription
        subscriptions = self.subscription_data.copy()
        
        # Handle ongoing subscriptions
        subscriptions.loc[subscriptions['end_date'].isna(), 'end_date'] = datetime.now()
        
        # Create a list of active months for each subscription
        active_periods = []
        
        for _, row in subscriptions.iterrows():
            customer_id = row['customer_id']
            start_period = row['start_date'].to_period('M')
            end_period = row['end_date'].to_period('M')
            
            # Generate all active periods for this subscription
            periods_range = pd.period_range(start=start_period, end=end_period, freq='M')
            for period in periods_range:
                active_periods.append({'customer_id': customer_id, 'active_period': period})
        
        active_periods_df = pd.DataFrame(active_periods)
        
        # Merge with cohort information
        customer_activity = pd.merge(
            active_periods_df, 
            customer_cohorts[['customer_id', 'cohort_month']], 
            on='customer_id'
        )
        
        # Calculate periods since first subscription
        customer_activity['periods_since_first'] = (
            customer_activity['active_period'].astype(int) - 
            customer_activity['cohort_month'].astype(int)
        )
        
        # Count unique customers by cohort and period
        cohort_data = customer_activity.groupby(['cohort_month', 'periods_since_first'])['customer_id'].nunique().reset_index()
        
        # Create the cohort retention matrix
        cohort_pivot = cohort_data.pivot_table(
            index='cohort_month',
            columns='periods_since_first',
            values='customer_id'
        )
        
        # Calculate retention percentages
        cohort_sizes = cohort_pivot[0]
        retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0) * 100
        
        # Limit to the requested number of periods
        retention_matrix = retention_matrix.iloc[:, :periods]
        
        return retention_matrix
    
    def plot_churn_rate(self, churn_data, metric='churn_rate_percent', title=None):
        """
        Plot churn rate trends over time.
        
        Parameters:
        -----------
        churn_data : pandas.DataFrame
            DataFrame containing churn data, as returned by calculate_customer_churn_rate
            or calculate_revenue_churn_rate.
        metric : str, default 'churn_rate_percent'
            Column name to plot ('churn_rate_percent' or 'revenue_churn_rate_percent').
        title : str, optional
            Plot title.
        
        Returns:
        --------
        matplotlib.figure.Figure
            The generated plot figure.
        """
        plt.figure(figsize=(12, 6))
        
        plt.plot(churn_data['period_start'], churn_data[metric], marker='o', linestyle='-')
        
        if title:
            plt.title(title, fontsize=14)
        else:
            metric_name = "Customer Churn Rate" if metric == 'churn_rate_percent' else "Revenue Churn Rate"
            plt.title(f"{metric_name} Over Time", fontsize=14)
        
        plt.xlabel('Period', fontsize=12)
        plt.ylabel(f"{metric} (%)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_cohort_heatmap(self, retention_matrix, title="Cohort Retention Heatmap"):
        """
        Plot cohort retention as a heatmap.
        
        Parameters:
        -----------
        retention_matrix : pandas.DataFrame
            Cohort retention matrix, as returned by create_cohort_analysis.
        title : str, default "Cohort Retention Heatmap"
            Plot title.
        
        Returns:
        --------
        matplotlib.figure.Figure
            The generated plot figure.
        """
        plt.figure(figsize=(12, 8))
        
        # Plot the heatmap
        plt.imshow(retention_matrix, cmap='Blues', aspect='auto')
        
        # Set title and labels
        plt.title(title, fontsize=14)
        plt.ylabel('Cohort Month', fontsize=12)
        plt.xlabel('Periods Since First Subscription', fontsize=12)
        
        # Set ticks
        plt.yticks(range(len(retention_matrix.index)), retention_matrix.index.astype(str), fontsize=10)
        plt.xticks(range(len(retention_matrix.columns)), retention_matrix.columns, fontsize=10)
        
        # Add a color bar
        cbar = plt.colorbar()
        cbar.set_label('Retention Rate (%)', fontsize=12)
        
        # Add percentage annotations on the heatmap
        for i in range(len(retention_matrix.index)):
            for j in range(len(retention_matrix.columns)):
                value = retention_matrix.iloc[i, j]
                if not np.isnan(value):
                    plt.text(j, i, f"{value:.1f}%", ha='center', va='center', 
                             color='white' if value > 50 else 'black', fontsize=9)
        
        plt.tight_layout()
        
        return plt.gcf()


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    # Generate customer data
    num_customers = 1000
    customer_ids = [f'CUST{i:05d}' for i in range(1, num_customers + 1)]
    
    customers = pd.DataFrame({
        'customer_id': customer_ids,
        'acquisition_date': pd.date_range(start='2022-01-01', periods=num_customers, freq='D')
    })
    
    # Generate subscription data
    subscriptions = []
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    for cust_id in customer_ids:
        # Random subscription start date
        sub_start = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days // 2))
        
        # Some subscriptions end, others are ongoing
        if np.random.random() < 0.3:  # 30% churn rate
            sub_end = sub_start + timedelta(days=np.random.randint(30, 365))
            if sub_end > end_date:
                sub_end = None
        else:
            sub_end = None
        
        # Random monthly revenue between $9.99 and $99.99
        monthly_revenue = round(np.random.uniform(9.99, 99.99), 2)
        
        subscriptions.append({
            'customer_id': cust_id,
            'start_date': sub_start,
            'end_date': sub_end,
            'monthly_revenue': monthly_revenue
        })
    
    subscriptions_df = pd.DataFrame(subscriptions)
    
    # Initialize the ChurnAnalyzer with the sample data
    analyzer = ChurnAnalyzer(customers, subscriptions_df)
    
    # Calculate customer churn rate (monthly)
    customer_churn = analyzer.calculate_customer_churn_rate(
        period_start='2022-01-01', 
        period_end='2023-12-31',
        interval='monthly'
    )
    print("\nMonthly Customer Churn Rate:")
    print(customer_churn.head())
    
    # Calculate revenue churn rate (quarterly)
    revenue_churn = analyzer.calculate_revenue_churn_rate(
        period_start='2022-01-01', 
        period_end='2023-12-31',
        interval='quarterly'
    )
    print("\nQuarterly Revenue Churn Rate:")
    print(revenue_churn.head())
    
    # Create and display cohort analysis
    retention_matrix = analyzer.create_cohort_analysis(periods=6)
    print("\nCohort Retention Matrix (%):")
    print(retention_matrix.head())
    
    # Save plots to files
    fig1 = analyzer.plot_churn_rate(customer_churn, title="Monthly Customer Churn Rate")
    fig1.savefig('customer_churn_rate.png')
    
    fig2 = analyzer.plot_churn_rate(revenue_churn, metric='revenue_churn_rate_percent', 
                                    title="Quarterly Revenue Churn Rate")
    fig2.savefig('revenue_churn_rate.png')
    
    fig3 = analyzer.plot_cohort_heatmap(retention_matrix)
    fig3.savefig('cohort_retention.png')
    
    print("\nAnalysis complete. Plots saved to files.")
