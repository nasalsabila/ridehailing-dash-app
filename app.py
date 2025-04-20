import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder

# Layout
st.set_page_config(layout="wide") 

# Load Data
@st.cache_data
def load_data():
    df = pd.read_excel('HR_Assessment_Data_Data Scientist_Raw Data.xlsx', parse_dates=['start_time', 'end_time'])
    return df

df = load_data()

tabs = st.tabs(["Overview Insights", "User Behavior & Segments", "Operational Insights", "Cancellation Model & Simulation"])

df['hour'] = df['start_time'].dt.hour

def time_period(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 22:
        return 'Evening'
    else:
        return 'Night'

df['time_period'] = df['hour'].apply(time_period)

usage_by_time = df['time_period'].value_counts().sort_index()


# TAB 1 - Overview Insights
with tabs[0]:
    st.header("📊 Overview Insights")

    df_completed = df[df['cancelled'] == 0].reset_index(drop=True)

    # USER METRICS
    num_users = df['user_id'].nunique()
    num_cancelled_users = df[df['cancelled'] == 1]['user_id'].nunique()
    highest_spending_user = df_completed.groupby('user_id')['fare'].sum().idxmax()
    most_generous_rater = df_completed.groupby('user_id')['rating'].mean().idxmax()
    longest_distance_user = df_completed.groupby('user_id')['distance_km'].sum().idxmax()

    # DRIVER METRICS
    num_drivers = df['driver_id'].nunique()
    num_cancelled_drivers = df[df['cancelled'] == 1]['driver_id'].nunique()
    highest_earning_driver = df_completed.groupby('driver_id')['fare'].sum().idxmax()
    highest_rated_driver = df_completed.groupby('driver_id')['rating'].mean().idxmax()
    longest_distance_driver = df_completed.groupby('driver_id')['distance_km'].sum().idxmax()

    # OPERATIONAL METRICS
    total_req_trip = df['ride_id'].nunique()
    num_cancelled = df['cancelled'].sum()
    ride_volume = total_req_trip - num_cancelled
    top_pickup = df['pickup_location'].value_counts().idxmax()
    top_dropoff = df['dropoff_location'].value_counts().idxmax()
    top_pickup_dropoff = (
        df.groupby(['pickup_location', 'dropoff_location'])
        .size()
        .sort_values(ascending=False)
        .idxmax()
    )
    top_pickup_dropoff_str = f"{top_pickup_dropoff[0]} ➝ {top_pickup_dropoff[1]}"


    # PERFORMANCE METRICS (COMPLETED ONLY)
    total_fare = df_completed['fare'].sum()
    sum_distance = df_completed['distance_km'].sum()
    avg_distance = df_completed['distance_km'].mean()
    avg_rating = df_completed['rating'].mean()

    # Combine
    summary_df = pd.DataFrame({
            'category': [
                'User Metrics', 'User Metrics', 'User Metrics', 'User Metrics', 'User Metrics',
                'Driver Metrics', 'Driver Metrics', 'Driver Metrics', 'Driver Metrics', 'Driver Metrics',
                'Operational Metrics', 'Operational Metrics', 'Operational Metrics', 'Operational Metrics', 'Operational Metrics', 'Operational Metrics',
                'Performance Metrics', 'Performance Metrics', 'Performance Metrics', 'Performance Metrics'
            ],
            'metric': [
                'Number of Users',
                'Number of Users with Cancellations',
                'Highest Spending User',
                'User Giving Highest Average Rating',
                'User with Longest Total Distance',

                'Number of Drivers',
                'Number of Drivers with Cancellations',
                'Highest Earning Driver',
                'Driver with Highest Average Rating',
                'Driver with Longest Total Distance',

                'Total Requested Ride',
                'Number of Cancellations',
                'Ride Volume (Completed)',
                'Top Pickup Location',
                'Top Dropoff Location',
                'Most Frequent Pickup ➝ Dropoff',

                'Total Fare (Completed)',
                'Total Distance (Completed)',
                'Average Distance (Completed)',
                'Average Rating (Completed)'
            ],
            'value': [
                num_users,
                num_cancelled_users,
                highest_spending_user,
                most_generous_rater,
                longest_distance_user,

                num_drivers,
                num_cancelled_drivers,
                highest_earning_driver,
                highest_rated_driver,
                longest_distance_driver,

                total_req_trip,
                num_cancelled,
                ride_volume,
                top_pickup,
                top_dropoff,
                top_pickup_dropoff_str,

                total_fare,
                sum_distance,
                avg_distance,
                avg_rating
            ]
        })

    st.subheader("Key Metrics")
    # st.metric("Total Users", num_users)
    # st.metric("Total Drivers", num_drivers)
    # st.metric("Requested Rides", total_req_trip)
    # st.metric("Completed Rides", ride_volume)
    # st.metric("Total Fare", f"${total_fare:.2f}")
    # st.metric("Avg Distance (km)", f"{avg_distance:.2f}")
    # st.metric("Avg Rating", f"{avg_rating:.2f}")
    st.dataframe(summary_df)

    st.subheader("Cancellation & Low Rating Insights")
    cancel_rate = df.groupby('hour')['cancelled'].sum().reset_index()

    # cancellation
    fig, ax = plt.subplots(figsize=(10, 5))  
    sns.barplot(x='hour', y='cancelled', data=cancel_rate, palette='Reds', ax=ax)

    ax.set_title('Total Cancellation by Hour')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Total Cancellation')
    ax.set_xticks(range(0, 24))
    sns.despine()
    plt.tight_layout()

    
    st.pyplot(fig)

    # low rating

    low_ratings_by_time_period = (
        df[df['rating'] <= 2]
        .groupby('time_period')['ride_id']
        .count()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))  


    sns.barplot(
        x='time_period',
        y='ride_id',
        data=low_ratings_by_time_period,
        order=['Morning', 'Afternoon', 'Evening', 'Night'],
        palette='Reds',
        ax=ax
    )

    ax.set_title('Low Rating by Time Period')
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Total Low Rating')
    sns.despine()
    plt.tight_layout()


    st.pyplot(fig)



# TAB 2 - User Behavior & Segments
with tabs[1]:
    st.header("🧑‍💼 User Behavior & Segmentation")
    
    all_users = df.groupby('user_id').agg(
        total_req_trip=('ride_id', 'nunique')
    ).reset_index()

    user_completed = df_completed.groupby('user_id').agg(
        total_completed_trips=('ride_id', 'nunique'),
        avg_rating=('rating', 'mean'),
        total_distance=('distance_km', 'sum'),
        total_fare=('fare', 'sum'),
        low_ratings=('rating', lambda x: (x <= 2).sum())
    ).reset_index()

    user_cancelled = df[df['cancelled'] == 1].groupby('user_id').agg(
        cancellation_count=('ride_id', 'nunique')
    ).reset_index()


    user_summary = all_users.merge(
        user_completed, on='user_id', how='left'
        ).merge(user_cancelled, on='user_id', how='left')

    user_summary = user_summary.fillna(0)

    # usage segmentation 
    freq_thresh = user_summary['total_completed_trips'].median()
    fare_thresh = user_summary['total_fare'].median()

    # rules
    def usage_segment_user(row):
        if row['total_completed_trips'] >= freq_thresh and row['total_fare'] >= fare_thresh:
            return 'Loyal & High Spender'
        elif row['total_completed_trips'] >= freq_thresh and row['total_fare'] < fare_thresh:
            return 'Frequent but Low Spender'
        elif row['total_completed_trips'] < freq_thresh and row['total_fare'] >= fare_thresh:
            return 'Occasional Big Spender'
        else:
            return 'Infrequent & Low Spender'

    user_summary['usage_based_segment'] = user_summary.apply(usage_segment_user, axis=1)

    # value based
    fare_thresh = user_summary['total_fare'].median()
    rating_thresh = user_summary[user_summary['avg_rating']>0]['avg_rating'].median()
    cancel_thresh = user_summary['cancellation_count'].median()

    def value_segment_user(row):
        if row['total_fare'] >= fare_thresh and row['avg_rating'] >= rating_thresh:
            return 'High Value & Loyal'
        elif row['cancellation_count'] >= rating_thresh:
            return 'Unreliable / Risky'
        else:
            return 'Regular'

    user_summary['value_based_segment'] = user_summary.apply(value_segment_user, axis=1)

    segment_df = user_summary[['user_id', 'total_completed_trips', 'total_fare', 'total_distance' ,'avg_rating', 'cancellation_count', 'usage_based_segment','value_based_segment']]

    st.dataframe(segment_df)

    col1, col2 = st.columns(2)


    with col1:
        usage_segment_counts = user_summary['usage_based_segment'].value_counts()
        most_frequent_segment = usage_segment_counts.index[0]

        fig1, ax1 = plt.subplots(figsize=(5, 4))  # Ukuran lebih kecil
        sns.countplot(
            x='usage_based_segment',
            data=user_summary,
            order=usage_segment_counts.index,
            palette=['skyblue' if segment != most_frequent_segment else 'steelblue' for segment in usage_segment_counts.index],
            ax=ax1
        )

        ax1.set_title('Usage-Based Segmentation', fontsize=12)
        ax1.set_xlabel('Segment')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=15)

        for p in ax1.patches:
            ax1.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 5), textcoords='offset points')

        sns.despine()
        plt.tight_layout()
        st.pyplot(fig1)

    # === Plot 2: Value Segment (ukuran compact) ===
    with col2:
        value_segment_counts = user_summary['value_based_segment'].value_counts()
        max_count = value_segment_counts.max()
        most_frequent_segments = value_segment_counts[value_segment_counts == max_count].index.tolist()

        fig2, ax2 = plt.subplots(figsize=(5, 4))  # Ukuran lebih kecil
        sns.countplot(
            x='value_based_segment',
            data=user_summary,
            order=value_segment_counts.index,
            palette=['skyblue' if segment not in most_frequent_segments else 'steelblue' for segment in value_segment_counts.index],
            ax=ax2
        )

        ax2.set_title('Value & Satisfaction-Based Segmentation', fontsize=12)
        ax2.set_xlabel('Segment')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=15)

        for p in ax2.patches:
            ax2.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 5), textcoords='offset points')

        sns.despine()
        plt.tight_layout()
        st.pyplot(fig2)

# TAB 3 - Operational Insights
with tabs[2]:
    st.header("🚗 Operational Insights")

    driver_summary = df_completed.groupby('driver_id').agg(
        total_rides=('ride_id', 'count'),
        avg_rating=('rating', 'mean'),
        total_distance=('distance_km', 'sum'),
        total_fare=('fare', 'sum'),
        low_ratings=('rating', lambda x: (x <= 2).sum())
    ).reset_index()


    cancellation_stats = df[df['cancelled'] == 1].groupby('driver_id').agg(
        cancellations=('ride_id', 'count')
    ).reset_index()

    driver_summary = driver_summary.merge(cancellation_stats, on='driver_id', how='left')
    driver_summary['cancellations'] = driver_summary['cancellations'].fillna(0).astype(int)

    q1_rides = driver_summary['total_rides'].quantile(0.25)
    q3_rides = driver_summary['total_rides'].quantile(0.75)
    q3_distance = driver_summary['total_distance'].quantile(0.75)
    q1_rating = driver_summary['avg_rating'].quantile(0.25)
    high_cancel = driver_summary['cancellations'].quantile(0.75)

    def categorize_driver(data):
        if (data['total_rides'] >= q3_rides) and (data['total_distance'] >= q3_distance):
            return 'Overworked'
        elif (data['total_rides'] <= q1_rides) or (data['avg_rating'] <= 2) or (data['cancellations'] >= high_cancel):
            return 'Underperforming'
        else:
            return 'Normal'

    driver_summary['performance_category'] = driver_summary.apply(categorize_driver, axis=1)

    cat_driver_df = driver_summary[['driver_id', 'total_rides', 'total_fare', 'total_distance', 'avg_rating', 'cancellations', 'performance_category']]

    st.subheader("Driver Stats")
    st.dataframe(cat_driver_df)

    st.subheader("Peak Time")
    fig, ax = plt.subplots(figsize=(8, 5))  
    sns.countplot(
        x='time_period', 
        data=df, order=['Morning', 'Afternoon', 'Evening', 'Night'], 
        palette='Blues',
        ax=ax
        )


    ax.set_title('Ride Usage Time Pattern')
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Ride Count')
    sns.despine()
    plt.tight_layout()

    st.pyplot(fig)

# TAB 4 - Cancellation Model & Simulation
with tabs[3]:
    st.header("🧠 Cancellation Model & Simulation")
    model = joblib.load('classification_model.pkl')

    le_pickup = LabelEncoder()
    le_dropoff = LabelEncoder()
    
    st.subheader("Input Features")
    
    
    fare = st.number_input('Fare (in currency)', min_value=0.0, value=20.0, step=0.1)
    distance_km = st.number_input('Distance (in km)', min_value=0.0, value=10.0, step=0.1)
    ride_duration_min = st.number_input('Ride Duration (in minutes)', min_value=1.0, value=15.0, step=1.0)
    hour = st.number_input('Hour of the Ride (0-23)', min_value=0, max_value=23, value=14)
    request_day = st.selectbox('Day of the week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

   
    request_day = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(request_day)

    fare_per_km = fare / distance_km if distance_km > 0 else 0.0

    user_prev_cancel = st.number_input('Previous User Cancellations', min_value=0, value=1)

    pickup_location = st.selectbox('Pickup Location', ['Location A', 'Location B', 'Location C', 'Location D'])
    dropoff_location = st.selectbox('Dropoff Location', ['Location A', 'Location B', 'Location C', 'Location D'])

    
    pickup_enc = le_pickup.fit_transform(['Location A', 'Location B', 'Location C', 'Location D'])[0]  # Simulating the encoding
    dropoff_enc = le_dropoff.fit_transform(['Location A', 'Location B', 'Location C', 'Location D'])[0]

    
    input_data = pd.DataFrame({
        'fare': [fare],
        'distance_km': [distance_km],
        'ride_duration_min': [ride_duration_min],
        'hour': [hour],
        'request_day': [request_day],
        'fare_per_km': [fare_per_km],
        'user_prev_cancel': [user_prev_cancel],
        'pickup_enc': [pickup_enc],
        'dropoff_enc': [dropoff_enc]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    if prediction == 1:
        st.markdown("### Prediction: The ride **will be canceled**.")
    else:
        st.markdown("### Prediction: The ride **will not be canceled**.")

    
    st.subheader("Model Feature Importance")

  
    importances = model.feature_importances_
    features = ['fare', 'distance_km', 'ride_duration_min', 'hour', 'request_day', 'fare_per_km', 'user_prev_cancel', 'pickup_enc', 'dropoff_enc']

    importance_df = pd.DataFrame({'feature': features, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='importance', y='feature', data=importance_df, palette="YlOrBr")
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    sns.despine()
    st.pyplot(fig)
