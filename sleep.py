import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

st.title("Sleep Patterns & Productivity Insights")

# User Inputs
st.header("Enter Participant Data")

# No. of participants
participants = st.number_input("Number of Participants", min_value=1, value=1, step=1)

# Empty DataFrame to store participant data
participant_data = pd.DataFrame()

# Collect data for each participant
for participant_id in range(1, participants + 1):
    st.subheader(f"Participant {participant_id}")
    
    # Sleep Data
    sleep_duration = st.number_input(f"Participant {participant_id} Sleep Duration (hours)", min_value=0.0, value=7.5, step=0.1)
    sleep_quality = st.number_input(f"Participant {participant_id} Sleep Quality (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
    sleep_consistency = st.number_input(f"Participant {participant_id} Sleep Consistency", min_value=0.0, value=0.8, step=0.1)
    
    # Work Schedules
    work_start = st.time_input(f"Participant {participant_id} Work Start Time", value=datetime.strptime('08:00', '%H:%M').time())
    work_end = st.time_input(f"Participant {participant_id} Work End Time", value=datetime.strptime('18:00', '%H:%M').time())
    breaks = st.number_input(f"Participant {participant_id} Number of Breaks", min_value=0, value=1, step=1)
    irregular_hours = st.checkbox(f"Participant {participant_id} Irregular Working Hours")
    
    # Performance Metrics
    task_completion_time = st.number_input(f"Participant {participant_id} Task Completion Time (hours)", min_value=0.0, value=3.0, step=0.1)
    project_deadlines_met = st.checkbox(f"Participant {participant_id} Project Deadlines Met")
    self_reported_productivity = st.number_input(f"Participant {participant_id} Self-Reported Productivity (1-10)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)

    # Append participant data to DataFrame
    participant_data = participant_data.append({
        'ParticipantID': participant_id,
        'SleepDuration': sleep_duration,
        'SleepQuality': sleep_quality,
        'SleepConsistency': sleep_consistency,
        'WorkStart': work_start,
        'WorkEnd': work_end,
        'Breaks': breaks,
        'IrregularHours': irregular_hours,
        'TaskCompletionTime': task_completion_time,
        'ProjectDeadlinesMet': project_deadlines_met,
        'SelfReportedProductivity': self_reported_productivity
    }, ignore_index=True)

# Feature Engineering for Work Schedules

# Convert 'WorkStart' and 'WorkEnd' columns to datetime
participant_data['WorkStart'] = participant_data['WorkStart'].apply(lambda x: datetime.combine(datetime.today(), x))
participant_data['WorkEnd'] = participant_data['WorkEnd'].apply(lambda x: datetime.combine(datetime.today(), x))

# Calculate regularity of work hours (assuming 8 hours as regular work hours)
participant_data['WorkRegularity'] = (participant_data['WorkEnd'] - participant_data['WorkStart']) == pd.Timedelta(hours=8)

# Calculate duration of work hours
participant_data['WorkDuration'] = (participant_data['WorkEnd'] - participant_data['WorkStart']).dt.total_seconds() / 3600

# Display the participant data
st.header("Participant Data")
st.write(participant_data)


# Check if the dataset is large enough for train-test split
if participants >= 2 and participants * 0.2 >= 1:
    # Select relevant features and target variable
    features = ['SleepDuration', 'SleepQuality', 'SleepConsistency', 'WorkRegularity', 'WorkDuration']
    target = 'SelfReportedProductivity'

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(participant_data[features], participant_data[target], test_size=0.2, random_state=42)

    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Check if the training set is not empty
    if not X_train_scaled.empty:
        model.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        predictions = model.predict(X_test_scaled)

        # Evaluate the model performance
        mse = mean_squared_error(y_test, predictions)
        st.write(f'Mean Squared Error: {mse}')

        # Feature Importance
        feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        st.write('\nFeature Importance:')
        st.bar_chart(feature_importance)
else:
    st.write("Not enough data for train-test split. Please enter data for at least 2 participants.")


# Analysis and Visualization
st.header("Analysis and Visualization")

# 1. Explore Sleep Patterns Visually
st.subheader("Explore Sleep Patterns Visually")

# Visualize the distribution of Sleep Duration
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.histplot(participant_data['SleepDuration'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Sleep Duration')

# Visualize the distribution of Sleep Quality
plt.subplot(2, 2, 2)
sns.histplot(participant_data['SleepQuality'], bins=20, kde=True, color='salmon')
plt.title('Distribution of Sleep Quality')

# Visualize the distribution of Sleep Consistency
plt.subplot(2, 2, 3)
sns.histplot(participant_data['SleepConsistency'], bins=20, kde=True, color='lightgreen')
plt.title('Distribution of Sleep Consistency')

st.pyplot(plt)

# 2. Analyze Work Schedules
st.subheader("Analyze Work Schedules")

# Visualize the distribution of Breaks during the day
plt.figure(figsize=(8, 4))
sns.countplot(x='Breaks', data=participant_data, palette='pastel')
plt.title('Distribution of Breaks during the Day')
plt.xlabel('Number of Breaks')
plt.ylabel('Count')

st.pyplot(plt)

# Visualize the distribution of Irregular Working Hours
plt.figure(figsize=(6, 4))
sns.countplot(x='IrregularHours', data=participant_data, palette='pastel')
plt.title('Distribution of Irregular Working Hours')
plt.xlabel('Irregular Working Hours (1: Yes, 0: No)')

st.pyplot(plt)

# 3. Correlation Analysis
st.subheader("Correlation Analysis")

# Select relevant columns for correlation analysis
selected_columns = ['SleepDuration', 'SleepQuality', 'SleepConsistency', 'WorkRegularity', 'WorkDuration', 'TaskCompletionTime', 'SelfReportedProductivity']

# Create a correlation matrix
correlation_matrix = participant_data[selected_columns].corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')

st.pyplot(plt)

# 4. Predictive Modeling
st.subheader("Predictive Modeling")

# Feature Engineering
# Extract relevant features from sleep data
participant_data['AverageSleepDuration'] = participant_data['SleepDuration']
participant_data['SleepQualityConsistency'] = participant_data['SleepQuality'] * participant_data['SleepConsistency']

# Feature Engineering for Work Schedules
# Assuming 'merged_data' is your DataFrame containing sleep, work, and performance data
# Calculate regularity of work hours (assuming 8 hours as regular work hours)
participant_data['WorkRegularity'] = (participant_data['WorkEnd'] - participant_data['WorkStart']) == pd.Timedelta(hours=8)

# Calculate duration of work hours
participant_data['WorkDuration'] = (participant_data['WorkEnd'] - participant_data['WorkStart']).dt.total_seconds() / 3600

# Select relevant features and target variable
features = ['SleepDuration', 'SleepQuality', 'SleepConsistency', 'WorkRegularity', 'WorkDuration']
target = 'SelfReportedProductivity'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(participant_data[features], participant_data[target], test_size=0.2, random_state=42)

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
predictions = model.predict(X_test_scaled)

# Evaluate the model performance
mse = mean_squared_error(y_test, predictions)
st.write(f'Mean Squared Error: {mse}')

# Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
st.subheader('Feature Importance')
st.write(feature_importance)

# Visualize Feature Importance
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance, y=feature_importance.index, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance Score')

st.pyplot(plt)

# 5. Derive Insights and Recommendations
st.subheader("Insights and Recommendations")

# Example Insights
insight_sleep_duration = "Participants with longer average sleep duration tend to have higher self-reported productivity."
insight_work_regularity = "Participants with regular work hours (assumed 8 hours) show a positive correlation with productivity."

# Example Recommendations
recommendation_sleep_duration = "Encourage employees to prioritize sufficient sleep duration to enhance productivity."
recommendation_work_regularity = "Promote regular work hours and breaks to improve overall work-life balance."

# Display insights and recommendations
st.write("Insights:")
st.write(insight_sleep_duration)
st.write(insight_work_regularity)

st.write("\nRecommendations:")
st.write(recommendation_sleep_duration)
st.write(recommendation_work_regularity)

# Display predictive model insights and recommendations if applicable
if mse < 0.5:  # Assuming a threshold for acceptable mean squared error
    insight_predictive_model = "The predictive model suggests a strong relationship between sleep/work features and productivity."

    # Additional recommendations based on the predictive model
    recommendation_model = "Consider integrating insights from the predictive model into employee wellness programs."

    st.write("\nPredictive Model Insights:")
    st.write(insight_predictive_model)
    st.write("\nPredictive Model Recommendations:")
    st.write(recommendation_model)
