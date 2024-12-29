import streamlit as st
import pickle
import pandas as pd

# Load the trained model pipelines
model_lr = pickle.load(open('ModelLR.pkl', 'rb'))
model_rfr = pickle.load(open('ModelRFR.pkl', 'rb'))

# Teams and cities
teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
         'England', 'West Indies', 'Pakistan', 'Sri Lanka']

cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town',
          'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban',
          'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion',
          'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton',
          'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi',
          'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff',
          'Christchurch', 'Trinidad']

st.title('Cricket Score Predictor')

# Model selection
model_choice = st.radio('Select the prediction model', ['Linear Regression', 'Random Forest Regressor'])

# Input fields
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

city = st.selectbox('Select city', sorted(cities))

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score', min_value=0)
with col4:
    overs = st.number_input('Overs done (works for over > 5)', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

last_five = st.number_input('Runs scored in last 5 overs', min_value=0, max_value=current_score)

if st.button('Predict Score'):
    if batting_team == bowling_team:
        st.error("Batting team and Bowling team cannot be the same. Please select different teams.")
    elif overs == 0:
        st.error("Overs cannot be zero. Please enter a valid value.")
    else:
        # Derived features
        balls_left = 120 - int(overs * 6)
        wickets_left = 10 - wickets
        crr = current_score / overs

        # Prepare input DataFrame
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'current_score': [current_score],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'crr': [crr],
            'last_five': [last_five]
        })

        try:
            # Select model based on user choice
            if model_choice == 'Linear Regression':
                result = model_lr.predict(input_df)
            else:
                result = model_rfr.predict(input_df)

            st.header("Predicted Score - " + str(int(result[0])))
        except Exception as e:
            st.error(f"An error occurred: {e}")
