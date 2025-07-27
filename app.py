# ==============================================================================
#                    ENHANCED AI HOUSING DASHBOARD PRO - CORRECTED
# ==============================================================================
# A completely redesigned housing prediction dashboard with modern UI,
# comprehensive data display, and advanced analytics capabilities.
#
# INSTALLATION:
# pip install streamlit pandas scikit-learn plotly matplotlib seaborn
#
# RUN:
# streamlit run housing_predictor.py
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="🏡 Housing Analytics Pro", 
    page_icon="🏠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Modern CSS Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styling */
    .stApp {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #2c3e50;
    }
    
    /* Main Content Area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 3rem;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(30, 60, 114, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        margin-top: 1rem;
        position: relative;
        z-index: 1;
    }
    
    /* Prediction Card */
    .prediction-showcase {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(238, 90, 36, 0.3);
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-showcase::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .prediction-showcase h2 {
        font-size: 4rem;
        font-weight: 800;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    .prediction-showcase p {
        font-size: 1.3rem;
        font-weight: 500;
        margin: 0.5rem 0;
        position: relative;
        z-index: 1;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        min-height: 200px;
    }
    
    /* Info Cards */
    .info-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(116, 185, 255, 0.3);
    }
    
    .info-card h3 {
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2d3436 0%, #636e72 100%);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(0, 184, 148, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(0, 184, 148, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.1);
        padding: 0.5rem;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: rgba(255,255,255,0.2);
        border-radius: 12px;
        color: #2c3e50;
        font-weight: 600;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
        color: white;
    }
    
    /* Metrics */
    .metric-container {
        background: rgba(255,255,255,0.9);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .metric-container h2 {
        color: #2c3e50;
        margin: 10px 0;
    }
    
    .metric-container h3 {
        color: #74b9ff;
        margin: 0;
    }
    
    /* Data Table Styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Enhanced Data Loading and Model Preparation ---
@st.cache_data
def load_comprehensive_data():
    """Load and prepare comprehensive housing data with enhanced features"""
    try:
        housing = fetch_california_housing()
        
        # Create comprehensive dataframe
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['MedHouseVal'] = housing.target
        
        # Add engineered features with proper error handling
        df['Rooms_per_Household'] = df['AveRooms'] / np.maximum(df['AveOccup'], 0.001)
        df['Bedrooms_ratio'] = df['AveBedrms'] / np.maximum(df['AveRooms'], 0.001)
        df['Population_density'] = df['Population'] / 1000  # Scaled for readability
        df['Income_per_room'] = df['MedInc'] / np.maximum(df['AveRooms'], 0.001)
        
        # Price categories for analysis
        df['Price_Category'] = pd.cut(df['MedHouseVal'], 
                                     bins=[0, 1.5, 3.0, 5.0, float('inf')], 
                                     labels=['Low', 'Medium', 'High', 'Luxury'])
        
        # Feature columns for modeling
        feature_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 
                        'AveOccup', 'Latitude', 'Longitude', 'Rooms_per_Household', 
                        'Bedrooms_ratio', 'Population_density', 'Income_per_room']
        
        X = df[feature_cols]
        y = df['MedHouseVal']
        
        return df, X, y, feature_cols
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

@st.cache_data
def train_models(X, y):
    """Train multiple models and return them with performance metrics"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            "🌲 Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1),
            "🎯 Ridge Regression": Pipeline([
                ('scaler', StandardScaler()), 
                ('model', Ridge(alpha=1.0))
            ]),
            "⚡ Lasso Regression": Pipeline([
                ('scaler', StandardScaler()), 
                ('model', Lasso(alpha=0.001, max_iter=1000))
            ])
        }
        
        model_performance = {}
        feature_importance = {}
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate performance
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_performance[name] = {'MAE': mae, 'R2': r2}
                
                # Feature importance
                if "Forest" in name:
                    importance = model.feature_importances_
                else:
                    importance = np.abs(model.named_steps['model'].coef_)
                
                feature_importance[name] = pd.Series(importance, index=X.columns).sort_values(ascending=False)
            
            except Exception as e:
                st.warning(f"Error training model {name}: {str(e)}")
                continue
        
        return models, model_performance, feature_importance
    
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return {}, {}, {}

# --- Initialize Session State ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# --- Load Data ---
if not st.session_state.data_loaded:
    with st.spinner("🚀 Loading housing data and training AI models..."):
        data_result = load_comprehensive_data()
        if data_result[0] is not None:
            df, X, y, feature_cols = data_result
            models, model_performance, feature_importance = train_models(X, y)
            
            # Store in session state
            st.session_state.df = df
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.feature_cols = feature_cols
            st.session_state.models = models
            st.session_state.model_performance = model_performance
            st.session_state.feature_importance = feature_importance
            st.session_state.data_loaded = True
        else:
            st.error("Failed to load data. Please refresh the page.")
            st.stop()

# Retrieve from session state
df = st.session_state.df
X = st.session_state.X
y = st.session_state.y
feature_cols = st.session_state.feature_cols
models = st.session_state.models
model_performance = st.session_state.model_performance
feature_importance = st.session_state.feature_importance

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🏡 Housing Analytics Pro</h1>
    <p>Advanced AI-Powered Real Estate Valuation & Market Analysis</p>
</div>
""", unsafe_allow_html=True)

# Check if models were successfully trained
if not models:
    st.error("No models were successfully trained. Please check your data and try again.")
    st.stop()

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("## 🎛️ Property Configuration")
    
    # Model selection
    available_models = list(models.keys())
    if available_models:
        selected_model = st.selectbox(
            "🤖 Choose AI Model", 
            available_models,
            help="Select the machine learning model for prediction"
        )
    else:
        st.error("No models available")
        st.stop()
    
    st.markdown("---")
    st.markdown("### 🏠 Property Features")
    
    # Input controls with better organization and error handling
    col1, col2 = st.columns(2)
    
    with col1:
        med_inc = st.slider('💰 Median Income ($10k)', 
                           float(X['MedInc'].min()), 
                           float(X['MedInc'].max()), 
                           float(X['MedInc'].mean()),
                           help="Median income in $10,000s")
        
        house_age = st.slider('🕐 House Age (yrs)', 
                             float(X['HouseAge'].min()), 
                             float(X['HouseAge'].max()), 
                             float(X['HouseAge'].mean()),
                             help="Median age of houses in years")
        
        ave_rooms = st.slider('🏠 Avg Rooms', 
                             float(X['AveRooms'].min()), 
                             15.0, 
                             float(X['AveRooms'].mean()),
                             help="Average number of rooms per household")
        
        ave_bedrms = st.slider('🛏️ Avg Bedrooms', 
                              float(X['AveBedrms'].min()), 
                              5.0, 
                              float(X['AveBedrms'].mean()),
                              help="Average number of bedrooms per household")
    
    with col2:
        population = st.slider('👥 Population', 
                              float(X['Population'].min()), 
                              10000.0, 
                              float(X['Population'].mean()),
                              help="Block group population")
        
        ave_occup = st.slider('🏘️ Avg Occupancy', 
                             float(X['AveOccup'].min()), 
                             10.0, 
                             float(X['AveOccup'].mean()),
                             help="Average occupancy per household")
        
        latitude = st.slider('📍 Latitude', 
                            32.5, 42.0, 36.78,
                            help="Geographic latitude")
        
        longitude = st.slider('📍 Longitude', 
                             -124.3, -114.3, -119.42,
                             help="Geographic longitude")
    
    st.markdown("---")
    
    # Model performance display
    if selected_model in model_performance:
        st.markdown("### 📊 Model Performance")
        perf = model_performance[selected_model]
        st.metric("R² Score", f"{perf['R2']:.3f}")
        st.metric("Mean Abs Error", f"${perf['MAE']*100000:,.0f}")
    
    st.markdown("---")
    predict_btn = st.button("🔮 Generate Prediction", use_container_width=True)

# --- Main Content ---
if predict_btn or 'last_prediction' in st.session_state:
    if predict_btn:
        try:
            # Prepare input data with proper error handling
            input_data = {
                'MedInc': med_inc,
                'HouseAge': house_age,
                'AveRooms': ave_rooms,
                'AveBedrms': ave_bedrms,
                'Population': population,
                'AveOccup': ave_occup,
                'Latitude': latitude,
                'Longitude': longitude,
                'Rooms_per_Household': ave_rooms / max(ave_occup, 0.001),
                'Bedrooms_ratio': ave_bedrms / max(ave_rooms, 0.001),
                'Population_density': population / 1000,
                'Income_per_room': med_inc / max(ave_rooms, 0.001)
            }
            
            input_df = pd.DataFrame([input_data], columns=feature_cols)
            
            # Generate predictions from all models
            predictions = {}
            for name, model in models.items():
                try:
                    pred = model.predict(input_df)[0] * 100000  # Convert to dollars
                    predictions[name] = pred
                except Exception as e:
                    st.warning(f"Error predicting with {name}: {str(e)}")
                    continue
            
            if selected_model in predictions:
                selected_prediction = predictions[selected_model]
                
                # Store in session state
                st.session_state.last_prediction = {
                    'input_data': input_data,
                    'predictions': predictions,
                    'selected_prediction': selected_prediction,
                    'selected_model_name': selected_model
                }
            else:
                st.error(f"Failed to generate prediction with {selected_model}")
        
        except Exception as e:
            st.error(f"Error generating prediction: {str(e)}")

    if 'last_prediction' in st.session_state:
        # Display results
        pred_data = st.session_state.last_prediction
        model_name_display = st.session_state.last_prediction['selected_model_name']
        
        # Main prediction display
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown(f"""
            <div class="prediction-showcase">
                <p>Predicted Value ({model_name_display})</p>
                <h2>${pred_data['selected_prediction']:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick stats
            market_avg = y.mean() * 100000
            difference = pred_data['selected_prediction'] - market_avg
            diff_pct = (difference / market_avg) * 100
            
            st.markdown(f"""
            <div class="info-card">
                <h3>📈 Market Comparison</h3>
                <p>Market Average: ${market_avg:,.0f}</p>
                <p>Difference: ${difference:+,.0f} ({diff_pct:+.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🧠 AI Analysis")
            
            # Top features analysis
            if model_name_display in feature_importance:
                top_features = feature_importance[model_name_display].head(3)
                
                st.markdown(f"**Primary Factor:** The model identifies **{top_features.index[0]}** as the most significant influence on this valuation.")
                st.markdown(f"**Secondary Factor:** The property's **{top_features.index[1]}** also plays a crucial role in determining the price.")
                
                # Market positioning
                if difference > 0:
                    st.markdown(f"🟢 This property is valued **{abs(diff_pct):.1f}% above** the market average, indicating premium characteristics.")
                else:
                    st.markdown(f"🔵 This property is valued **{abs(diff_pct):.1f}% below** the market average, suggesting a potential value opportunity.")
                
                # Model consensus
                if len(pred_data['predictions']) > 1:
                    pred_values = list(pred_data['predictions'].values())
                    consensus_range = (max(pred_values) - min(pred_values)) / pred_data['selected_prediction']
                    if consensus_range < 0.15:
                        st.markdown("✅ **Strong model consensus** provides high confidence in this prediction.")
                    else:
                        st.markdown("⚠️ **Moderate model variance** suggests that different AI approaches value this property type differently. Reviewing input parameters is recommended.")
        
        # Detailed Analysis Tabs
        st.markdown("---")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Model Comparison", "🎯 Feature Analysis", "🗺️ Location Map", "📋 Housing Data", "📈 Market Insights"])
        
        with tab1:
            st.markdown("### Model Predictions Comparison")
            
            if pred_data['predictions']:
                pred_df = pd.DataFrame.from_dict(pred_data['predictions'], orient='index', columns=['Prediction'])
                pred_df['Model'] = pred_df.index
                
                fig_models = px.bar(pred_df, x='Model', y='Prediction', title="Prediction Comparison Across Models",
                                   color='Prediction', color_continuous_scale='viridis')
                fig_models.update_traces(texttemplate='$%{y:,.0f}', textposition='outside')
                fig_models.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_models, use_container_width=True)
            
            st.markdown("### Model Performance Metrics")
            if model_performance:
                perf_df = pd.DataFrame(model_performance).T
                perf_df['R² Score'] = perf_df['R2'].apply(lambda x: f"{x:.3f}")
                perf_df['Mean Absolute Error'] = perf_df['MAE'].apply(lambda x: f"${x*100000:,.0f}")
                st.dataframe(perf_df[['R² Score', 'Mean Absolute Error']], use_container_width=True)
        
        with tab2:
            st.markdown("### Feature Importance Analysis")
            
            if model_name_display in feature_importance:
                top_features_chart = feature_importance[model_name_display].head(10)
                fig_features = px.bar(x=top_features_chart.values, y=top_features_chart.index, orientation='h',
                                      title=f"Top 10 Features - {model_name_display}", 
                                      labels={'x': 'Importance Score', 'y': 'Features'})
                fig_features.update_traces(texttemplate='%{x:.3f}', textposition='auto')
                fig_features.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_features, use_container_width=True)
        
        with tab3:
            st.markdown("### Interactive California Housing Map")
            
            try:
                map_sample = df.sample(n=min(5000, len(df)), random_state=42)
                
                fig_map = px.scatter_mapbox(map_sample, lat="Latitude", lon="Longitude", color="MedHouseVal",
                                            size="Population", color_continuous_scale="viridis", size_max=15,
                                            zoom=5, mapbox_style="carto-positron", title="California Housing Prices Distribution",
                                            hover_name="Price_Category", hover_data={'MedHouseVal': ':.2f', 'Population': True})
                
                fig_map.add_trace(go.Scattermapbox(lat=[pred_data['input_data']['Latitude']], 
                                                   lon=[pred_data['input_data']['Longitude']],
                                                   mode='markers', 
                                                   marker=go.scattermapbox.Marker(size=20, color='red', symbol='star'),
                                                   hoverinfo='text', 
                                                   text=f"Your Prediction: ${pred_data['selected_prediction']:,.0f}",
                                                   name="Your Property"))
                
                fig_map.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating map: {str(e)}")
        
        with tab4:
            st.markdown("### Explore the California Housing Dataset")
            
            try:
                colA, colB = st.columns(2)
                price_range = colA.slider("Filter by Price Range ($100k)", 
                                          float(df['MedHouseVal'].min()), 
                                          float(df['MedHouseVal'].max()),
                                          (float(df['MedHouseVal'].min()), float(df['MedHouseVal'].max())))
                selected_category = colB.selectbox("Filter by Price Category", 
                                                   ['All'] + list(df['Price_Category'].cat.categories))
                
                filtered_df = df[(df['MedHouseVal'] >= price_range[0]) & (df['MedHouseVal'] <= price_range[1])]
                if selected_category != 'All':
                    filtered_df = filtered_df[filtered_df['Price_Category'] == selected_category]
                
                st.dataframe(filtered_df.head(1000), use_container_width=True, height=400)
                
                @st.cache_data
                def convert_df_to_csv(df_to_convert):
                    return df_to_convert.to_csv(index=False).encode('utf-8')

                csv = convert_df_to_csv(filtered_df)
                st.download_button(label="📥 Download Filtered Data as CSV", data=csv,
                                   file_name='california_housing_filtered.csv', mime='text/csv')
            except Exception as e:
                st.error(f"Error in data exploration: {str(e)}")
        
        with tab5:
            st.markdown("### Market Insights & Analytics")
            
            try:
                colA, colB = st.columns(2)
                with colA:
                    fig_hist = px.histogram(df, x='MedHouseVal', nbins=50, title="Price Distribution")
                    fig_hist.update_layout(height=350, yaxis_title="Number of Districts", xaxis_title="House Value ($100k)")
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                with colB:
                    fig_box = px.box(df, x='Price_Category', y='MedHouseVal', title="Price Distribution by Category", 
                                     color='Price_Category')
                    fig_box.update_layout(height=350, xaxis_title="Price Category", yaxis_title="House Value ($100k)")
                    st.plotly_chart(fig_box, use_container_width=True)
                
                st.markdown("#### Feature Correlations with House Value")
                correlations = df[feature_cols + ['MedHouseVal']].corr()['MedHouseVal'].drop('MedHouseVal').sort_values(key=abs, ascending=False)
                
                fig_corr = px.bar(x=correlations.values, y=correlations.index, orientation='h',
                                  title="How Features Correlate with House Value",
                                  color=correlations.values, color_continuous_scale='RdYlBu')
                fig_corr.update_layout(height=500, yaxis={'categoryorder': 'total ascending'}, 
                                       xaxis_title="Correlation Coefficient")
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.error(f"Error in market insights: {str(e)}")

else:
    # --- Welcome Screen ---
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("## 🎯 Welcome to Housing Analytics Pro")
        st.markdown("This advanced platform combines machine learning with comprehensive market data to provide accurate real estate valuations.")
        
        st.markdown("### 🚀 Features:")
        st.markdown("""
        - **AI-Powered Predictions:** Three different ML models for robust valuations.
        - **Interactive Analysis:** Explore feature importance and model comparisons.  
        - **Market Insights:** Comprehensive data visualization and analytics.
        - **Geographic Mapping:** Interactive maps showing price distributions.
        - **Data Export:** Download filtered datasets for further analysis.
        """)
        
        st.markdown("*👈 Configure your property parameters in the sidebar and click 'Generate Prediction' to begin!*")
        
        st.markdown("### 🔍 Why Choose Our Platform?")
        st.markdown("""
        - **Accuracy**: Models trained on 20,000+ California housing records
        - **Speed**: Instant predictions with real-time analysis
        - **Transparency**: See exactly how predictions are made
        - **Comprehensive**: 12+ features analyzed for each property
        """)
    
    with col2:
        st.markdown("### 📊 Quick Dataset Overview")
        
        # Create a sample visualization
        sample_data = df.sample(n=1000, random_state=42)
        fig_preview = px.scatter(sample_data, x='MedInc', y='MedHouseVal', 
                                color='Price_Category', size='Population',
                                title="Income vs House Value Preview",
                                labels={'MedInc': 'Median Income ($10k)', 'MedHouseVal': 'House Value ($100k)'},
                                height=400)
        fig_preview.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_preview, use_container_width=True)
        
        st.markdown("### 📈 Live Data Insights")
        
        # Add some quick stats
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("🏠 Total Properties", f"{len(df):,}")
            st.metric("💰 Avg Price", f"${df['MedHouseVal'].mean()*100000:,.0f}")
        with col2b:
            st.metric("📍 Locations", f"{df['Latitude'].nunique():,}")
            st.metric("🔧 Features", len(feature_cols))
    
    # Dataset overview
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📊 Dataset Size</h3>
            <h2>{len(df):,}</h2>
            <p>Housing Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>💰 Average Price</h3>
            <h2>${df['MedHouseVal'].mean() * 100000:,.0f}</h2>
            <p>Across California</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>⚙️ Features Analyzed</h3>
            <h2>{len(feature_cols)}</h2>
            <p>Data Points per Record</p>
        </div>
        """, unsafe_allow_html=True)