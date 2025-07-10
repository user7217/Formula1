import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_collector import F1DataCollector
from feature_engineer import FeatureEngineer
from ml_models import MLModelManager
from predictor import RacePredictor
from json_database import JSONDatabaseManager
from driver_analytics import DriverAnalytics
from utils import load_cached_data, save_cached_data, get_current_season_schedule

# Page configuration
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏎️ Formula 1 Race Predictor")
st.markdown("Advanced ML-powered race prediction using FastF1 telemetry data")

# Add explanation box
with st.expander("ℹ️ How This Works", expanded=False):
    st.markdown("""
    **This system predicts 2025 F1 race results using machine learning:**
    
    1. **📊 Data Collection**: Gathers real F1 telemetry, race results, and weather data from 2020-2024 seasons
    2. **🤖 Model Training**: Trains multiple ML algorithms (Random Forest, XGBoost, Neural Networks) on historical data
    3. **🔮 2025 Predictions**: Uses trained models to predict upcoming 2025 race outcomes
    4. **✅ Accuracy Testing**: Validates predictions against actual 2024 results to measure accuracy
    
    **Note**: 2025 race data doesn't exist yet - that's what we're predicting! We train on past data to forecast the future.
    """)

# Initialize JSON database (lighter storage)
db_manager = JSONDatabaseManager()
db_init_success = False
try:
    # Ensure database exists and is healthy
    db_manager.ensure_database_exists()
    db_init_success = db_manager.create_tables()
    if db_init_success:
        st.success("✅ JSON database initialized successfully - lighter and faster storage!")
    else:
        st.warning("⚠️ Database initialization issues. Using cache fallback.")
except Exception as e:
    st.warning(f"⚠️ Database initialization failed: {str(e)}, using cache fallback.")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'predictor_ready' not in st.session_state:
    st.session_state.predictor_ready = False

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Data Collection", "Model Training", "Race Predictions", "2024 Validation", "Model Performance", "Database Management", "Driver Analytics"]
)

@st.cache_data(ttl=3600)
def initialize_components():
    """Initialize all components with caching"""
    collector = F1DataCollector()
    engineer = FeatureEngineer()
    model_manager = MLModelManager()
    predictor = RacePredictor()
    return collector, engineer, model_manager, predictor

collector, engineer, model_manager, predictor = initialize_components()
analytics = DriverAnalytics()

if page == "Data Collection":
    st.header("📊 Data Collection & Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Collection Settings")
        st.info("📊 Collect historical F1 data to train machine learning models. This data will be used to predict 2025 race results.")
        
        seasons = st.multiselect(
            "Select seasons to collect training data",
            options=[2020, 2021, 2022, 2023, 2024],
            default=[2022, 2023, 2024],
            help="Historical seasons used to train the prediction models"
        )
        
        include_weather = st.checkbox("Include weather data", value=True)
        include_telemetry = st.checkbox("Include detailed telemetry", value=True)
        
        if st.button("Start Data Collection", type="primary"):
            if not seasons:
                st.error("Please select at least one season")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    total_seasons = len(seasons)
                    all_data = {}
                    
                    for i, season in enumerate(seasons):
                        status_text.text(f"Collecting data for {season} season...")
                        progress_bar.progress((i + 1) / total_seasons)
                        
                        season_data = collector.collect_season_data(
                            season, 
                            include_weather=include_weather,
                            include_telemetry=include_telemetry
                        )
                        all_data[season] = season_data
                    
                    # Process and engineer features
                    status_text.text("Processing and engineering features...")
                    processed_data = engineer.process_all_data(all_data)
                    
                    # Save to cache and database
                    status_text.text("Saving data to cache and database...")
                    save_cached_data('processed_data', processed_data)
                    
                    # Prioritize database storage for quick future access
                    db_saved = False
                    if db_init_success:
                        try:
                            if db_manager.save_race_data(processed_data):
                                db_saved = True
                                st.info("💾 Data successfully saved to JSON database for quick future access")
                            else:
                                st.warning("⚠️ Database save had issues, data saved to cache")
                        except Exception as e:
                            st.warning(f"⚠️ Database save failed: {str(e)}, data saved to cache")
                    else:
                        st.warning("⚠️ Database not available, data saved to cache only")
                    
                    st.session_state.data_loaded = True
                    
                    status_text.text("✅ Data collection completed successfully!")
                    st.success(f"Collected data for {len(seasons)} seasons with {len(processed_data)} total records")
                    
                    # Show storage summary
                    storage_msg = "💾 Data storage complete - "
                    if db_saved:
                        storage_msg += "stored in database for instant future access"
                    else:
                        storage_msg += "stored in local cache"
                    st.success(storage_msg)
                    
                except Exception as e:
                    st.error(f"Error during data collection: {str(e)}")
                    st.info("This might be due to API rate limits or network issues. Try again in a few minutes.")
    
    with col2:
        st.subheader("Data Status")
        
        # Check for data in cache and database
        cached_data = load_cached_data('processed_data')
        db_data = []
        
        # Prioritize JSON database data for faster access
        if db_init_success:
            try:
                db_data = db_manager.get_race_data(limit=5000)  # Get more records from JSON database
                if db_data:
                    st.info(f"📊 Found {len(db_data)} records in JSON database for quick access")
            except Exception as e:
                st.warning(f"JSON database access issue: {str(e)}")
                db_data = []
        
        if cached_data is not None or db_data:
            data_source = "cache" if cached_data else "database"
            data_to_use = cached_data if cached_data else db_data
            
            st.success(f"✅ Data available from {data_source} ({len(data_to_use)} records)")
            st.session_state.data_loaded = True
            
            # Show data summary
            df = pd.DataFrame(data_to_use)
            if not df.empty:
                st.write("**Data Summary:**")
                st.write(f"- Total races: {df['race_name'].nunique()}")
                st.write(f"- Seasons: {sorted(df['season'].unique())}")
                st.write(f"- Drivers: {df['driver'].nunique()}")
                st.write(f"- Teams: {df['team'].nunique()}")
                
                # Show feature columns
                feature_cols = [col for col in df.columns if col not in ['race_name', 'driver', 'team', 'position']]
                st.write(f"- Features: {len(feature_cols)}")
                
                # Database status
                if db_data:
                    st.write(f"🗄️ Database contains {len(db_data)} records")
        else:
            st.warning("⚠️ No data found in cache or database. Please collect data first.")

elif page == "Model Training":
    st.header("🤖 Machine Learning Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("Please collect data first from the Data Collection page.")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Configuration")
        
        models_to_train = st.multiselect(
            "Select models to train",
            options=["Random Forest", "XGBoost", "Neural Network", "Gradient Boosting"],
            default=["Random Forest", "XGBoost"]
        )
        
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
        
        target_variable = st.selectbox(
            "Prediction target",
            options=["position", "points", "podium_finish"],
            index=0
        )
        
        if st.button("Train Models", type="primary"):
            cached_data = load_cached_data('processed_data')
            if cached_data is None:
                st.error("No data available for training")
                st.stop()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Prepare data for training
                status_text.text("Preparing training data...")
                progress_bar.progress(0.1)
                
                df = pd.DataFrame(cached_data)
                X, y = engineer.prepare_training_data(df, target_variable)
                
                # Train models
                total_models = len(models_to_train)
                model_results = {}
                
                for i, model_name in enumerate(models_to_train):
                    status_text.text(f"Training {model_name}...")
                    progress_bar.progress(0.2 + (i * 0.7) / total_models)
                    
                    results = model_manager.train_model(
                        model_name, X, y, test_size=test_size, cv_folds=cv_folds
                    )
                    model_results[model_name] = results
                
                status_text.text("Saving trained models...")
                progress_bar.progress(0.9)
                
                # Save results to cache and database
                status_text.text("Saving model results...")
                save_cached_data('model_results', model_results)
                save_cached_data('trained_models', model_manager.models)
                db_manager.save_model_results(model_results)
                st.session_state.models_trained = True
                
                status_text.text("✅ Model training completed!")
                progress_bar.progress(1.0)
                
                st.success(f"Successfully trained {len(models_to_train)} models")
                st.info("💾 Model results saved to database")
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
    
    with col2:
        st.subheader("Training Results")
        
        model_results = load_cached_data('model_results')
        if model_results:
            st.session_state.models_trained = True
            
            # Create performance comparison
            performance_data = []
            for model_name, results in model_results.items():
                performance_data.append({
                    'Model': model_name,
                    'Test Accuracy': results.get('test_score', 0),
                    'CV Score': results.get('cv_score_mean', 0),
                    'CV Std': results.get('cv_score_std', 0)
                })
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                
                # Bar chart of model performance
                fig = px.bar(
                    perf_df, 
                    x='Model', 
                    y='Test Accuracy',
                    title="Model Performance Comparison",
                    error_y='CV Std'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance table
                st.write("**Detailed Performance Metrics:**")
                st.dataframe(perf_df, use_container_width=True)
        else:
            st.info("No trained models found. Please train models first.")

elif page == "Race Predictions":
    st.header("🏁 2025 Season Race Predictions")
    st.info("🔮 Generate predictions for upcoming 2025 F1 races using trained machine learning models.")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first from the Model Training page using historical data (2020-2024).")
        st.stop()
    
    # Load current season schedule
    schedule_2025 = get_current_season_schedule(2025)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Prediction Settings")
        
        if schedule_2025:
            race_options = [f"{race['race_name']} - {race['date']}" for race in schedule_2025]
            selected_race = st.selectbox("Select race to predict", race_options)
            
            model_choice = st.selectbox(
                "Select prediction model",
                options=["Ensemble", "Random Forest", "XGBoost", "Neural Network"]
            )
            
            if st.button("Generate Predictions", type="primary"):
                try:
                    # Get race index
                    race_idx = race_options.index(selected_race)
                    race_info = schedule_2025[race_idx]
                    
                    # Generate predictions
                    with st.spinner("Generating predictions..."):
                        predictions = predictor.predict_race(
                            race_info, 
                            model_choice.lower().replace(" ", "_")
                        )
                    
                    # Save predictions to database
                    db_manager.save_predictions(predictions, race_info, model_choice.lower())
                    
                    st.session_state.current_predictions = predictions
                    st.session_state.current_race = race_info
                    st.success("Predictions generated successfully!")
                    st.info("💾 Predictions saved to database")
                    
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
        else:
            st.info("Loading 2025 race schedule...")
    
    with col2:
        st.subheader("Race Predictions")
        
        if hasattr(st.session_state, 'current_predictions'):
            predictions = st.session_state.current_predictions
            race_info = st.session_state.current_race
            
            st.write(f"**Race:** {race_info['race_name']}")
            st.write(f"**Date:** {race_info['date']}")
            st.write(f"**Circuit:** {race_info.get('circuit', 'TBD')}")
            
            # Predictions table
            pred_df = pd.DataFrame(predictions)
            pred_df = pred_df.sort_values('predicted_position')
            
            # Add confidence intervals if available
            fig = px.bar(
                pred_df.head(10), 
                x='driver', 
                y='confidence',
                color='predicted_position',
                title="Top 10 Predicted Finishing Positions",
                labels={'confidence': 'Prediction Confidence', 'driver': 'Driver'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed predictions table
            st.write("**Detailed Predictions:**")
            display_cols = ['predicted_position', 'driver', 'team', 'confidence', 'predicted_points']
            if all(col in pred_df.columns for col in display_cols):
                st.dataframe(
                    pred_df[display_cols].head(20), 
                    use_container_width=True,
                    column_config={
                        'predicted_position': 'Position',
                        'confidence': st.column_config.ProgressColumn(
                            'Confidence',
                            min_value=0,
                            max_value=1
                        )
                    }
                )
        else:
            st.info("Select a race and generate predictions to view results.")

elif page == "2024 Validation":
    st.header("✅ 2024 Season Accuracy Validation")
    st.info("🎯 Test prediction accuracy by comparing model predictions against actual 2024 F1 race results.")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first from the Model Training page using historical data.")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Validation Settings")
        
        validation_races = st.multiselect(
            "Select 2024 races for validation",
            options=[
                "Bahrain GP", "Saudi Arabia GP", "Australia GP", "Japan GP",
                "China GP", "Miami GP", "Emilia Romagna GP", "Monaco GP",
                "Canada GP", "Spain GP", "Austria GP", "Britain GP",
                "Hungary GP", "Belgium GP", "Netherlands GP", "Italy GP",
                "Azerbaijan GP", "Singapore GP", "United States GP", "Mexico GP",
                "Brazil GP", "Las Vegas GP", "Qatar GP", "Abu Dhabi GP"
            ],
            default=["Bahrain GP", "Australia GP", "Monaco GP", "Britain GP"]
        )
        
        if st.button("Run Validation", type="primary"):
            if not validation_races:
                st.error("Please select at least one race for validation")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    validation_results = {}
                    total_races = len(validation_races)
                    
                    for i, race in enumerate(validation_races):
                        status_text.text(f"Validating {race}...")
                        progress_bar.progress((i + 1) / total_races)
                        
                        # Get actual 2024 results and compare with predictions
                        race_validation = predictor.validate_race_prediction(race, 2024)
                        validation_results[race] = race_validation
                        
                        # Save to database
                        if 'accuracy_metrics' in race_validation:
                            db_manager.save_validation_results(race_validation)
                    
                    save_cached_data('validation_results', validation_results)
                    status_text.text("✅ Validation completed!")
                    st.success(f"Validated predictions for {len(validation_races)} races")
                    st.info("💾 Validation results saved to database")
                    
                except Exception as e:
                    st.error(f"Error during validation: {str(e)}")
    
    with col2:
        st.subheader("Validation Results")
        
        validation_results = load_cached_data('validation_results')
        if validation_results:
            # Calculate overall accuracy metrics
            all_accuracies = []
            race_accuracies = []
            
            for race, results in validation_results.items():
                if 'accuracy_metrics' in results:
                    metrics = results['accuracy_metrics']
                    all_accuracies.append(metrics.get('position_accuracy', 0))
                    race_accuracies.append({
                        'Race': race,
                        'Position Accuracy': metrics.get('position_accuracy', 0),
                        'Top 3 Accuracy': metrics.get('top3_accuracy', 0),
                        'Points Accuracy': metrics.get('points_accuracy', 0)
                    })
            
            if race_accuracies:
                # Overall accuracy
                avg_accuracy = np.mean(all_accuracies)
                st.metric("Average Position Accuracy", f"{avg_accuracy:.1%}")
                
                # Race-by-race accuracy
                acc_df = pd.DataFrame(race_accuracies)
                
                fig = px.line(
                    acc_df, 
                    x='Race', 
                    y=['Position Accuracy', 'Top 3 Accuracy', 'Points Accuracy'],
                    title="Prediction Accuracy by Race",
                    markers=True
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.write("**Detailed Validation Results:**")
                st.dataframe(acc_df, use_container_width=True)
        else:
            st.info("No validation results available. Please run validation first.")

elif page == "Model Performance":
    st.header("📈 Model Performance Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance")
        
        model_results = load_cached_data('model_results')
        if model_results:
            model_choice = st.selectbox(
                "Select model for feature analysis",
                options=list(model_results.keys())
            )
            
            if model_choice and 'feature_importance' in model_results[model_choice]:
                importance_data = model_results[model_choice]['feature_importance']
                
                # Create feature importance plot
                fig = px.bar(
                    x=list(importance_data.values()),
                    y=list(importance_data.keys()),
                    orientation='h',
                    title=f"{model_choice} Feature Importance",
                    labels={'x': 'Importance Score', 'y': 'Features'}
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model results available for feature analysis.")
    
    with col2:
        st.subheader("Model Comparison")
        
        if model_results:
            # Create comprehensive comparison
            comparison_data = []
            for model_name, results in model_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Training Accuracy': results.get('train_score', 0),
                    'Test Accuracy': results.get('test_score', 0),
                    'Cross-Validation Mean': results.get('cv_score_mean', 0),
                    'Cross-Validation Std': results.get('cv_score_std', 0)
                })
            
            comp_df = pd.DataFrame(comparison_data)
            
            # Radar chart for model comparison
            fig = go.Figure()
            
            metrics = ['Training Accuracy', 'Test Accuracy', 'Cross-Validation Mean']
            for _, row in comp_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row[metric] for metric in metrics],
                    theta=metrics,
                    fill='toself',
                    name=row['Model']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Model Performance Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics table
            st.write("**Performance Metrics:**")
            st.dataframe(comp_df, use_container_width=True)
        else:
            st.info("No model comparison data available.")
    
    # Additional analytics
    st.subheader("Prediction Confidence Analysis")
    
    validation_results = load_cached_data('validation_results')
    if validation_results:
        # Analyze prediction confidence vs accuracy
        confidence_accuracy = []
        
        for race, results in validation_results.items():
            if 'predictions' in results and 'actual_results' in results:
                predictions = results['predictions']
                for pred in predictions:
                    confidence_accuracy.append({
                        'Race': race,
                        'Driver': pred.get('driver', ''),
                        'Confidence': pred.get('confidence', 0),
                        'Prediction Correct': pred.get('correct_prediction', False)
                    })
        
        if confidence_accuracy:
            conf_df = pd.DataFrame(confidence_accuracy)
            
            # Confidence vs accuracy scatter plot
            fig = px.scatter(
                conf_df,
                x='Confidence',
                y='Prediction Correct',
                color='Race',
                title="Prediction Confidence vs Accuracy",
                labels={'Prediction Correct': 'Prediction Accuracy (0=Wrong, 1=Correct)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No validation results available for confidence analysis.")

elif page == "Database Management":
    st.header("🗄️ JSON Database Management")
    st.info("📊 Now using lightweight JSON database for faster performance and easier portability!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Database Overview")
        
        # Database health check
        health_status = db_manager.check_database_health()
        
        # Display health status
        if health_status["status"] == "healthy":
            st.success("✅ Database is healthy")
        elif health_status["status"] == "warning":
            st.warning("⚠️ Database has some issues")
        else:
            st.error("❌ Database has serious problems")
        
        # Show database files status
        st.write("**Database Files:**")
        for file_type, file_info in health_status["files"].items():
            status_icon = "✅" if file_info["exists"] and file_info["readable"] else "❌"
            st.write(f"{status_icon} {file_type}: {file_info['records']} records, {file_info['size_mb']} MB")
        
        st.write(f"**Total database size:** {health_status['total_size_mb']} MB")
        
        # Show issues if any
        if health_status["issues"]:
            st.write("**Issues found:**")
            for issue in health_status["issues"]:
                st.write(f"- {issue}")
        
        # Show recommendations
        if health_status["recommendations"]:
            st.write("**Recommendations:**")
            for rec in health_status["recommendations"]:
                st.write(f"- {rec}")
        
        # Database creation and repair tools
        st.subheader("Database Tools")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("🔧 Ensure Database Exists", help="Create missing database files"):
                with st.spinner("Checking and creating database files..."):
                    success = db_manager.ensure_database_exists()
                    if success:
                        st.success("Database files checked and created successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to create database files")
        
        with col_b:
            if st.button("🔄 Refresh Status", help="Refresh database health status"):
                st.rerun()
        
        # Get database statistics
        try:
            db_race_data = db_manager.get_race_data(limit=5)
            model_history = db_manager.get_model_performance_history()
            accuracy_summary = db_manager.get_prediction_accuracy_summary()
            
            st.write("**Race Data:**")
            if db_race_data:
                total_records = len(db_manager.get_race_data())
                st.write(f"- Total records: {total_records}")
                df = pd.DataFrame(db_race_data)
                if not df.empty:
                    st.write(f"- Seasons covered: {sorted(df['season'].unique())}")
                    st.write(f"- Unique drivers: {df['driver'].nunique()}")
                    st.write(f"- Unique races: {df['race_name'].nunique()}")
            else:
                st.write("- No race data found")
            
            st.write("**Model Performance History:**")
            if not model_history.empty:
                st.write(f"- Training sessions: {len(model_history)}")
                st.write(f"- Models tested: {model_history['model_name'].nunique()}")
                best_model = model_history.loc[model_history['test_score'].idxmax()]
                st.write(f"- Best performing: {best_model['model_name']} ({best_model['test_score']:.3f})")
            else:
                st.write("- No model training history")
            
            st.write("**Prediction Accuracy:**")
            if accuracy_summary:
                st.write(f"- Validated races: {accuracy_summary.get('total_validated_races', 0)}")
                st.write(f"- Average position accuracy: {accuracy_summary.get('average_position_accuracy', 0):.1%}")
                st.write(f"- Average top-3 accuracy: {accuracy_summary.get('average_top3_accuracy', 0):.1%}")
                st.write(f"- Average position error: {accuracy_summary.get('average_position_error', 0):.1f}")
            else:
                st.write("- No validation results available")
                
        except Exception as e:
            st.error(f"Error retrieving database statistics: {str(e)}")
    
    with col2:
        st.subheader("Database Operations")
        
        # Data management buttons
        if st.button("🔄 Refresh Database Stats"):
            st.rerun()
        
        if st.button("🧹 Clean Old Data (30+ days)"):
            try:
                db_manager.cleanup_old_data(days_old=30)
                st.success("✅ Cleaned up old predictions and validation data")
            except Exception as e:
                st.error(f"Error cleaning database: {str(e)}")
        
        # Export options
        st.write("**Export Data:**")
        
        if st.button("📁 Export Recent Race Data"):
            try:
                recent_data = db_manager.get_race_data(limit=1000)
                if recent_data:
                    df = pd.DataFrame(recent_data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"f1_race_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No race data available for export")
            except Exception as e:
                st.error(f"Error exporting data: {str(e)}")
        
        # Database health check
        st.write("**JSON Database Health:**")
        try:
            # Test JSON database operations
            stats = db_manager.get_database_stats()
            test_data = db_manager.get_race_data(limit=1)
            
            if stats and test_data is not None:
                st.success("✅ JSON database healthy and accessible")
                
                # Show record counts
                st.write(f"- Race data records: {stats.get('race_records', 0)}")
                st.write(f"- Trained models: {stats.get('trained_models', 0)}")
                st.write(f"- Total predictions: {stats.get('total_predictions', 0)}")
                st.write(f"- Validation records: {stats.get('validation_records', 0)}")
                st.write(f"- Total database size: {stats.get('total_database_size', '0 KB')}")
            else:
                st.warning("⚠️ JSON database partially accessible")
                
        except Exception as e:
            st.error(f"❌ JSON database health check failed: {str(e)}")
    
    # Recent data preview
    st.subheader("Recent Data Preview")
    
    tab1, tab2, tab3 = st.tabs(["Race Data", "Model Performance", "Predictions"])
    
    with tab1:
        try:
            recent_races = db_manager.get_race_data(limit=20)
            if recent_races:
                df = pd.DataFrame(recent_races)
                display_cols = ['season', 'race_name', 'driver', 'team', 'position', 'points', 'created_at']
                available_cols = [col for col in display_cols if col in df.columns]
                if available_cols:
                    st.dataframe(df[available_cols].head(10), use_container_width=True)
                else:
                    st.write("Race data structure differs from expected format")
            else:
                st.info("No recent race data available")
        except Exception as e:
            st.error(f"Error loading recent race data: {str(e)}")
    
    with tab2:
        try:
            model_history = db_manager.get_model_performance_history()
            if not model_history.empty:
                display_cols = ['model_name', 'test_score', 'cv_score_mean', 'training_data_size', 'created_at']
                st.dataframe(model_history[display_cols].head(10), use_container_width=True)
            else:
                st.info("No model performance history available")
        except Exception as e:
            st.error(f"Error loading model performance history: {str(e)}")
    
    with tab3:
        try:
            predictions = db_manager._load_json(db_manager.predictions_file)
            
            if predictions:
                # Sort by created_at and take latest 10
                sorted_predictions = sorted(predictions, 
                                          key=lambda x: x.get('created_at', ''), 
                                          reverse=True)[:10]
                
                pred_data = []
                for pred in sorted_predictions:
                    pred_data.append({
                        'race_name': pred.get('race_name', 'Unknown'),
                        'driver': pred.get('driver', 'Unknown'),
                        'predicted_position': pred.get('predicted_position', 0),
                        'confidence': pred.get('confidence', 0),
                        'model_used': pred.get('model_used', 'Unknown'),
                        'created_at': pred.get('created_at', 'Unknown')
                    })
                
                st.dataframe(pd.DataFrame(pred_data), use_container_width=True)
            else:
                st.info("No recent predictions available")
        except Exception as e:
            st.error(f"Error loading recent predictions: {str(e)}")

elif page == "Driver Analytics":
    st.header("🏎️ Driver Analytics & Performance")
    st.info("📊 Detailed analysis of individual driver performance, statistics, and race results.")
    
    # Get available data
    try:
        race_data = db_manager.get_race_data()
        if not race_data:
            # Fall back to cached data if database is empty
            cached_data = load_cached_data('processed_data')
            race_data = cached_data if cached_data else []
        
        if race_data:
            df = pd.DataFrame(race_data)
            
            # Driver selection
            col1, col2 = st.columns([1, 1])
            
            with col1:
                drivers = sorted(df['driver'].unique())
                selected_driver = st.selectbox("Select Driver", drivers)
            
            with col2:
                seasons = sorted(df['season'].unique())
                selected_seasons = st.multiselect(
                    "Select Seasons", 
                    seasons, 
                    default=seasons[-2:] if len(seasons) >= 2 else seasons
                )
            
            if selected_driver and selected_seasons:
                # Filter data for selected driver and seasons
                driver_data = df[
                    (df['driver'] == selected_driver) & 
                    (df['season'].isin(selected_seasons))
                ]
                
                if not driver_data.empty:
                    # Calculate comprehensive stats using analytics
                    driver_stats = analytics.calculate_driver_stats(driver_data)
                    basic_stats = driver_stats.get('basic_stats', {})
                    consistency = driver_stats.get('consistency_metrics', {})
                    
                    # Driver Profile Section
                    st.subheader(f"🏆 {selected_driver} Profile")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_races = basic_stats.get('total_races', len(driver_data))
                        st.metric("Total Races", total_races)
                    
                    with col2:
                        wins = basic_stats.get('wins', len(driver_data[driver_data['position'] == 1]))
                        st.metric("Wins", wins)
                    
                    with col3:
                        podiums = basic_stats.get('podiums', len(driver_data[driver_data['position'] <= 3]))
                        st.metric("Podiums", podiums)
                    
                    with col4:
                        total_points = basic_stats.get('total_points', driver_data['points'].sum())
                        st.metric("Total Points", f"{total_points:.0f}")
                    
                    # Performance Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        avg_position = basic_stats.get('average_position', driver_data['position'].mean())
                        st.metric("Average Position", f"{avg_position:.1f}")
                    
                    with col2:
                        consistency_score = consistency.get('consistency_score', 0)
                        st.metric("Consistency Score", f"{consistency_score:.2f}")
                    
                    with col3:
                        points_per_race = basic_stats.get('points_per_race', driver_data['points'].mean())
                        st.metric("Points per Race", f"{points_per_race:.1f}")
                    
                    with col4:
                        if 'reliability' in driver_stats:
                            dnf_rate = driver_stats['reliability'].get('dnf_rate', 0)
                            st.metric("DNF Rate", f"{dnf_rate:.1f}%")
                        else:
                            st.metric("DNF Rate", "N/A")
                    
                    # Performance Analysis Tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["Race Results", "Performance Trends", "Team Analysis", "Track Performance"])
                    
                    with tab1:
                        st.subheader("Race Results")
                        
                        # Recent race results
                        display_cols = ['season', 'race_name', 'position', 'grid_position', 'points', 'team']
                        if 'fastest_lap' in driver_data.columns:
                            display_cols.append('fastest_lap')
                        
                        available_cols = [col for col in display_cols if col in driver_data.columns]
                        
                        # Sort by season and race
                        race_results = driver_data[available_cols].sort_values(['season', 'race_name'], ascending=[False, True])
                        st.dataframe(race_results, use_container_width=True)
                        
                        # Position distribution
                        st.subheader("Position Distribution")
                        position_counts = driver_data['position'].value_counts().sort_index()
                        
                        fig = px.bar(
                            x=position_counts.index,
                            y=position_counts.values,
                            title=f"{selected_driver} - Finishing Position Distribution",
                            labels={'x': 'Finishing Position', 'y': 'Number of Races'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        st.subheader("Performance Trends")
                        
                        # Points trend over time
                        if len(selected_seasons) > 1:
                            season_stats = driver_data.groupby('season').agg({
                                'points': 'sum',
                                'position': 'mean',
                                'race_name': 'count'
                            }).reset_index()
                            season_stats.columns = ['Season', 'Total Points', 'Average Position', 'Races']
                            
                            fig = px.line(
                                season_stats,
                                x='Season',
                                y='Total Points',
                                title=f"{selected_driver} - Points by Season",
                                markers=True
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Average position trend
                            fig2 = px.line(
                                season_stats,
                                x='Season',
                                y='Average Position',
                                title=f"{selected_driver} - Average Position by Season",
                                markers=True
                            )
                            fig2.update_layout(yaxis=dict(autorange="reversed"))  # Lower position numbers are better
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Race-by-race performance
                        if 'race_name' in driver_data.columns:
                            race_performance = driver_data[['race_name', 'season', 'position', 'points']].copy()
                            race_performance['race_label'] = race_performance['season'].astype(str) + ' - ' + race_performance['race_name']
                            
                            fig3 = px.scatter(
                                race_performance,
                                x=range(len(race_performance)),
                                y='position',
                                color='points',
                                title=f"{selected_driver} - Race by Race Performance",
                                labels={'x': 'Race Number', 'y': 'Finishing Position'},
                                hover_data=['race_label', 'points']
                            )
                            fig3.update_yaxis(autorange="reversed")
                            st.plotly_chart(fig3, use_container_width=True)
                    
                    with tab3:
                        st.subheader("Team Analysis")
                        
                        # Performance by team
                        if 'team' in driver_data.columns:
                            team_stats = driver_data.groupby('team').agg({
                                'points': ['sum', 'mean'],
                                'position': 'mean',
                                'race_name': 'count'
                            }).round(2)
                            
                            team_stats.columns = ['Total Points', 'Points per Race', 'Average Position', 'Races']
                            team_stats = team_stats.reset_index()
                            
                            st.write(f"**{selected_driver} Performance by Team:**")
                            st.dataframe(team_stats, use_container_width=True)
                            
                            # Team performance visualization
                            if len(team_stats) > 1:
                                fig = px.bar(
                                    team_stats,
                                    x='team',
                                    y='Points per Race',
                                    title=f"{selected_driver} - Points per Race by Team",
                                    color='Points per Race'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with tab4:
                        st.subheader("Track Performance")
                        
                        # Performance by circuit
                        if 'circuit' in driver_data.columns:
                            track_stats = driver_data.groupby('circuit').agg({
                                'points': ['sum', 'mean'],
                                'position': 'mean',
                                'race_name': 'count'
                            }).round(2)
                            
                            track_stats.columns = ['Total Points', 'Points per Race', 'Average Position', 'Races']
                            track_stats = track_stats.reset_index()
                            track_stats = track_stats.sort_values('Points per Race', ascending=False)
                            
                            st.write(f"**{selected_driver} Performance by Circuit:**")
                            st.dataframe(track_stats.head(10), use_container_width=True)
                            
                            # Best and worst tracks
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Best Tracks (by points per race):**")
                                best_tracks = track_stats.head(5)[['circuit', 'Points per Race', 'Average Position']]
                                st.dataframe(best_tracks, use_container_width=True)
                            
                            with col2:
                                st.write("**Most Challenging Tracks:**")
                                worst_tracks = track_stats.tail(5)[['circuit', 'Points per Race', 'Average Position']]
                                st.dataframe(worst_tracks, use_container_width=True)
                    
                    # Advanced Analytics Section
                    st.subheader("📈 Advanced Analytics")
                    
                    analytics_tabs = st.tabs(["Driver Comparison", "Teammate Analysis", "Season Progression", "Detailed Stats"])
                    
                    with analytics_tabs[0]:
                        st.write("**Compare with another driver:**")
                        comparison_driver = st.selectbox(
                            "Select driver to compare",
                            options=['None'] + [d for d in drivers if d != selected_driver]
                        )
                        
                        if comparison_driver and comparison_driver != 'None':
                            comp_data = df[
                                (df['driver'] == comparison_driver) & 
                                (df['season'].isin(selected_seasons))
                            ]
                            
                            if not comp_data.empty:
                                # Use analytics engine for detailed comparison
                                comparison = analytics.compare_drivers(driver_data, comp_data)
                                
                                if 'head_to_head' in comparison:
                                    h2h = comparison['head_to_head']
                                    
                                    # Create comparison metrics
                                    driver1_stats = comparison['driver1_stats']['basic_stats']
                                    driver2_stats = comparison['driver2_stats']['basic_stats']
                                    
                                    metrics_df = pd.DataFrame({
                                        selected_driver: [
                                            driver1_stats.get('total_races', 0),
                                            driver1_stats.get('wins', 0),
                                            driver1_stats.get('podiums', 0),
                                            driver1_stats.get('total_points', 0),
                                            driver1_stats.get('average_position', 0),
                                            driver1_stats.get('points_per_race', 0)
                                        ],
                                        comparison_driver: [
                                            driver2_stats.get('total_races', 0),
                                            driver2_stats.get('wins', 0),
                                            driver2_stats.get('podiums', 0),
                                            driver2_stats.get('total_points', 0),
                                            driver2_stats.get('average_position', 0),
                                            driver2_stats.get('points_per_race', 0)
                                        ]
                                    }, index=['Total Races', 'Wins', 'Podiums', 'Total Points', 'Avg Position', 'Points/Race'])
                                    
                                    st.dataframe(metrics_df.round(2), use_container_width=True)
                                    
                                    # Radar chart comparison
                                    fig = go.Figure()
                                    
                                    categories = ['Wins', 'Podiums', 'Points/Race', 'Consistency']
                                    
                                    # Normalize values for radar chart
                                    driver1_values = [
                                        driver1_stats.get('wins', 0) / max(1, driver1_stats.get('total_races', 1)),
                                        driver1_stats.get('podiums', 0) / max(1, driver1_stats.get('total_races', 1)),
                                        driver1_stats.get('points_per_race', 0) / 25,  # Normalize to max F1 points
                                        comparison['driver1_stats']['consistency_metrics'].get('consistency_score', 0)
                                    ]
                                    
                                    driver2_values = [
                                        driver2_stats.get('wins', 0) / max(1, driver2_stats.get('total_races', 1)),
                                        driver2_stats.get('podiums', 0) / max(1, driver2_stats.get('total_races', 1)),
                                        driver2_stats.get('points_per_race', 0) / 25,
                                        comparison['driver2_stats']['consistency_metrics'].get('consistency_score', 0)
                                    ]
                                    
                                    fig.add_trace(go.Scatterpolar(
                                        r=driver1_values,
                                        theta=categories,
                                        fill='toself',
                                        name=selected_driver
                                    ))
                                    
                                    fig.add_trace(go.Scatterpolar(
                                        r=driver2_values,
                                        theta=categories,
                                        fill='toself',
                                        name=comparison_driver
                                    ))
                                    
                                    fig.update_layout(
                                        polar=dict(
                                            radialaxis=dict(
                                                visible=True,
                                                range=[0, 1]
                                            )),
                                        showlegend=True,
                                        title="Driver Performance Comparison"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    with analytics_tabs[1]:
                        st.write("**Teammate Analysis:**")
                        teammate_comparison = analytics.get_teammate_comparison(df, selected_driver)
                        
                        if teammate_comparison:
                            for teammate_info, comparison_data in teammate_comparison.items():
                                st.write(f"**vs {teammate_info}:**")
                                
                                if 'head_to_head' in comparison_data:
                                    h2h = comparison_data['head_to_head']
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        wins_diff = h2h.get('wins_difference', 0)
                                        st.metric("Wins Advantage", f"{wins_diff:+.0f}")
                                    with col2:
                                        points_diff = h2h.get('total_points_difference', 0)
                                        st.metric("Points Advantage", f"{points_diff:+.0f}")
                                    with col3:
                                        pos_advantage = h2h.get('position_advantage', 0)
                                        st.metric("Position Advantage", f"{pos_advantage:+.1f}")
                        else:
                            st.info("No teammate data available for comparison")
                    
                    with analytics_tabs[2]:
                        st.write("**Season Progression Analysis:**")
                        progression = analytics.analyze_season_progression(driver_data)
                        
                        if progression:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                trend_direction = "📈 Improving" if progression.get('improving', False) else "📉 Declining"
                                st.metric("Performance Trend", trend_direction)
                                
                                early_points = progression.get('early_season_avg_points', 0)
                                late_points = progression.get('late_season_avg_points', 0)
                                st.metric("Early Season Avg Points", f"{early_points:.1f}")
                                st.metric("Late Season Avg Points", f"{late_points:.1f}")
                            
                            with col2:
                                points_trend = progression.get('points_trend_slope', 0)
                                st.metric("Points Trend Slope", f"{points_trend:+.2f}")
                                
                                position_trend = progression.get('position_trend_slope', 0)
                                st.metric("Position Trend Slope", f"{position_trend:+.2f}")
                        else:
                            st.info("Insufficient data for season progression analysis")
                    
                    with analytics_tabs[3]:
                        st.write("**Detailed Statistics:**")
                        
                        if 'qualifying_analysis' in driver_stats:
                            qual_stats = driver_stats['qualifying_analysis']
                            st.write("**Qualifying vs Race Performance:**")
                            
                            qual_col1, qual_col2, qual_col3 = st.columns(3)
                            with qual_col1:
                                avg_grid = qual_stats.get('avg_grid_position', 0)
                                st.metric("Average Grid Position", f"{avg_grid:.1f}")
                            with qual_col2:
                                avg_race_pos = qual_stats.get('avg_race_position', 0)
                                st.metric("Average Race Position", f"{avg_race_pos:.1f}")
                            with qual_col3:
                                avg_gain = qual_stats.get('avg_position_gain', 0)
                                st.metric("Average Position Change", f"{avg_gain:+.1f}")
                        
                        # Show all detailed stats in expandable sections
                        if driver_stats:
                            with st.expander("View All Statistics"):
                                st.json(driver_stats)
                
                else:
                    st.warning(f"No data found for {selected_driver} in selected seasons.")
            
            else:
                st.info("Please select a driver and at least one season to view analytics.")
        
        else:
            st.warning("No race data available. Please collect data first using the Data Collection page.")
    
    except Exception as e:
        st.error(f"Error loading driver analytics: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "🏎️ F1 Race Predictor | Built with Streamlit and FastF1 | "
    "Data from Formula 1 official timing systems"
)
