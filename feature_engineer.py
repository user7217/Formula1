import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Feature engineering for F1 race prediction"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        
    def process_all_data(self, all_season_data: dict) -> list:
        """Process and engineer features for all collected data"""
        # Combine all season data
        combined_data = []
        for season, season_data in all_season_data.items():
            if isinstance(season_data, list):
                combined_data.extend(season_data)
            else:
                print(f"Warning: Unexpected data type for season {season}: {type(season_data)}")
        
        if not combined_data:
            raise ValueError("No data provided for feature engineering")
        
        # Convert to DataFrame with error handling
        try:
            df = pd.DataFrame(combined_data)
            print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        except Exception as e:
            print(f"Error creating DataFrame: {str(e)}")
            # Check data consistency
            if combined_data:
                print(f"Sample data keys: {list(combined_data[0].keys()) if combined_data else 'None'}")
                print(f"First few data points have these key counts: {[len(d.keys()) for d in combined_data[:5]]}")
            raise
        
        try:
            # Engineer features step by step
            print("Engineering basic features...")
            df = self._engineer_basic_features(df)
            
            print("Engineering performance features...")
            df = self._engineer_performance_features(df)
            
            print("Engineering track features...")
            df = self._engineer_track_features(df)
            
            print("Engineering temporal features...")
            df = self._engineer_temporal_features(df)
            
            print("Engineering competitive features...")
            df = self._engineer_competitive_features(df)
            
            print("Handling missing values...")
            df = self._handle_missing_values(df)
            
            print(f"Final DataFrame shape: {df.shape}")
            
            # Convert back to list of dictionaries
            return df.to_dict('records')
            
        except Exception as e:
            print(f"Error during feature engineering: {str(e)}")
            print(f"DataFrame columns: {list(df.columns) if 'df' in locals() else 'DataFrame not created'}")
            raise
    
    def _engineer_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer basic race features"""
        # Convert position to numeric, handling DNF cases
        df['position_numeric'] = pd.to_numeric(df['position'], errors='coerce')
        df['finished_race'] = ~df['position_numeric'].isna()
        df['points_scored'] = df['points'].fillna(0)
        
        # Podium finish indicator
        df['podium_finish'] = (df['position_numeric'] <= 3) & (df['finished_race'])
        df['points_finish'] = (df['position_numeric'] <= 10) & (df['finished_race'])
        
        # Grid position analysis
        df['grid_position_numeric'] = pd.to_numeric(df['grid_position'], errors='coerce')
        df['grid_to_finish_change'] = df['grid_position_numeric'] - df['position_numeric']
        df['improved_from_grid'] = df['grid_to_finish_change'] > 0
        
        # Time-based features
        df['fastest_lap_achieved'] = df['fastest_lap'].notna()
        
        # Convert lap times to seconds for analysis
        df = self._convert_time_columns(df)
        
        return df
    
    def _convert_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert time columns to seconds for numerical analysis"""
        time_columns = ['time', 'fastest_lap_time', 'q1_time', 'q2_time', 'q3_time']
        
        for col in time_columns:
            if col in df.columns:
                df[f'{col}_seconds'] = df[col].apply(self._time_to_seconds)
        
        return df
    
    def _time_to_seconds(self, time_str) -> float:
        """Convert time string to seconds"""
        if pd.isna(time_str) or time_str is None:
            return np.nan
        
        try:
            time_str = str(time_str)
            if ':' in time_str:
                # Format: MM:SS.mmm or H:MM:SS.mmm
                parts = time_str.split(':')
                if len(parts) == 2:
                    minutes, seconds = parts
                    return float(minutes) * 60 + float(seconds)
                elif len(parts) == 3:
                    hours, minutes, seconds = parts
                    return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
            else:
                # Assume it's already in seconds
                return float(time_str)
        except:
            return np.nan
    
    def _engineer_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer driver and team performance features"""
        # Driver performance metrics
        df['driver_season_avg_position'] = df.groupby(['season', 'driver'])['position_numeric'].transform('mean')
        df['driver_season_points'] = df.groupby(['season', 'driver'])['points'].transform('sum')
        df['driver_season_podiums'] = df.groupby(['season', 'driver'])['podium_finish'].transform('sum')
        
        # Team performance metrics
        df['team_season_avg_position'] = df.groupby(['season', 'team'])['position_numeric'].transform('mean')
        df['team_season_points'] = df.groupby(['season', 'team'])['points'].transform('sum')
        
        # Recent form (last 3 races)
        df = df.sort_values(['driver', 'date'])
        df['driver_recent_avg_position'] = df.groupby('driver')['position_numeric'].rolling(
            window=3, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        # Driver-track combination history
        df['driver_track_avg_position'] = df.groupby(['driver', 'circuit'])['position_numeric'].transform('mean')
        df['driver_track_experience'] = df.groupby(['driver', 'circuit']).cumcount() + 1
        
        # Qualifying performance relative to race
        df['qual_to_race_change'] = df['qual_position'] - df['position_numeric']
        df['outperformed_qualifying'] = df['qual_to_race_change'] > 0
        
        return df
    
    def _engineer_track_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer track-specific features"""
        # Track performance indicators
        df['track_suits_driver'] = df['driver_track_avg_position'] < df['driver_season_avg_position']
        df['track_suits_team'] = df.groupby(['team', 'circuit'])['position_numeric'].transform('mean') < df['team_season_avg_position']
        
        # Weather impact features
        df['hot_conditions'] = df['air_temp'] > 30
        df['cold_conditions'] = df['air_temp'] < 15
        df['high_track_temp'] = df['track_temp'] > 45
        df['wet_conditions'] = df['rainfall'] == True
        
        # Track characteristics impact
        df['power_sensitive_track'] = df['track_length'] > 5.5
        df['technical_track'] = df['track_corners'] > 16
        df['overtaking_friendly'] = df['track_overtaking_difficulty'] <= 2
        
        return df
    
    def _engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer time-based features"""
        # Convert date to datetime with proper timezone handling
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
        
        # Remove timezone info if present to avoid conversion issues
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        
        # Season progress
        df['season_round_normalized'] = df['round'] / df.groupby('season')['round'].transform('max')
        df['early_season'] = df['season_round_normalized'] < 0.3
        df['mid_season'] = (df['season_round_normalized'] >= 0.3) & (df['season_round_normalized'] < 0.7)
        df['late_season'] = df['season_round_normalized'] >= 0.7
        
        # Day of year (for seasonal effects)
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        
        # Championship phase
        df['championship_deciding_phase'] = df['season_round_normalized'] > 0.8
        
        return df
    
    def _engineer_competitive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer competitive dynamics features"""
        # Championship standings simulation (simplified)
        df = df.sort_values(['season', 'date', 'position_numeric'])
        
        # Points gaps and standings
        df['season_points_running'] = df.groupby(['season', 'driver'])['points'].cumsum()
        df['championship_leader_points'] = df.groupby(['season', 'date'])['season_points_running'].transform('max')
        df['points_behind_leader'] = df['championship_leader_points'] - df['season_points_running']
        
        # Team dynamics
        df['teammate_position'] = df.groupby(['season', 'date', 'team'])['position_numeric'].transform(
            lambda x: x.iloc[1] if len(x) > 1 else np.nan
        )
        df['beating_teammate'] = df['position_numeric'] < df['teammate_position']
        
        # Grid performance relative to team
        df['team_avg_grid'] = df.groupby(['season', 'date', 'team'])['grid_position_numeric'].transform('mean')
        df['grid_vs_team_avg'] = df['grid_position_numeric'] - df['team_avg_grid']
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        try:
            # Simple fill strategy that preserves all columns
            print(f"Handling missing values for DataFrame with {df.shape[1]} columns")
            
            # Fill numeric columns with median, categorical with mode
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    # Numeric columns - fill with median or 0 if all NaN
                    if df[col].notna().any():
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(0)
                else:
                    # Categorical columns - fill with mode or 'Unknown'
                    if df[col].notna().any():
                        mode_val = df[col].mode()
                        fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                        df[col] = df[col].fillna(fill_val)
                    else:
                        df[col] = df[col].fillna('Unknown')
            
            print(f"Successfully handled missing values, final shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"Error in _handle_missing_values: {str(e)}")
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {list(df.columns)}")
            # Fallback: simple fill
            return df.fillna({'object': 'Unknown', 'bool': False}).fillna(0)
    
    def prepare_training_data(self, df: pd.DataFrame, target_column: str = 'position_numeric'):
        """Prepare data for machine learning training"""
        # Select feature columns (exclude target and identifier columns)
        exclude_columns = [
            'driver', 'team', 'race_name', 'date', 'circuit', 'country',
            'position', 'position_numeric', 'status', 'time', 'fastest_lap_time',
            'q1_time', 'q2_time', 'q3_time', 'driver_abbr', 'fastest_lap',
            'points', 'season_points_running', 'championship_leader_points'
        ]
        
        if target_column == 'points':
            exclude_columns.remove('points')
        elif target_column == 'podium_finish':
            exclude_columns.append('points_finish')
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        self.feature_columns = feature_columns
        
        # Prepare features
        X = df[feature_columns].copy()
        
        # Encode categorical variables
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns
        for col in categorical_features:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.encoders[col].transform(X[col].astype(str))
        
        # Scale numerical features
        numerical_features = X.select_dtypes(include=[np.number]).columns
        if 'scaler' not in self.scalers:
            self.scalers['scaler'] = StandardScaler()
            X[numerical_features] = self.scalers['scaler'].fit_transform(X[numerical_features])
        else:
            X[numerical_features] = self.scalers['scaler'].transform(X[numerical_features])
        
        # Prepare target - map user-friendly names to actual column names
        if target_column == 'position' or target_column == 'position_numeric':
            y = df['position_numeric'].copy()
        elif target_column == 'points':
            y = df['points'].copy()
        elif target_column == 'podium_finish':
            y = df['podium_finish'].astype(int).copy()
        else:
            raise ValueError(f"Unknown target column: {target_column}. Available options: 'position', 'points', 'podium_finish'")
        
        return X, y
    
    def prepare_prediction_data(self, race_data: dict) -> pd.DataFrame:
        """Prepare data for prediction (single race)"""
        # Convert to DataFrame
        df = pd.DataFrame([race_data])
        
        # Apply same feature engineering steps
        df = self._engineer_basic_features(df)
        df = self._engineer_performance_features(df)
        df = self._engineer_track_features(df)
        df = self._engineer_temporal_features(df)
        df = self._engineer_competitive_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Select same feature columns used in training
        if self.feature_columns:
            # Ensure all required columns exist
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0  # Default value for missing features
            
            X = df[self.feature_columns].copy()
            
            # Apply same encodings and scaling
            categorical_features = X.select_dtypes(include=['object', 'bool']).columns
            for col in categorical_features:
                if col in self.encoders:
                    try:
                        X[col] = self.encoders[col].transform(X[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        X[col] = 0
            
            numerical_features = X.select_dtypes(include=[np.number]).columns
            if 'scaler' in self.scalers:
                X[numerical_features] = self.scalers['scaler'].transform(X[numerical_features])
            
            return X
        else:
            raise ValueError("No feature columns defined. Please train model first.")
    
    def get_feature_importance_names(self) -> list:
        """Get the names of features used for training"""
        return self.feature_columns if self.feature_columns else []