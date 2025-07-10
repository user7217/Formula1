import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')

# Configure engine with SSL and connection handling
connect_args = {}
if DATABASE_URL and 'postgres://' in DATABASE_URL:
    # Handle SSL connection issues
    connect_args = {
        "sslmode": "prefer",
        "connect_timeout": 10,
    }

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class RaceData(Base):
    __tablename__ = "race_data"
    
    id = Column(Integer, primary_key=True, index=True)
    season = Column(Integer, nullable=False)
    race_name = Column(String, nullable=False)
    round_number = Column(Integer, nullable=False)
    date = Column(DateTime, nullable=False)
    circuit = Column(String, nullable=False)
    country = Column(String, nullable=False)
    driver = Column(String, nullable=False)
    driver_abbr = Column(String, nullable=False)
    team = Column(String, nullable=False)
    position = Column(Integer, nullable=True)
    grid_position = Column(Integer, nullable=True)
    points = Column(Float, nullable=False, default=0.0)
    status = Column(String, nullable=False)
    fastest_lap = Column(Boolean, default=False)
    fastest_lap_time = Column(Float, nullable=True)
    qual_position = Column(Integer, nullable=True)
    q1_time = Column(Float, nullable=True)
    q2_time = Column(Float, nullable=True)
    q3_time = Column(Float, nullable=True)
    
    # Weather data
    air_temp = Column(Float, nullable=True)
    track_temp = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    pressure = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)
    rainfall = Column(Boolean, default=False)
    
    # Telemetry data
    avg_lap_time = Column(Float, nullable=True)
    best_lap_time = Column(Float, nullable=True)
    total_laps = Column(Integer, nullable=True)
    pit_stops = Column(Integer, nullable=True)
    dnf = Column(Boolean, default=False)
    avg_sector1 = Column(Float, nullable=True)
    avg_sector2 = Column(Float, nullable=True)
    avg_sector3 = Column(Float, nullable=True)
    tire_compounds_used = Column(Integer, nullable=True)
    primary_compound = Column(String, nullable=True)
    
    # Track characteristics
    track_length = Column(Float, nullable=True)
    track_corners = Column(Integer, nullable=True)
    track_drs_zones = Column(Integer, nullable=True)
    track_elevation_change = Column(Float, nullable=True)
    track_overtaking_difficulty = Column(Integer, nullable=True)
    
    # Performance metrics
    driver_skill_rating = Column(Float, nullable=True)
    driver_consistency = Column(Float, nullable=True)
    driver_wet_weather = Column(Float, nullable=True)
    team_car_performance = Column(Float, nullable=True)
    team_strategy_rating = Column(Float, nullable=True)
    team_pit_stop_avg = Column(Float, nullable=True)
    
    # Engineered features (stored as JSON)
    features = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelResults(Base):
    __tablename__ = "model_results"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # 'classification' or 'regression'
    train_score = Column(Float, nullable=False)
    test_score = Column(Float, nullable=False)
    cv_score_mean = Column(Float, nullable=False)
    cv_score_std = Column(Float, nullable=False)
    feature_importance = Column(JSON, nullable=True)
    hyperparameters = Column(JSON, nullable=True)
    training_data_size = Column(Integer, nullable=False)
    target_variable = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Predictions(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    season = Column(Integer, nullable=False)
    race_name = Column(String, nullable=False)
    round_number = Column(Integer, nullable=False)
    driver = Column(String, nullable=False)
    team = Column(String, nullable=False)
    model_used = Column(String, nullable=False)
    predicted_position = Column(Integer, nullable=False)
    predicted_points = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    actual_position = Column(Integer, nullable=True)
    actual_points = Column(Integer, nullable=True)
    prediction_accuracy = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ValidationResults(Base):
    __tablename__ = "validation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    race_name = Column(String, nullable=False)
    season = Column(Integer, nullable=False)
    model_used = Column(String, nullable=False)
    total_drivers_compared = Column(Integer, nullable=False)
    position_accuracy = Column(Float, nullable=False)
    top3_accuracy = Column(Float, nullable=False)
    points_accuracy = Column(Float, nullable=False)
    mean_position_error = Column(Float, nullable=False)
    median_position_error = Column(Float, nullable=False)
    validation_details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Manages database operations for F1 race predictor"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for database operations"""
        logger = logging.getLogger('DatabaseManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def create_tables(self):
        """Create all database tables"""
        try:
            # Test connection first
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Error creating database tables: {str(e)}")
            # Don't raise the error, just log it to avoid breaking the app
            return False
        return True
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def save_race_data(self, race_data_list: List[Dict]) -> bool:
        """Save race data to database for quick future access"""
        if not DATABASE_URL:
            self.logger.warning("No database URL configured, skipping save")
            return False
            
        session = self.get_session()
        saved_count = 0
        try:
            for data in race_data_list:
                # Convert datetime strings to datetime objects
                if 'date' in data and isinstance(data['date'], str):
                    data['date'] = pd.to_datetime(data['date'])
                
                # Extract features for JSON storage
                features = {}
                for key, value in data.items():
                    if key not in RaceData.__table__.columns.keys():
                        features[key] = value
                
                # Create race data record
                race_record = RaceData(
                    **{k: v for k, v in data.items() if k in RaceData.__table__.columns.keys()},
                    features=features if features else None
                )
                
                # Check if record already exists
                existing = session.query(RaceData).filter(
                    RaceData.season == data['season'],
                    RaceData.race_name == data['race_name'],
                    RaceData.driver == data['driver']
                ).first()
                
                if not existing:
                    session.add(race_record)
            
            session.commit()
            self.logger.info(f"Saved {len(race_data_list)} race data records to database")
            return True
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving race data: {str(e)}")
            return False
        finally:
            session.close()
    
    def save_model_results(self, model_results: Dict) -> bool:
        """Save model training results to database"""
        session = self.get_session()
        try:
            for model_name, results in model_results.items():
                model_record = ModelResults(
                    model_name=model_name,
                    model_type=results.get('model_type', 'regression'),
                    train_score=results.get('train_score', 0.0),
                    test_score=results.get('test_score', 0.0),
                    cv_score_mean=results.get('cv_score_mean', 0.0),
                    cv_score_std=results.get('cv_score_std', 0.0),
                    feature_importance=results.get('feature_importance', {}),
                    hyperparameters=results.get('hyperparameters', {}),
                    training_data_size=results.get('training_data_size', 0),
                    target_variable=results.get('target_variable', 'position')
                )
                session.add(model_record)
            
            session.commit()
            self.logger.info(f"Saved results for {len(model_results)} models to database")
            return True
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving model results: {str(e)}")
            return False
        finally:
            session.close()
    
    def save_predictions(self, predictions: List[Dict], race_info: Dict, model_used: str) -> bool:
        """Save race predictions to database"""
        session = self.get_session()
        try:
            for pred in predictions:
                prediction_record = Predictions(
                    season=race_info.get('season', 2025),
                    race_name=race_info['race_name'],
                    round_number=race_info.get('round', 1),
                    driver=pred['driver'],
                    team=pred['team'],
                    model_used=model_used,
                    predicted_position=pred['predicted_position'],
                    predicted_points=pred['predicted_points'],
                    confidence=pred['confidence']
                )
                session.add(prediction_record)
            
            session.commit()
            self.logger.info(f"Saved {len(predictions)} predictions to database")
            return True
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving predictions: {str(e)}")
            return False
        finally:
            session.close()
    
    def save_validation_results(self, validation_data: Dict) -> bool:
        """Save validation results to database"""
        session = self.get_session()
        try:
            if 'accuracy_metrics' in validation_data:
                metrics = validation_data['accuracy_metrics']
                validation_record = ValidationResults(
                    race_name=validation_data['race_name'],
                    season=validation_data['season'],
                    model_used=validation_data.get('model_used', 'ensemble'),
                    total_drivers_compared=metrics.get('total_drivers_compared', 0),
                    position_accuracy=metrics.get('position_accuracy', 0.0),
                    top3_accuracy=metrics.get('top3_accuracy', 0.0),
                    points_accuracy=metrics.get('points_accuracy', 0.0),
                    mean_position_error=metrics.get('mean_position_error', 0.0),
                    median_position_error=metrics.get('median_position_error', 0.0),
                    validation_details=validation_data
                )
                session.add(validation_record)
                session.commit()
                self.logger.info("Saved validation results to database")
                return True
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving validation results: {str(e)}")
            return False
        finally:
            session.close()
    
    def get_race_data(self, season: Optional[int] = None, limit: Optional[int] = None) -> List[Dict]:
        """Retrieve race data from database"""
        if not DATABASE_URL:
            self.logger.warning("No database URL configured")
            return []
            
        session = self.get_session()
        try:
            query = session.query(RaceData)
            
            if season:
                query = query.filter(RaceData.season == season)
            
            if limit:
                query = query.limit(limit)
            
            results = query.all()
            
            # Convert to dictionaries
            race_data = []
            for result in results:
                data = {column.name: getattr(result, column.name) for column in result.__table__.columns}
                # Add features from JSON field
                if result.features:
                    data.update(result.features)
                race_data.append(data)
            
            return race_data
            
        except Exception as e:
            self.logger.error(f"Error retrieving race data: {str(e)}")
            return []
        finally:
            session.close()
    
    def get_model_performance_history(self) -> pd.DataFrame:
        """Get model performance history"""
        session = self.get_session()
        try:
            results = session.query(ModelResults).order_by(ModelResults.created_at.desc()).all()
            
            data = []
            for result in results:
                data.append({
                    'model_name': result.model_name,
                    'model_type': result.model_type,
                    'train_score': result.train_score,
                    'test_score': result.test_score,
                    'cv_score_mean': result.cv_score_mean,
                    'cv_score_std': result.cv_score_std,
                    'training_data_size': result.training_data_size,
                    'target_variable': result.target_variable,
                    'created_at': result.created_at
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"Error retrieving model performance history: {str(e)}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def get_prediction_accuracy_summary(self) -> Dict:
        """Get overall prediction accuracy summary"""
        session = self.get_session()
        try:
            validation_results = session.query(ValidationResults).all()
            
            if not validation_results:
                return {}
            
            total_races = len(validation_results)
            avg_position_accuracy = sum(r.position_accuracy for r in validation_results) / total_races
            avg_top3_accuracy = sum(r.top3_accuracy for r in validation_results) / total_races
            avg_points_accuracy = sum(r.points_accuracy for r in validation_results) / total_races
            avg_position_error = sum(r.mean_position_error for r in validation_results) / total_races
            
            return {
                'total_validated_races': total_races,
                'average_position_accuracy': avg_position_accuracy,
                'average_top3_accuracy': avg_top3_accuracy,
                'average_points_accuracy': avg_points_accuracy,
                'average_position_error': avg_position_error,
                'last_validation': max(r.created_at for r in validation_results)
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving prediction accuracy summary: {str(e)}")
            return {}
        finally:
            session.close()
    
    def cleanup_old_data(self, days_old: int = 30):
        """Clean up old prediction and validation data"""
        session = self.get_session()
        try:
            cutoff_date = datetime.utcnow() - pd.Timedelta(days=days_old)
            
            # Delete old predictions
            old_predictions = session.query(Predictions).filter(
                Predictions.created_at < cutoff_date
            ).delete()
            
            # Delete old validation results
            old_validations = session.query(ValidationResults).filter(
                ValidationResults.created_at < cutoff_date
            ).delete()
            
            session.commit()
            self.logger.info(f"Cleaned up {old_predictions} old predictions and {old_validations} old validation results")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error cleaning up old data: {str(e)}")
        finally:
            session.close()

# Initialize database
db_manager = DatabaseManager()