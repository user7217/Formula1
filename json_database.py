import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

class JSONDatabaseManager:
    """Manages JSON-based database operations for F1 race predictor"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Define JSON file paths
        self.race_data_file = self.data_dir / "race_data.json"
        self.model_results_file = self.data_dir / "model_results.json"
        self.predictions_file = self.data_dir / "predictions.json"
        self.validation_results_file = self.data_dir / "validation_results.json"
        
        self.logger = self._setup_logger()
        
        # Initialize files if they don't exist
        self._initialize_files()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for database operations"""
        logger = logging.getLogger('JSONDatabaseManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _initialize_files(self):
        """Initialize JSON files with empty structures"""
        files_to_init = [
            (self.race_data_file, []),
            (self.model_results_file, {}),
            (self.predictions_file, []),
            (self.validation_results_file, [])
        ]
        
        for file_path, default_data in files_to_init:
            if not file_path.exists():
                self._save_json(file_path, default_data)
    
    def _load_json(self, file_path: Path) -> Any:
        """Load data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Error loading {file_path}: {str(e)}")
            return [] if 'data' in str(file_path) or 'predictions' in str(file_path) or 'validation' in str(file_path) else {}
    
    def _save_json(self, file_path: Path, data: Any):
        """Save data to JSON file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving {file_path}: {str(e)}")
            raise
    
    def create_tables(self) -> bool:
        """Create JSON database structure (compatibility method)"""
        try:
            self._initialize_files()
            self.logger.info("JSON database structure initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error creating JSON database structure: {str(e)}")
            return False
    
    def ensure_database_exists(self) -> bool:
        """Ensure all database files exist and are properly initialized"""
        try:
            # Check if data directory exists
            if not self.data_dir.exists():
                self.logger.info(f"Creating data directory: {self.data_dir}")
                self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Check and create each database file
            files_status = {}
            files_to_check = [
                (self.race_data_file, [], "race data"),
                (self.model_results_file, {}, "model results"),
                (self.predictions_file, [], "predictions"),
                (self.validation_results_file, [], "validation results")
            ]
            
            for file_path, default_data, description in files_to_check:
                if not file_path.exists():
                    self.logger.info(f"Creating {description} database file: {file_path}")
                    self._save_json(file_path, default_data)
                    files_status[description] = "created"
                else:
                    # Verify file is readable
                    try:
                        self._load_json(file_path)
                        files_status[description] = "exists"
                    except Exception as e:
                        self.logger.warning(f"Corrupted {description} file, recreating: {str(e)}")
                        self._save_json(file_path, default_data)
                        files_status[description] = "recreated"
            
            # Log status
            self.logger.info(f"Database status: {files_status}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error ensuring database exists: {str(e)}")
            return False
    
    def check_database_health(self) -> Dict[str, Any]:
        """Check the health and status of the JSON database"""
        health_status = {
            "status": "healthy",
            "issues": [],
            "files": {},
            "total_size_mb": 0,
            "recommendations": []
        }
        
        try:
            # Check each file
            files_to_check = [
                (self.race_data_file, "race_data"),
                (self.model_results_file, "model_results"),
                (self.predictions_file, "predictions"),
                (self.validation_results_file, "validation_results")
            ]
            
            for file_path, file_type in files_to_check:
                file_info = {
                    "exists": file_path.exists(),
                    "size_mb": 0,
                    "records": 0,
                    "readable": False
                }
                
                if file_path.exists():
                    # Get file size
                    file_info["size_mb"] = round(file_path.stat().st_size / (1024 * 1024), 2)
                    health_status["total_size_mb"] += file_info["size_mb"]
                    
                    # Check if readable
                    try:
                        data = self._load_json(file_path)
                        file_info["readable"] = True
                        
                        # Count records
                        if isinstance(data, list):
                            file_info["records"] = len(data)
                        elif isinstance(data, dict):
                            file_info["records"] = len(data.keys())
                        
                    except Exception as e:
                        file_info["readable"] = False
                        health_status["issues"].append(f"{file_type} file is corrupted: {str(e)}")
                        health_status["status"] = "warning"
                else:
                    health_status["issues"].append(f"{file_type} file does not exist")
                    health_status["status"] = "warning"
                
                health_status["files"][file_type] = file_info
            
            # Add recommendations
            if health_status["total_size_mb"] > 100:
                health_status["recommendations"].append("Consider cleaning up old data - database is over 100MB")
            
            if len(health_status["issues"]) > 0:
                health_status["recommendations"].append("Run ensure_database_exists() to fix missing files")
            
            return health_status
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["issues"].append(f"Health check failed: {str(e)}")
            return health_status
    
    def save_race_data(self, race_data_list: List[Dict]) -> bool:
        """Save race data to JSON database for quick future access"""
        try:
            # Load existing data
            existing_data = self._load_json(self.race_data_file)
            
            # Convert to DataFrame for easier manipulation
            if existing_data:
                existing_df = pd.DataFrame(existing_data)
            else:
                existing_df = pd.DataFrame()
            
            new_df = pd.DataFrame(race_data_list)
            
            # Remove duplicates based on season, race_name, and driver
            if not existing_df.empty:
                # Merge and remove duplicates
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(
                    subset=['season', 'race_name', 'driver'], 
                    keep='last'
                )
            else:
                combined_df = new_df
            
            # Convert datetime objects to strings for JSON serialization
            for col in combined_df.columns:
                if combined_df[col].dtype == 'datetime64[ns]' or 'datetime' in str(combined_df[col].dtype):
                    # Handle timezone-aware datetimes
                    if hasattr(combined_df[col].dt, 'tz') and combined_df[col].dt.tz is not None:
                        combined_df[col] = combined_df[col].dt.tz_localize(None)
                    combined_df[col] = combined_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Save to JSON
            self._save_json(self.race_data_file, combined_df.to_dict('records'))
            
            self.logger.info(f"Saved {len(race_data_list)} race records to JSON database for quick future access")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving race data to JSON: {str(e)}")
            return False
    
    def get_race_data(self, season: Optional[int] = None, limit: Optional[int] = None) -> List[Dict]:
        """Retrieve race data from JSON database"""
        try:
            data = self._load_json(self.race_data_file)
            
            if not data:
                return []
            
            # Filter by season if specified
            if season is not None:
                data = [record for record in data if record.get('season') == season]
            
            # Apply limit if specified
            if limit is not None:
                data = data[-limit:]  # Get most recent records
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error retrieving race data from JSON: {str(e)}")
            return []
    
    def save_model_results(self, model_results: Dict) -> bool:
        """Save model training results to JSON database"""
        try:
            # Load existing results
            existing_results = self._load_json(self.model_results_file)
            
            # Add timestamp to new results
            for model_name, results in model_results.items():
                results['timestamp'] = datetime.now().isoformat()
                existing_results[model_name] = results
            
            # Save updated results
            self._save_json(self.model_results_file, existing_results)
            
            self.logger.info(f"Saved results for {len(model_results)} models to JSON database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model results to JSON: {str(e)}")
            return False
    
    def save_predictions(self, predictions: List[Dict], race_info: Dict, model_used: str) -> bool:
        """Save race predictions to JSON database"""
        try:
            # Load existing predictions
            existing_predictions = self._load_json(self.predictions_file)
            
            # Add metadata to predictions
            for pred in predictions:
                pred.update({
                    'season': race_info.get('season', 2025),
                    'race_name': race_info['race_name'],
                    'round_number': race_info.get('round', 1),
                    'model_used': model_used,
                    'created_at': datetime.now().isoformat()
                })
            
            # Add new predictions
            existing_predictions.extend(predictions)
            
            # Save updated predictions
            self._save_json(self.predictions_file, existing_predictions)
            
            self.logger.info(f"Saved {len(predictions)} predictions to JSON database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving predictions to JSON: {str(e)}")
            return False
    
    def save_validation_results(self, validation_data: Dict) -> bool:
        """Save validation results to JSON database"""
        try:
            # Load existing validation results
            existing_results = self._load_json(self.validation_results_file)
            
            # Add timestamp
            validation_data['created_at'] = datetime.now().isoformat()
            
            # Add new validation result
            existing_results.append(validation_data)
            
            # Save updated results
            self._save_json(self.validation_results_file, existing_results)
            
            self.logger.info("Saved validation results to JSON database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving validation results to JSON: {str(e)}")
            return False
    
    def get_model_performance_history(self) -> pd.DataFrame:
        """Get model performance history from JSON database"""
        try:
            model_results = self._load_json(self.model_results_file)
            
            if not model_results:
                return pd.DataFrame()
            
            # Convert to DataFrame
            records = []
            for model_name, results in model_results.items():
                record = results.copy()
                record['model_name'] = model_name
                records.append(record)
            
            return pd.DataFrame(records)
            
        except Exception as e:
            self.logger.error(f"Error getting model performance history: {str(e)}")
            return pd.DataFrame()
    
    def get_prediction_accuracy_summary(self) -> Dict:
        """Get overall prediction accuracy summary from JSON database"""
        try:
            predictions = self._load_json(self.predictions_file)
            validations = self._load_json(self.validation_results_file)
            
            if not predictions and not validations:
                return {}
            
            summary = {}
            
            # Calculate prediction metrics
            if predictions:
                accurate_predictions = [p for p in predictions if p.get('actual_position')]
                summary['total_predictions'] = len(predictions)
                summary['validated_predictions'] = len(accurate_predictions)
                
                if accurate_predictions:
                    position_errors = [abs(p['predicted_position'] - p['actual_position']) for p in accurate_predictions]
                    summary['mean_position_error'] = sum(position_errors) / len(position_errors)
                    summary['accuracy_within_1'] = sum(1 for e in position_errors if e <= 1) / len(position_errors)
                    summary['accuracy_within_3'] = sum(1 for e in position_errors if e <= 3) / len(position_errors)
            
            # Calculate validation metrics
            if validations:
                summary['total_validated_races'] = len(validations)
                if validations:
                    avg_pos_acc = sum(v.get('position_accuracy', 0) for v in validations) / len(validations)
                    avg_top3_acc = sum(v.get('top3_accuracy', 0) for v in validations) / len(validations)
                    avg_pos_err = sum(v.get('mean_position_error', 0) for v in validations) / len(validations)
                    
                    summary['average_position_accuracy'] = avg_pos_acc
                    summary['average_top3_accuracy'] = avg_top3_acc
                    summary['average_position_error'] = avg_pos_err
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting prediction accuracy summary: {str(e)}")
            return {}
    
    def cleanup_old_data(self, days_old: int = 30):
        """Clean up old prediction and validation data from JSON database"""
        try:
            cutoff_date = datetime.now() - pd.Timedelta(days=days_old)
            
            # Clean predictions
            predictions = self._load_json(self.predictions_file)
            filtered_predictions = [
                p for p in predictions 
                if 'created_at' not in p or pd.to_datetime(p['created_at']) > cutoff_date
            ]
            self._save_json(self.predictions_file, filtered_predictions)
            
            # Clean validation results
            validations = self._load_json(self.validation_results_file)
            filtered_validations = [
                v for v in validations 
                if 'created_at' not in v or pd.to_datetime(v['created_at']) > cutoff_date
            ]
            self._save_json(self.validation_results_file, filtered_validations)
            
            self.logger.info(f"Cleaned up data older than {days_old} days")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {str(e)}")
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the JSON database"""
        try:
            stats = {}
            
            # Race data stats
            race_data = self._load_json(self.race_data_file)
            stats['race_records'] = len(race_data)
            stats['race_data_size'] = f"{self.race_data_file.stat().st_size / 1024:.1f} KB"
            
            # Model results stats
            model_results = self._load_json(self.model_results_file)
            stats['trained_models'] = len(model_results)
            stats['model_results_size'] = f"{self.model_results_file.stat().st_size / 1024:.1f} KB"
            
            # Predictions stats
            predictions = self._load_json(self.predictions_file)
            stats['total_predictions'] = len(predictions)
            stats['predictions_size'] = f"{self.predictions_file.stat().st_size / 1024:.1f} KB"
            
            # Validation stats
            validations = self._load_json(self.validation_results_file)
            stats['validation_records'] = len(validations)
            stats['validation_size'] = f"{self.validation_results_file.stat().st_size / 1024:.1f} KB"
            
            # Total size
            total_size = sum([
                self.race_data_file.stat().st_size,
                self.model_results_file.stat().st_size,
                self.predictions_file.stat().st_size,
                self.validation_results_file.stat().st_size
            ])
            stats['total_database_size'] = f"{total_size / 1024:.1f} KB"
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting database stats: {str(e)}")
            return {}