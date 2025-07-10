import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import fastf1
from data_collector import F1DataCollector
from feature_engineer import FeatureEngineer
from ml_models import MLModelManager
from utils import load_cached_data, save_cached_data
import warnings
warnings.filterwarnings('ignore')

class RacePredictor:
    """Handles race predictions and validation"""
    
    def __init__(self):
        self.collector = F1DataCollector()
        self.engineer = FeatureEngineer()
        self.model_manager = MLModelManager()
        self.current_drivers, self.current_teams = self.collector.get_current_drivers_teams()
        
    def predict_race(self, race_info: Dict, model_choice: str = 'ensemble') -> List[Dict]:
        """Predict race results for a specific race"""
        
        # Load trained models
        trained_models = load_cached_data('trained_models')
        if trained_models:
            self.model_manager.models = trained_models
        else:
            raise ValueError("No trained models available. Please train models first.")
        
        # Generate predictions for all current drivers
        predictions = []
        
        for i, driver in enumerate(self.current_drivers):
            # Get driver's team (simplified mapping)
            team = self._get_driver_team(driver)
            
            # Create race data for this driver
            driver_race_data = self._create_race_data_for_driver(
                race_info, driver, team, i + 1
            )
            
            # Prepare data for prediction
            try:
                X = self.engineer.prepare_prediction_data(driver_race_data)
                
                # Make prediction
                if model_choice == 'ensemble':
                    ensemble_result = self.model_manager.predict_with_ensemble(X)
                    predicted_position = ensemble_result['ensemble_prediction'][0]
                    confidence = self._calculate_confidence(ensemble_result)
                else:
                    predicted_position = self.model_manager.predict_race_position(model_choice, X)[0]
                    confidence = 0.75  # Default confidence
                
                # Calculate predicted points
                predicted_points = self._position_to_points(predicted_position)
                
                predictions.append({
                    'driver': driver,
                    'team': team,
                    'predicted_position': max(1, min(20, round(predicted_position))),
                    'confidence': confidence,
                    'predicted_points': predicted_points,
                    'grid_estimate': i + 1  # Simplified grid position estimate
                })
                
            except Exception as e:
                # Fallback prediction
                predictions.append({
                    'driver': driver,
                    'team': team,
                    'predicted_position': i + 1,
                    'confidence': 0.5,
                    'predicted_points': self._position_to_points(i + 1),
                    'grid_estimate': i + 1
                })
        
        # Sort by predicted position
        predictions.sort(key=lambda x: x['predicted_position'])
        
        # Adjust for position consistency (ensure unique positions)
        self._adjust_predictions_for_consistency(predictions)
        
        return predictions
    
    def _get_driver_team(self, driver: str) -> str:
        """Get team for driver (simplified mapping)"""
        driver_team_mapping = {
            'Max Verstappen': 'Red Bull Racing',
            'Sergio Perez': 'Red Bull Racing',
            'Lewis Hamilton': 'Mercedes',
            'George Russell': 'Mercedes',
            'Charles Leclerc': 'Ferrari',
            'Carlos Sainz': 'Ferrari',
            'Lando Norris': 'McLaren',
            'Oscar Piastri': 'McLaren',
            'Fernando Alonso': 'Aston Martin',
            'Lance Stroll': 'Aston Martin',
            'Esteban Ocon': 'Alpine',
            'Pierre Gasly': 'Alpine',
            'Alexander Albon': 'Williams',
            'Logan Sargeant': 'Williams',
            'Valtteri Bottas': 'Alfa Romeo',
            'Zhou Guanyu': 'Alfa Romeo',
            'Kevin Magnussen': 'Haas',
            'Nico Hulkenberg': 'Haas',
            'Yuki Tsunoda': 'AlphaTauri',
            'Daniel Ricciardo': 'AlphaTauri'
        }
        
        return driver_team_mapping.get(driver, 'Unknown Team')
    
    def _create_race_data_for_driver(self, race_info: Dict, driver: str, team: str, grid_pos: int) -> Dict:
        """Create race data structure for prediction"""
        
        # Get historical performance data
        historical_perf = self.collector._get_historical_performance(2025, driver, team)
        
        # Get track characteristics
        track_chars = self.collector._get_track_characteristics(race_info.get('location', 'Unknown'))
        
        # Estimate weather conditions (simplified)
        weather_data = {
            'air_temp': 25.0,
            'track_temp': 35.0,
            'humidity': 60.0,
            'pressure': 1013.0,
            'wind_speed': 5.0,
            'rainfall': False
        }
        
        # Create comprehensive race data
        race_data = {
            'season': 2025,
            'race_name': race_info['race_name'],
            'round': race_info.get('round', 1),
            'date': race_info['date'],
            'circuit': race_info.get('circuit', race_info.get('location', 'Unknown')),
            'country': race_info.get('country', 'Unknown'),
            'driver': driver,
            'driver_abbr': driver[:3].upper(),
            'team': team,
            'grid_position': grid_pos,
            'qual_position': grid_pos,  # Simplified
            
            # Performance metrics
            'driver_skill_rating': historical_perf['driver_skill_rating'],
            'driver_consistency': historical_perf['driver_consistency'],
            'driver_wet_weather': historical_perf['driver_wet_weather'],
            'team_car_performance': historical_perf['team_car_performance'],
            'team_strategy_rating': historical_perf['team_strategy_rating'],
            'team_pit_stop_avg': historical_perf['team_pit_stop_avg'],
            
            # Track characteristics
            'track_length': track_chars['track_length'],
            'track_corners': track_chars['track_corners'],
            'track_drs_zones': track_chars['track_drs_zones'],
            'track_elevation_change': track_chars['track_elevation_change'],
            'track_overtaking_difficulty': track_chars['track_overtaking_difficulty'],
            
            # Weather
            'air_temp': weather_data['air_temp'],
            'track_temp': weather_data['track_temp'],
            'humidity': weather_data['humidity'],
            'pressure': weather_data['pressure'],
            'wind_speed': weather_data['wind_speed'],
            'rainfall': weather_data['rainfall'],
            
            # Telemetry estimates
            'avg_lap_time': 90.0,
            'best_lap_time': 85.0,
            'total_laps': 50,
            'pit_stops': 2,
            'dnf': False,
            'avg_sector1': 25.0,
            'avg_sector2': 35.0,
            'avg_sector3': 30.0,
            'tire_compounds_used': 2,
            'primary_compound': 'MEDIUM',
            
            # Seasonal context
            'driver_championship_position': historical_perf['driver_championship_position'],
            'team_championship_position': historical_perf['team_championship_position'],
            'recent_form': historical_perf['recent_form'],
            
            # Race specific
            'points': 0,  # To be predicted
            'position': None,  # To be predicted
            'fastest_lap': False,
            'status': 'Finished',
        }
        
        return race_data
    
    def _calculate_confidence(self, ensemble_result: Dict) -> float:
        """Calculate prediction confidence based on model agreement"""
        predictions = list(ensemble_result['individual_predictions'].values())
        
        if len(predictions) < 2:
            return 0.75
        
        # Calculate variance among predictions
        variance = np.var(predictions)
        
        # Convert variance to confidence (lower variance = higher confidence)
        max_variance = 25  # Maximum expected variance for positions
        confidence = max(0.5, 1.0 - (variance / max_variance))
        
        return min(0.95, confidence)
    
    def _position_to_points(self, position: float) -> int:
        """Convert finishing position to F1 points"""
        points_system = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        
        pos = round(position)
        return points_system.get(pos, 0)
    
    def _adjust_predictions_for_consistency(self, predictions: List[Dict]):
        """Ensure predictions have consistent, unique positions"""
        # Sort by predicted position
        predictions.sort(key=lambda x: x['predicted_position'])
        
        # Assign sequential positions
        for i, pred in enumerate(predictions, 1):
            pred['predicted_position'] = i
            pred['predicted_points'] = self._position_to_points(i)
    
    def validate_race_prediction(self, race_name: str, season: int = 2024) -> Dict:
        """Validate predictions against actual 2024 race results"""
        
        try:
            # Get actual race results from FastF1
            actual_results = self._get_actual_race_results(race_name, season)
            
            if not actual_results:
                return {'error': f'No actual results found for {race_name} {season}'}
            
            # Generate predictions for the same race
            race_info = {
                'race_name': race_name,
                'date': f'{season}-06-01',  # Simplified date
                'location': 'Unknown',
                'round': 1
            }
            
            predictions = self.predict_race(race_info, 'ensemble')
            
            # Compare predictions with actual results
            validation_results = self._compare_predictions_with_actual(predictions, actual_results)
            
            return {
                'race_name': race_name,
                'season': season,
                'predictions': predictions,
                'actual_results': actual_results,
                'accuracy_metrics': validation_results
            }
            
        except Exception as e:
            return {'error': f'Validation failed: {str(e)}'}
    
    def _get_actual_race_results(self, race_name: str, season: int) -> List[Dict]:
        """Get actual race results from FastF1"""
        try:
            # Load race session
            race = fastf1.get_session(season, race_name, 'R')
            race.load()
            
            results = []
            race_results = race.results
            
            for _, row in race_results.iterrows():
                results.append({
                    'driver': row['FullName'],
                    'position': row['Position'],
                    'points': row['Points'],
                    'team': row['TeamName'],
                    'status': row['Status']
                })
            
            return results
            
        except Exception as e:
            # Return empty results if data unavailable
            return []
    
    def _compare_predictions_with_actual(self, predictions: List[Dict], 
                                       actual_results: List[Dict]) -> Dict:
        """Compare predictions with actual results"""
        
        # Create lookup dictionaries
        pred_dict = {pred['driver']: pred for pred in predictions}
        actual_dict = {result['driver']: result for result in actual_results}
        
        # Find common drivers
        common_drivers = set(pred_dict.keys()) & set(actual_dict.keys())
        
        if not common_drivers:
            return {'error': 'No common drivers found between predictions and actual results'}
        
        # Calculate accuracy metrics
        position_errors = []
        correct_positions = 0
        correct_top3 = 0
        correct_points = 0
        
        for driver in common_drivers:
            pred = pred_dict[driver]
            actual = actual_dict[driver]
            
            # Position accuracy
            pred_pos = pred['predicted_position']
            actual_pos = actual['position']
            
            if pd.notna(actual_pos):
                position_error = abs(pred_pos - actual_pos)
                position_errors.append(position_error)
                
                # Exact position match
                if position_error == 0:
                    correct_positions += 1
                
                # Top 3 accuracy
                if (pred_pos <= 3 and actual_pos <= 3) or (pred_pos > 3 and actual_pos > 3):
                    correct_top3 += 1
                
                # Points finish accuracy
                pred_points = pred['predicted_points'] > 0
                actual_points = actual['points'] > 0
                if pred_points == actual_points:
                    correct_points += 1
        
        total_drivers = len(common_drivers)
        
        return {
            'total_drivers_compared': total_drivers,
            'position_accuracy': correct_positions / total_drivers if total_drivers > 0 else 0,
            'top3_accuracy': correct_top3 / total_drivers if total_drivers > 0 else 0,
            'points_accuracy': correct_points / total_drivers if total_drivers > 0 else 0,
            'mean_position_error': np.mean(position_errors) if position_errors else 0,
            'median_position_error': np.median(position_errors) if position_errors else 0,
        }
    
    def predict_championship_outcome(self, remaining_races: List[Dict]) -> Dict:
        """Predict championship outcome based on remaining races"""
        
        # Current championship standings (simplified)
        current_standings = self._get_current_championship_standings()
        
        # Predict each remaining race
        championship_projections = {}
        
        for driver in self.current_drivers:
            total_predicted_points = current_standings.get(driver, 0)
            
            for race in remaining_races:
                predictions = self.predict_race(race)
                driver_prediction = next((p for p in predictions if p['driver'] == driver), None)
                
                if driver_prediction:
                    total_predicted_points += driver_prediction['predicted_points']
            
            championship_projections[driver] = total_predicted_points
        
        # Sort by predicted total points
        sorted_championship = sorted(
            championship_projections.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            'projected_final_standings': sorted_championship,
            'championship_winner': sorted_championship[0][0] if sorted_championship else None,
            'winning_points': sorted_championship[0][1] if sorted_championship else 0,
        }
    
    def _get_current_championship_standings(self) -> Dict[str, int]:
        """Get current championship points (simplified)"""
        # This would ideally fetch real current standings
        # For demo purposes, provide realistic standings
        standings = {}
        for i, driver in enumerate(self.current_drivers):
            # Simulate current points based on typical season progression
            if i == 0:  # Championship leader
                standings[driver] = 350
            elif i < 3:  # Top contenders
                standings[driver] = 300 - (i * 25)
            elif i < 6:  # Midfield leaders
                standings[driver] = 200 - (i * 20)
            elif i < 10:  # Points scorers
                standings[driver] = 100 - (i * 10)
            else:  # Back markers
                standings[driver] = max(0, 50 - (i * 5))
        
        return standings