import json
import os
import pickle
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Cache directory
CACHE_DIR = 'cache'
DATA_DIR = 'data'

def ensure_directories():
    """Ensure required directories exist"""
    for directory in [CACHE_DIR, DATA_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

def load_cached_data(cache_key: str) -> Optional[Any]:
    """Load data from cache"""
    ensure_directories()
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data
    except Exception as e:
        print(f"Error loading cached data for {cache_key}: {str(e)}")
    
    return None

def save_cached_data(cache_key: str, data: Any) -> bool:
    """Save data to cache"""
    ensure_directories()
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving cached data for {cache_key}: {str(e)}")
        return False

def get_cache_info() -> Dict[str, Any]:
    """Get information about cached data"""
    ensure_directories()
    cache_info = {}
    
    try:
        cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
        
        for cache_file in cache_files:
            cache_key = cache_file.replace('.pkl', '')
            file_path = os.path.join(CACHE_DIR, cache_file)
            
            # Get file stats
            stat = os.stat(file_path)
            cache_info[cache_key] = {
                'file_size': stat.st_size,
                'modified_time': datetime.fromtimestamp(stat.st_mtime),
                'exists': True
            }
    except Exception as e:
        print(f"Error getting cache info: {str(e)}")
    
    return cache_info

def clear_cache(cache_key: Optional[str] = None):
    """Clear cache data"""
    ensure_directories()
    
    try:
        if cache_key:
            # Clear specific cache
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)
        else:
            # Clear all cache
            cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')]
            for cache_file in cache_files:
                os.remove(os.path.join(CACHE_DIR, cache_file))
    except Exception as e:
        print(f"Error clearing cache: {str(e)}")

def get_current_season_schedule(season: int = 2025) -> List[Dict]:
    """Get F1 race schedule for specified season"""
    
    # 2025 F1 season schedule (projected)
    if season == 2025:
        schedule = [
            {'race_name': 'Bahrain Grand Prix', 'date': '2025-03-02', 'location': 'Bahrain', 'round': 1},
            {'race_name': 'Saudi Arabian Grand Prix', 'date': '2025-03-09', 'location': 'Saudi Arabia', 'round': 2},
            {'race_name': 'Australian Grand Prix', 'date': '2025-03-23', 'location': 'Australia', 'round': 3},
            {'race_name': 'Chinese Grand Prix', 'date': '2025-04-13', 'location': 'China', 'round': 4},
            {'race_name': 'Miami Grand Prix', 'date': '2025-05-04', 'location': 'United States', 'round': 5},
            {'race_name': 'Emilia Romagna Grand Prix', 'date': '2025-05-18', 'location': 'Italy', 'round': 6},
            {'race_name': 'Monaco Grand Prix', 'date': '2025-05-25', 'location': 'Monaco', 'round': 7},
            {'race_name': 'Spanish Grand Prix', 'date': '2025-06-01', 'location': 'Spain', 'round': 8},
            {'race_name': 'Canadian Grand Prix', 'date': '2025-06-15', 'location': 'Canada', 'round': 9},
            {'race_name': 'Austrian Grand Prix', 'date': '2025-06-29', 'location': 'Austria', 'round': 10},
            {'race_name': 'British Grand Prix', 'date': '2025-07-06', 'location': 'United Kingdom', 'round': 11},
            {'race_name': 'Hungarian Grand Prix', 'date': '2025-07-20', 'location': 'Hungary', 'round': 12},
            {'race_name': 'Belgian Grand Prix', 'date': '2025-08-31', 'location': 'Belgium', 'round': 13},
            {'race_name': 'Dutch Grand Prix', 'date': '2025-09-07', 'location': 'Netherlands', 'round': 14},
            {'race_name': 'Italian Grand Prix', 'date': '2025-09-14', 'location': 'Italy', 'round': 15},
            {'race_name': 'Singapore Grand Prix', 'date': '2025-09-21', 'location': 'Singapore', 'round': 16},
            {'race_name': 'United States Grand Prix', 'date': '2025-10-19', 'location': 'United States', 'round': 17},
            {'race_name': 'Mexican Grand Prix', 'date': '2025-10-26', 'location': 'Mexico', 'round': 18},
            {'race_name': 'Brazilian Grand Prix', 'date': '2025-11-09', 'location': 'Brazil', 'round': 19},
            {'race_name': 'Las Vegas Grand Prix', 'date': '2025-11-23', 'location': 'United States', 'round': 20},
            {'race_name': 'Qatar Grand Prix', 'date': '2025-11-30', 'location': 'Qatar', 'round': 21},
            {'race_name': 'Abu Dhabi Grand Prix', 'date': '2025-12-07', 'location': 'United Arab Emirates', 'round': 22}
        ]
    else:
        # For other seasons, would fetch from FastF1 or other source
        schedule = []
    
    return schedule

def calculate_season_progress(current_date: str = None) -> float:
    """Calculate how far through the current F1 season we are"""
    if current_date is None:
        current_date = datetime.now().strftime('%Y-%m-%d')
    
    current_dt = datetime.strptime(current_date, '%Y-%m-%d')
    season = current_dt.year
    
    schedule = get_current_season_schedule(season)
    if not schedule:
        return 0.0
    
    # Find completed races
    completed_races = 0
    for race in schedule:
        race_date = datetime.strptime(race['date'], '%Y-%m-%d')
        if race_date <= current_dt:
            completed_races += 1
    
    return completed_races / len(schedule)

def format_time_delta(seconds: float) -> str:
    """Format time difference in seconds to readable format"""
    if pd.isna(seconds) or seconds is None:
        return "N/A"
    
    if seconds < 60:
        return f"{seconds:.3f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}:{secs:06.3f}"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}:{minutes:02d}:{secs:06.3f}"

def create_data_hash(data: Any) -> str:
    """Create hash for data validation"""
    try:
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    except:
        return "unknown"

def validate_model_input(data: Dict) -> List[str]:
    """Validate input data for model prediction"""
    errors = []
    
    required_fields = [
        'driver', 'team', 'season', 'race_name', 'circuit',
        'air_temp', 'track_temp', 'grid_position'
    ]
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif data[field] is None:
            errors.append(f"Field {field} cannot be None")
    
    # Validate data types and ranges
    if 'season' in data and data['season']:
        if not isinstance(data['season'], int) or data['season'] < 2020 or data['season'] > 2030:
            errors.append("Season must be an integer between 2020 and 2030")
    
    if 'grid_position' in data and data['grid_position']:
        if not isinstance(data['grid_position'], (int, float)) or data['grid_position'] < 1 or data['grid_position'] > 20:
            errors.append("Grid position must be between 1 and 20")
    
    if 'air_temp' in data and data['air_temp']:
        if not isinstance(data['air_temp'], (int, float)) or data['air_temp'] < -10 or data['air_temp'] > 50:
            errors.append("Air temperature must be between -10 and 50 degrees Celsius")
    
    return errors

def export_predictions_to_csv(predictions: List[Dict], filename: str = None) -> str:
    """Export predictions to CSV file"""
    ensure_directories()
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"f1_predictions_{timestamp}.csv"
    
    filepath = os.path.join(DATA_DIR, filename)
    
    try:
        df = pd.DataFrame(predictions)
        df.to_csv(filepath, index=False)
        return filepath
    except Exception as e:
        print(f"Error exporting predictions: {str(e)}")
        return ""

def import_historical_results(filepath: str) -> Optional[List[Dict]]:
    """Import historical race results from CSV"""
    try:
        if not os.path.exists(filepath):
            return None
        
        df = pd.read_csv(filepath)
        return df.to_dict('records')
    except Exception as e:
        print(f"Error importing historical results: {str(e)}")
        return None

def get_model_performance_summary(model_results: Dict) -> Dict:
    """Generate summary of model performance"""
    if not model_results:
        return {}
    
    summary = {
        'total_models': len(model_results),
        'best_model': None,
        'best_score': 0,
        'average_score': 0,
        'model_scores': {}
    }
    
    scores = []
    for model_name, results in model_results.items():
        test_score = results.get('test_score', 0)
        scores.append(test_score)
        summary['model_scores'][model_name] = test_score
        
        if test_score > summary['best_score']:
            summary['best_score'] = test_score
            summary['best_model'] = model_name
    
    if scores:
        summary['average_score'] = sum(scores) / len(scores)
    
    return summary

def log_prediction_request(race_info: Dict, model_used: str, timestamp: str = None):
    """Log prediction request for analytics"""
    ensure_directories()
    
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    log_entry = {
        'timestamp': timestamp,
        'race_name': race_info.get('race_name', 'Unknown'),
        'race_date': race_info.get('date', 'Unknown'),
        'model_used': model_used,
        'request_hash': create_data_hash(race_info)
    }
    
    log_file = os.path.join(DATA_DIR, 'prediction_log.json')
    
    try:
        # Load existing log
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = []
        
        # Add new entry
        log_data.append(log_entry)
        
        # Save updated log
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
            
    except Exception as e:
        print(f"Error logging prediction request: {str(e)}")

def get_prediction_analytics() -> Dict:
    """Get analytics on prediction usage"""
    ensure_directories()
    log_file = os.path.join(DATA_DIR, 'prediction_log.json')
    
    analytics = {
        'total_predictions': 0,
        'most_predicted_race': None,
        'most_used_model': None,
        'recent_activity': []
    }
    
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            analytics['total_predictions'] = len(log_data)
            
            # Most predicted race
            race_counts = {}
            model_counts = {}
            
            for entry in log_data:
                race = entry.get('race_name', 'Unknown')
                model = entry.get('model_used', 'Unknown')
                
                race_counts[race] = race_counts.get(race, 0) + 1
                model_counts[model] = model_counts.get(model, 0) + 1
            
            if race_counts:
                analytics['most_predicted_race'] = max(race_counts, key=race_counts.get)
            
            if model_counts:
                analytics['most_used_model'] = max(model_counts, key=model_counts.get)
            
            # Recent activity (last 10)
            analytics['recent_activity'] = log_data[-10:] if len(log_data) > 10 else log_data
            
    except Exception as e:
        print(f"Error getting prediction analytics: {str(e)}")
    
    return analytics