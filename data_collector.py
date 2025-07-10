import fastf1
import pandas as pd
import numpy as np 
from datetime import datetime
import requests 
import time 
import logging 
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")
import os 
if not os.path.exists('cache'):
    os.makedirs('cache')
fastf1.Cache.enable_cache('cache')

class F1DataCollector:
    """Collects F1 data from FastF1 and other sources"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.weather_api_key = None  # Will be set from environment if available
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for data collection"""
        logger = logging.getLogger('F1DataCollector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def collect_season_data(self, season: int, include_weather: bool = True, 
                          include_telemetry: bool = True) -> List[Dict]:
        """Collect complete season data including race results, qualifying, and telemetry"""
        self.logger.info(f"Starting data collection for {season} season")
        
        try:
            # Get season schedule
            schedule = fastf1.get_event_schedule(season)
            season_data = []
            
            for _, event in schedule.iterrows():
                if pd.isna(event['Session5Date']):  # Skip events without race data
                    continue
                    
                try:
                    event_data = self._collect_event_data(
                        season, event, include_weather, include_telemetry
                    )
                    if event_data:
                        season_data.extend(event_data)
                    
                    # Rate limiting to avoid API overload
                    time.sleep(2)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to collect data for {event['EventName']}: {str(e)}")
                    continue
            
            self.logger.info(f"Collected data for {len(season_data)} driver-race combinations in {season}")
            return season_data
            
        except Exception as e:
            self.logger.error(f"Error collecting season {season} data: {str(e)}")
            raise
    
    def _collect_event_data(self, season: int, event: pd.Series, 
                           include_weather: bool, include_telemetry: bool) -> List[Dict]:
        """Collect data for a single race event"""
        event_name = event['EventName']
        self.logger.info(f"Collecting data for {event_name}")
        
        try:
            # Load race session
            race = fastf1.get_session(season, event_name, 'R')
            race.load()
            
            # Load qualifying session
            try:
                qualifying = fastf1.get_session(season, event_name, 'Q')
                qualifying.load()
            except:
                qualifying = None
                self.logger.warning(f"No qualifying data available for {event_name}")
            
            # Get race results
            race_results = race.results
            if race_results.empty:
                self.logger.warning(f"No race results available for {event_name}")
                return []
            
            event_data = []
            
            for _, driver_result in race_results.iterrows():
                try:
                    driver_data = self._extract_driver_race_data(
                        race, qualifying, driver_result, event, 
                        include_weather, include_telemetry
                    )
                    if driver_data:
                        event_data.append(driver_data)
                        
                except Exception as e:
                    driver_name = driver_result.get('Abbreviation', driver_result.get('DriverNumber', 'Unknown'))
                    self.logger.debug(f"Error extracting data for driver {driver_name}: {str(e)}")
                    continue
            
            return event_data
            
        except Exception as e:
            self.logger.error(f"Error collecting event data for {event_name}: {str(e)}")
            return []
    
    def _extract_driver_race_data(self, race, qualifying, driver_result: pd.Series, 
                                 event: pd.Series, include_weather: bool, 
                                 include_telemetry: bool) -> Optional[Dict]:
        """Extract comprehensive data for a single driver's race performance"""
        
        try:
            # Handle missing Abbreviation field
            driver_abbr = driver_result.get('Abbreviation', driver_result.get('DriverNumber', 'UNK'))
            
            # Basic race information - handle missing fields gracefully
            data = {
                'season': int(race.event.year),
                'race_name': str(event['EventName']),
                'round': int(event['RoundNumber']),
                'date': self._convert_to_naive_datetime(event['Session5Date']),
                'circuit': str(event['Location']),
                'country': str(event['Country']),
                'driver': str(driver_result.get('FullName', 'Unknown')),
                'driver_abbr': str(driver_abbr),
                'team': str(driver_result.get('TeamName', 'Unknown')),
                'position': int(driver_result.get('Position', 0)) if pd.notna(driver_result.get('Position', 0)) else 0,
                'grid_position': int(driver_result.get('GridPosition', 0)) if pd.notna(driver_result.get('GridPosition', 0)) else 0,
                'points': float(driver_result.get('Points', 0)) if pd.notna(driver_result.get('Points', 0)) else 0.0,
                'status': str(driver_result.get('Status', 'Unknown')),
                'time': str(driver_result['Time']) if 'Time' in driver_result and pd.notna(driver_result['Time']) else None,
                'fastest_lap': bool(driver_result.get('FastestLap', False)),
                'fastest_lap_time': str(driver_result['FastestLapTime']) if 'FastestLapTime' in driver_result and pd.notna(driver_result['FastestLapTime']) else None,
            }
            
            # Qualifying data
            if qualifying is not None:
                try:
                    qual_results = qualifying.results
                    # Handle both Abbreviation and DriverNumber for matching
                    if 'Abbreviation' in qual_results.columns:
                        driver_qual = qual_results[qual_results['Abbreviation'] == driver_abbr]
                    elif 'DriverNumber' in qual_results.columns:
                        driver_qual = qual_results[qual_results['DriverNumber'] == driver_abbr]
                    else:
                        driver_qual = pd.DataFrame()  # Empty dataframe if no matching column
                    if not driver_qual.empty:
                        qual_row = driver_qual.iloc[0]
                        data.update({
                            'qual_position': int(qual_row.get('Position', 0)) if pd.notna(qual_row.get('Position', 0)) else 0,
                            'q1_time': str(qual_row['Q1']) if 'Q1' in qual_row and pd.notna(qual_row['Q1']) else None,
                            'q2_time': str(qual_row['Q2']) if 'Q2' in qual_row and pd.notna(qual_row['Q2']) else None,
                            'q3_time': str(qual_row['Q3']) if 'Q3' in qual_row and pd.notna(qual_row['Q3']) else None,
                        })
                except Exception as e:
                    self.logger.debug(f"Error getting qualifying data for {driver_abbr}: {str(e)}")
                    pass
            
            # Weather data
            if include_weather:
                weather_data = self._get_weather_data(race, event)
                data.update(weather_data)
            
            # Telemetry data
            if include_telemetry:
                telemetry_data = self._get_telemetry_summary(race, driver_abbr)
                data.update(telemetry_data)
            
            # Track-specific data
            track_data = self._get_track_characteristics(event['Location'])
            data.update(track_data)
            
            # Driver/Team historical performance
            historical_data = self._get_historical_performance(
                race.event.year, driver_result['FullName'], driver_result['TeamName']
            )
            data.update(historical_data)
            
            return data
            
        except Exception as e:
            # Only log as debug to reduce noise in logs
            self.logger.debug(f"Error extracting driver data: {str(e)}")
            return None
    
    def _get_weather_data(self, race, event: pd.Series) -> Dict:
        """Extract weather information for the race"""
        try:
            # Try to get weather from FastF1 first
            weather_data = {}
            
            if hasattr(race, 'weather_data') and race.weather_data is not None:
                weather = race.weather_data
                if not weather.empty:
                    # Average weather conditions during race
                    weather_data = {
                        'air_temp': weather['AirTemp'].mean(),
                        'track_temp': weather['TrackTemp'].mean(),
                        'humidity': weather['Humidity'].mean(),
                        'pressure': weather['Pressure'].mean(),
                        'wind_speed': weather['WindSpeed'].mean(),
                        'rainfall': weather['Rainfall'].any(),
                    }
            
            # If no FastF1 weather data, provide defaults
            if not weather_data:
                weather_data = {
                    'air_temp': 25.0,  # Default reasonable values
                    'track_temp': 35.0,
                    'humidity': 60.0,
                    'pressure': 1013.0,
                    'wind_speed': 5.0,
                    'rainfall': False,
                }
                
            return weather_data
            
        except Exception as e:
            self.logger.warning(f"Error getting weather data: {str(e)}")
            return {
                'air_temp': 25.0,
                'track_temp': 35.0,
                'humidity': 60.0,
                'pressure': 1013.0,
                'wind_speed': 5.0,
                'rainfall': False,
            }
    
    def _get_telemetry_summary(self, race, driver_abbr: str) -> Dict:
        """Extract telemetry summary statistics for a driver"""
        try:
            # Get driver's laps
            driver_laps = race.laps.pick_driver(driver_abbr)
            
            if driver_laps.empty:
                return self._default_telemetry_data()
            
            # Calculate telemetry statistics
            telemetry_data = {
                'avg_lap_time': driver_laps['LapTime'].mean().total_seconds() if not driver_laps['LapTime'].isna().all() else 90.0,
                'best_lap_time': driver_laps['LapTime'].min().total_seconds() if not driver_laps['LapTime'].isna().all() else 85.0,
                'total_laps': len(driver_laps),
                'pit_stops': len(driver_laps[driver_laps['PitOutTime'].notna()]),
                'dnf': 'DNF' in str(driver_laps['TrackStatus'].iloc[-1]) if not driver_laps.empty else False,
            }
            
            # Sector times analysis
            if not driver_laps['Sector1Time'].isna().all():
                telemetry_data.update({
                    'avg_sector1': driver_laps['Sector1Time'].mean().total_seconds(),
                    'avg_sector2': driver_laps['Sector2Time'].mean().total_seconds(),
                    'avg_sector3': driver_laps['Sector3Time'].mean().total_seconds(),
                })
            
            # Tire strategy
            tire_compounds = driver_laps['Compound'].dropna().unique()
            telemetry_data['tire_compounds_used'] = len(tire_compounds)
            telemetry_data['primary_compound'] = tire_compounds[0] if len(tire_compounds) > 0 else 'MEDIUM'
            
            return telemetry_data
            
        except Exception as e:
            self.logger.warning(f"Error getting telemetry data for {driver_abbr}: {str(e)}")
            return self._default_telemetry_data()
    
    def _default_telemetry_data(self) -> Dict:
        """Return default telemetry data when actual data is unavailable"""
        return {
            'avg_lap_time': 90.0,
            'best_lap_time': 85.0,
            'total_laps': 50,
            'pit_stops': 2,
            'dnf': False,
            'avg_sector1': 25.0,
            'avg_sector2': 35.0,
            'avg_sector3': 30.0,
            'tire_compounds_used': 2,
            'primary_compound': 'MEDIUM'
        }
    
    def _get_track_characteristics(self, location: str) -> Dict:
        """Get track-specific characteristics"""
        # Track characteristics database (simplified)
        track_data = {
            'Bahrain': {'length': 5.412, 'corners': 15, 'drs_zones': 3, 'elevation_change': 32, 'overtaking_difficulty': 'medium'},
            'Saudi Arabia': {'length': 6.174, 'corners': 27, 'drs_zones': 3, 'elevation_change': 108, 'overtaking_difficulty': 'medium'},
            'Australia': {'length': 5.278, 'corners': 16, 'drs_zones': 2, 'elevation_change': 15, 'overtaking_difficulty': 'medium'},
            'Monaco': {'length': 3.337, 'corners': 19, 'drs_zones': 1, 'elevation_change': 42, 'overtaking_difficulty': 'very_hard'},
            'Spain': {'length': 4.675, 'corners': 16, 'drs_zones': 2, 'elevation_change': 31, 'overtaking_difficulty': 'hard'},
            'United Kingdom': {'length': 5.891, 'corners': 18, 'drs_zones': 2, 'elevation_change': 15, 'overtaking_difficulty': 'medium'},
        }
        
        default_track = {'length': 5.0, 'corners': 15, 'drs_zones': 2, 'elevation_change': 30, 'overtaking_difficulty': 'medium'}
        track_info = track_data.get(location, default_track)
        
        # Convert to numerical values for ML
        difficulty_map = {'very_easy': 1, 'easy': 2, 'medium': 3, 'hard': 4, 'very_hard': 5}
        track_info['overtaking_difficulty_score'] = difficulty_map[track_info['overtaking_difficulty']]
        
        return {
            'track_length': track_info['length'],
            'track_corners': track_info['corners'],
            'track_drs_zones': track_info['drs_zones'],
            'track_elevation_change': track_info['elevation_change'],
            'track_overtaking_difficulty': track_info['overtaking_difficulty_score']
        }
    
    def _get_historical_performance(self, current_season: int, driver_name: str, team_name: str) -> Dict:
        """Get historical performance metrics for driver and team"""
        # This would ideally query a database of historical performance
        # For now, we'll provide reasonable defaults based on known performance patterns
        
        # Simplified driver performance mapping (this would be more sophisticated in production)
        driver_ratings = {
            'Max Verstappen': {'skill_rating': 95, 'consistency': 92, 'wet_weather': 88},
            'Lewis Hamilton': {'skill_rating': 93, 'consistency': 89, 'wet_weather': 95},
            'Charles Leclerc': {'skill_rating': 90, 'consistency': 85, 'wet_weather': 87},
            'George Russell': {'skill_rating': 87, 'consistency': 88, 'wet_weather': 85},
            'Carlos Sainz': {'skill_rating': 86, 'consistency': 87, 'wet_weather': 83},
            'Lando Norris': {'skill_rating': 85, 'consistency': 86, 'wet_weather': 82},
        }
        
        team_ratings = {
            'Red Bull Racing': {'car_performance': 95, 'strategy': 90, 'pit_stop_avg': 2.3},
            'Mercedes': {'car_performance': 85, 'strategy': 88, 'pit_stop_avg': 2.5},
            'Ferrari': {'car_performance': 88, 'strategy': 82, 'pit_stop_avg': 2.7},
            'McLaren': {'car_performance': 82, 'strategy': 85, 'pit_stop_avg': 2.6},
            'Aston Martin': {'car_performance': 80, 'strategy': 83, 'pit_stop_avg': 2.8},
        }
        
        # Default values for unknown drivers/teams
        default_driver = {'skill_rating': 75, 'consistency': 75, 'wet_weather': 75}
        default_team = {'car_performance': 75, 'strategy': 75, 'pit_stop_avg': 3.0}
        
        driver_perf = driver_ratings.get(driver_name, default_driver)
        team_perf = team_ratings.get(team_name, default_team)
        
        return {
            'driver_skill_rating': driver_perf['skill_rating'],
            'driver_consistency': driver_perf['consistency'],
            'driver_wet_weather': driver_perf['wet_weather'],
            'team_car_performance': team_perf['car_performance'],
            'team_strategy_rating': team_perf['strategy'],
            'team_pit_stop_avg': team_perf['pit_stop_avg'],
            'driver_championship_position': 10,  # Would be calculated from actual data
            'team_championship_position': 5,     # Would be calculated from actual data
            'recent_form': 75,  # Average of last 5 races - would be calculated
        }
    
    def get_current_drivers_teams(self, season: int = 2025) -> Tuple[List[str], List[str]]:
        """Get current season drivers and teams"""
        # This would ideally be fetched from current F1 data
        # For 2025, we'll use expected lineups
        drivers_2025 = [
            'Max Verstappen', 'Sergio Perez', 'Lewis Hamilton', 'George Russell',
            'Charles Leclerc', 'Carlos Sainz', 'Lando Norris', 'Oscar Piastri',
            'Fernando Alonso', 'Lance Stroll', 'Esteban Ocon', 'Pierre Gasly',
            'Alexander Albon', 'Logan Sargeant', 'Valtteri Bottas', 'Zhou Guanyu',
            'Kevin Magnussen', 'Nico Hulkenberg', 'Yuki Tsunoda', 'Daniel Ricciardo'
        ]
        
        teams_2025 = [
            'Red Bull Racing', 'Mercedes', 'Ferrari', 'McLaren',
            'Aston Martin', 'Alpine', 'Williams', 'Alfa Romeo',
            'Haas', 'AlphaTauri'
        ]
        
        return drivers_2025, teams_2025
    
    def _convert_to_naive_datetime(self, dt_value):
        """Convert timezone-aware datetime to naive datetime for database storage"""
        try:
            if pd.isna(dt_value) or dt_value is None:
                return datetime.now()
            
            # Convert to pandas datetime with UTC timezone handling
            if isinstance(dt_value, str):
                dt = pd.to_datetime(dt_value, utc=True)
            else:
                dt = pd.to_datetime(dt_value, utc=True)
            
            # If timezone-aware, convert to UTC then remove timezone info
            if hasattr(dt, 'tz') and dt.tz is not None:
                dt = dt.tz_convert('UTC').tz_localize(None)
            
            # Convert to Python datetime object
            if hasattr(dt, 'to_pydatetime'):
                return dt.to_pydatetime()
            else:
                return dt
            
        except Exception as e:
            self.logger.warning(f"Error converting datetime {dt_value}: {str(e)}")
            return datetime.now()
