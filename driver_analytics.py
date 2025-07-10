import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

class DriverAnalytics:
    """Analytics engine for driver performance analysis"""
    
    def __init__(self):
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for driver analytics"""
        logger = logging.getLogger('DriverAnalytics')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def calculate_driver_stats(self, driver_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive driver statistics"""
        if driver_data.empty:
            return {}
        
        stats = {
            'basic_stats': {
                'total_races': len(driver_data),
                'wins': len(driver_data[driver_data['position'] == 1]),
                'podiums': len(driver_data[driver_data['position'] <= 3]),
                'top_5': len(driver_data[driver_data['position'] <= 5]),
                'top_10': len(driver_data[driver_data['position'] <= 10]),
                'total_points': driver_data['points'].sum(),
                'average_position': driver_data['position'].mean(),
                'points_per_race': driver_data['points'].mean(),
                'best_finish': driver_data['position'].min(),
                'worst_finish': driver_data['position'].max()
            },
            'consistency_metrics': {
                'position_std': driver_data['position'].std(),
                'points_std': driver_data['points'].std(),
                'consistency_score': self._calculate_consistency_score(driver_data)
            },
            'season_stats': self._calculate_season_stats(driver_data),
            'track_performance': self._calculate_track_performance(driver_data),
            'team_performance': self._calculate_team_performance(driver_data)
        }
        
        # Add qualifying vs race performance if grid data available
        if 'grid_position' in driver_data.columns:
            stats['qualifying_analysis'] = self._analyze_qualifying_performance(driver_data)
        
        # Add DNF analysis if available
        if 'dnf' in driver_data.columns:
            stats['reliability'] = self._analyze_reliability(driver_data)
        
        return stats
    
    def _calculate_consistency_score(self, driver_data: pd.DataFrame) -> float:
        """Calculate a consistency score (lower variance = higher consistency)"""
        if len(driver_data) < 2:
            return 0.0
        
        # Normalize positions to 0-1 scale
        max_position = driver_data['position'].max()
        normalized_positions = (max_position - driver_data['position']) / max_position
        
        # Calculate coefficient of variation (inverted for consistency)
        mean_performance = normalized_positions.mean()
        std_performance = normalized_positions.std()
        
        if mean_performance == 0:
            return 0.0
        
        consistency = 1 - (std_performance / mean_performance)
        return max(0, min(1, consistency))  # Clamp between 0 and 1
    
    def _calculate_season_stats(self, driver_data: pd.DataFrame) -> Dict:
        """Calculate statistics by season"""
        if 'season' not in driver_data.columns:
            return {}
        
        season_stats = driver_data.groupby('season').agg({
            'points': ['sum', 'mean'],
            'position': ['mean', 'min', 'max'],
            'race_name': 'count'
        }).round(2)
        
        season_stats.columns = ['total_points', 'avg_points', 'avg_position', 'best_position', 'worst_position', 'races']
        return season_stats.to_dict('index')
    
    def _calculate_track_performance(self, driver_data: pd.DataFrame) -> Dict:
        """Calculate performance by track/circuit"""
        if 'circuit' not in driver_data.columns:
            return {}
        
        track_stats = driver_data.groupby('circuit').agg({
            'points': ['sum', 'mean', 'count'],
            'position': ['mean', 'min']
        }).round(2)
        
        track_stats.columns = ['total_points', 'avg_points', 'appearances', 'avg_position', 'best_position']
        return track_stats.to_dict('index')
    
    def _calculate_team_performance(self, driver_data: pd.DataFrame) -> Dict:
        """Calculate performance by team"""
        if 'team' not in driver_data.columns:
            return {}
        
        team_stats = driver_data.groupby('team').agg({
            'points': ['sum', 'mean', 'count'],
            'position': ['mean', 'min']
        }).round(2)
        
        team_stats.columns = ['total_points', 'avg_points', 'races', 'avg_position', 'best_position']
        return team_stats.to_dict('index')
    
    def _analyze_qualifying_performance(self, driver_data: pd.DataFrame) -> Dict:
        """Analyze qualifying vs race performance"""
        valid_data = driver_data.dropna(subset=['grid_position', 'position'])
        
        if valid_data.empty:
            return {}
        
        # Calculate position changes
        valid_data = valid_data.copy()
        valid_data['position_change'] = valid_data['grid_position'] - valid_data['position']
        
        return {
            'avg_grid_position': valid_data['grid_position'].mean(),
            'avg_race_position': valid_data['position'].mean(),
            'avg_position_gain': valid_data['position_change'].mean(),
            'best_position_gain': valid_data['position_change'].max(),
            'worst_position_loss': valid_data['position_change'].min(),
            'races_gained_positions': len(valid_data[valid_data['position_change'] > 0]),
            'races_lost_positions': len(valid_data[valid_data['position_change'] < 0]),
            'qualifying_vs_race_correlation': valid_data['grid_position'].corr(valid_data['position'])
        }
    
    def _analyze_reliability(self, driver_data: pd.DataFrame) -> Dict:
        """Analyze DNF patterns and reliability"""
        total_races = len(driver_data)
        dnfs = driver_data['dnf'].sum() if 'dnf' in driver_data.columns else 0
        
        return {
            'total_dnfs': dnfs,
            'dnf_rate': (dnfs / total_races) * 100 if total_races > 0 else 0,
            'finishing_rate': ((total_races - dnfs) / total_races) * 100 if total_races > 0 else 0,
            'points_scoring_rate': (len(driver_data[driver_data['points'] > 0]) / total_races) * 100 if total_races > 0 else 0
        }
    
    def compare_drivers(self, driver1_data: pd.DataFrame, driver2_data: pd.DataFrame) -> Dict:
        """Compare two drivers head-to-head"""
        stats1 = self.calculate_driver_stats(driver1_data)
        stats2 = self.calculate_driver_stats(driver2_data)
        
        comparison = {
            'driver1_stats': stats1,
            'driver2_stats': stats2,
            'head_to_head': self._head_to_head_comparison(stats1, stats2)
        }
        
        return comparison
    
    def _head_to_head_comparison(self, stats1: Dict, stats2: Dict) -> Dict:
        """Create head-to-head comparison metrics"""
        if not stats1 or not stats2:
            return {}
        
        basic1 = stats1.get('basic_stats', {})
        basic2 = stats2.get('basic_stats', {})
        
        comparison = {}
        
        for metric in ['total_races', 'wins', 'podiums', 'total_points', 'points_per_race']:
            if metric in basic1 and metric in basic2:
                comparison[f'{metric}_difference'] = basic1[metric] - basic2[metric]
                comparison[f'{metric}_ratio'] = basic1[metric] / basic2[metric] if basic2[metric] != 0 else float('inf')
        
        # Average position comparison (lower is better)
        if 'average_position' in basic1 and 'average_position' in basic2:
            comparison['position_advantage'] = basic2['average_position'] - basic1['average_position']
        
        return comparison
    
    def get_driver_rankings(self, all_data: pd.DataFrame, season: Optional[int] = None) -> pd.DataFrame:
        """Get driver rankings for a season or overall"""
        if season:
            data = all_data[all_data['season'] == season]
        else:
            data = all_data
        
        # Calculate rankings
        rankings = data.groupby('driver').agg({
            'points': 'sum',
            'position': 'mean',
            'race_name': 'count'
        }).round(2)
        
        rankings.columns = ['total_points', 'avg_position', 'races']
        rankings = rankings.sort_values('total_points', ascending=False)
        rankings['rank'] = range(1, len(rankings) + 1)
        
        return rankings.reset_index()
    
    def analyze_season_progression(self, driver_data: pd.DataFrame) -> Dict:
        """Analyze how driver performance evolved during a season"""
        if 'race_name' not in driver_data.columns or len(driver_data) < 3:
            return {}
        
        # Sort by season and race order (approximate)
        sorted_data = driver_data.sort_values(['season', 'race_name'])
        
        # Calculate rolling averages
        window_size = min(5, len(sorted_data) // 2)
        sorted_data = sorted_data.copy()
        sorted_data['rolling_points'] = sorted_data['points'].rolling(window=window_size, min_periods=1).mean()
        sorted_data['rolling_position'] = sorted_data['position'].rolling(window=window_size, min_periods=1).mean()
        
        # Trend analysis
        races = range(len(sorted_data))
        points_trend = np.polyfit(races, sorted_data['points'], 1)[0]  # Slope of points trend
        position_trend = np.polyfit(races, sorted_data['position'], 1)[0]  # Slope of position trend
        
        return {
            'points_trend_slope': points_trend,
            'position_trend_slope': position_trend,
            'improving': points_trend > 0 or position_trend < 0,  # Positive points or negative position slope
            'early_season_avg_points': sorted_data['points'].head(window_size).mean(),
            'late_season_avg_points': sorted_data['points'].tail(window_size).mean(),
            'early_season_avg_position': sorted_data['position'].head(window_size).mean(),
            'late_season_avg_position': sorted_data['position'].tail(window_size).mean()
        }
    
    def get_teammate_comparison(self, all_data: pd.DataFrame, driver: str, season: Optional[int] = None) -> Dict:
        """Compare driver with teammates"""
        driver_data = all_data[all_data['driver'] == driver]
        
        if season:
            driver_data = driver_data[driver_data['season'] == season]
        
        if driver_data.empty or 'team' not in driver_data.columns:
            return {}
        
        # Get teams the driver raced for
        teams = driver_data['team'].unique()
        teammate_comparisons = {}
        
        for team in teams:
            # Get all drivers from the same team in the same races
            team_races = driver_data[driver_data['team'] == team]['race_name'].unique()
            team_data = all_data[
                (all_data['team'] == team) & 
                (all_data['race_name'].isin(team_races))
            ]
            
            if season:
                team_data = team_data[team_data['season'] == season]
            
            teammates = team_data['driver'].unique()
            teammates = [t for t in teammates if t != driver]
            
            for teammate in teammates:
                teammate_data = team_data[team_data['driver'] == teammate]
                driver_team_data = team_data[team_data['driver'] == driver]
                
                if not teammate_data.empty and not driver_team_data.empty:
                    comparison = self.compare_drivers(driver_team_data, teammate_data)
                    teammate_comparisons[f"{teammate} ({team})"] = comparison
        
        return teammate_comparisons