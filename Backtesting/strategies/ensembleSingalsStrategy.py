from .base import Strategy
import numpy as np
import pandas as pd
from ..models.ensemble_model import EnsembleModel

class EnsembleSignalsStrategy(Strategy):
    '''
    Ensemble Strategy that combines multiple strategies to make trading decisions.'''
    
    def __init__(self, strategies=None, model_type='random_forest', training_period=60):
        """
        Initialize with a list of strategies and an ensemble ML model.
        
        Parameters:
        -----------
        strategies : list
            List of Strategy objects to provide input signals
        model_type : str
            Type of ensemble model: 'random_forest' or 'gradient_boosting'
        training_period : int
            Number of days to use for training the model
        """
        super().__init__()
        self.strategies = strategies if strategies else []
        self.model = EnsembleModel(model_type, training_period)
        
    def add_strategy(self, strategy):
        """Add a strategy to the ensemble."""
        self.strategies.append(strategy)
    
    def _prepare_features(self, data):
        """
        Generate features from all strategies for model training/prediction.
        """
        if not self.strategies:
            return None
            
        features = []
        for i, strategy in enumerate(self.strategies):
            signals = strategy.generate_signals(data)
            if signals is not None:
                if isinstance(signals, pd.Series):
                    features.append(signals.rename(f'strategy_{i}'))
                else:
                    # Convert to Series if it's not already
                    features.append(pd.Series(signals, index=data.index, name=f'strategy_{i}'))
        
        if not features:
            return None
            
        # Combine all strategy signals as features
        return pd.concat(features, axis=1)
    
    def train(self, data, return_data):
        """
        Train the ensemble model on historical data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Market data for feature generation
        return_data : pandas.Series
            Future returns to use as target variable (shifted accordingly)
            
        Returns:
        --------
        bool
            True if training was successful
        """
        features = self._prepare_features(data)
        if features is None:
            return False
            
        # Create target: 1 for positive returns, -1 for negative, 0 for flat
        target = pd.Series(np.sign(return_data), index=return_data.index)
        
        # Train the model
        return self.model.train(features, target)
    
    def generate_signals(self, data, target_returns=None):
        """
        Generate trading signals using the trained ensemble model.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Market data for signal generation
        target_returns : pandas.Series, optional
            Future returns for training if needed
            
        Returns:
        --------
        pandas.Series
            Trading signals
        """
        # Train if target returns provided
        if target_returns is not None:
            self.train(data, target_returns)
            
        # Generate features
        features = self._prepare_features(data)
        if features is None:
            return None
            
        # Get predictions
        predictions = self.model.predict(features)
        if predictions is None:
            return None
            
        # Return signals as a pandas Series
        return pd.Series(predictions, index=data.index)