"""
Custom metrics plugin for precision, recall, and F1 score.
"""
import torch
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


class PrecisionRecallF1Metrics(PluginMetric):
    """
    Plugin to compute Precision, Recall, and F1 score at the end of each epoch and experience.
    """
    
    def __init__(self):
        super().__init__()
        self.reset()
    
    def reset(self):
        """Reset the metric state."""
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_true, y_pred):
        """Update with new predictions."""
        self.y_true.extend(y_true.cpu().numpy())
        self.y_pred.extend(y_pred.cpu().numpy())
    
    def result(self):
        """Compute precision, recall, and F1."""
        if len(self.y_true) == 0:
            return {}
        
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def after_training_iteration(self, strategy, **kwargs):
        """Called after each training iteration."""
        y_true = strategy.mb_y
        y_pred = torch.argmax(strategy.mb_output, dim=1)
        self.update(y_true, y_pred)
    
    def after_training_epoch(self, strategy, **kwargs):
        """Called after each training epoch."""
        metrics = self.result()
        results = []
        
        for metric_name, value in metrics.items():
            results.append(MetricValue(
                self, metric_name, value, strategy.clock.train_exp_counter
            ))
        
        self.reset()
        return results
    
    def after_eval_iteration(self, strategy, **kwargs):
        """Called after each evaluation iteration."""
        y_true = strategy.mb_y
        y_pred = torch.argmax(strategy.mb_output, dim=1)
        self.update(y_true, y_pred)
    
    def after_eval_exp(self, strategy, **kwargs):
        """Called after each evaluation experience."""
        metrics = self.result()
        results = []
        
        for metric_name, value in metrics.items():
            results.append(MetricValue(
                self, f'Eval_{metric_name}', value, strategy.clock.train_exp_counter
            ))
        
        self.reset()
        return results
    
    def __str__(self):
        return "PrecisionRecallF1"
