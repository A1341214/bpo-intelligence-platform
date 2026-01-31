import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class BPOFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def create_features(self, agents_df, performance_df, calls_df):
        """Create features for ML models"""
        
        # Aggregate performance metrics
        perf_agg = performance_df.groupby('agent_id').agg({
            'calls_handled': ['mean', 'std', 'sum'],
            'avg_handle_time': ['mean', 'std'],
            'first_call_resolution_rate': ['mean', 'min'],
            'customer_satisfaction_score': ['mean', 'std', 'min'],
            'schedule_adherence': ['mean', 'min'],
            'transfer_rate': 'mean',
            'escalation_rate': 'mean'
        }).reset_index()
        
        perf_agg.columns = ['_'.join(col).strip('_') for col in perf_agg.columns.values]
        
        # Aggregate call metrics
        call_agg = calls_df.groupby('agent_id').agg({
            'call_duration': ['mean', 'std'],
            'hold_time': 'mean',
            'customer_satisfaction': 'mean'
        }).reset_index()
        
        call_agg.columns = ['_'.join(col).strip('_') for col in call_agg.columns.values]
        
        # Merge all features
        features = agents_df.merge(perf_agg, on='agent_id', how='left')
        features = features.merge(call_agg, on='agent_id', how='left')
        
        # Create additional features
        features['tenure_months'] = features['tenure_days'] / 30
        features['tenure_category'] = pd.cut(features['tenure_days'], 
                                              bins=[0, 90, 180, 365, 1825],
                                              labels=['0-3mo', '3-6mo', '6-12mo', '12mo+'])
        
        # Performance trend
        features['perf_variability'] = (
            features['customer_satisfaction_score_std'] /
            (features['customer_satisfaction_score_mean'] + 0.001)
        )
        
        return features
    
    def encode_categorical(self, df, categorical_cols):
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df_encoded
    
    def scale_features(self, df, numerical_cols, fit=True):
        """Scale numerical features"""
        df_scaled = df.copy()
        
        if fit:
            df_scaled[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df_scaled[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df_scaled