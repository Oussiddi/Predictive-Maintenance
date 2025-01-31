import streamlit as st
import pandas as pd
import numpy as np
from kafka import KafkaConsumer
import json
from datetime import datetime
import time
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config

def preprocess_data(data, model_data):
    try:
        df = pd.DataFrame([data])
        
        numerical_data = df[Config.NUMERICAL_FEATURES]
        numerical_scaled = model_data['numerical_scaler'].transform(numerical_data)
        
        categorical_encoded = model_data['label_encoder'].transform([df['Type'].iloc[0]])
        
        X = np.column_stack([numerical_scaled, categorical_encoded])
        X = pd.DataFrame(X, columns=model_data['feature_names'])
        
        return X
        
    except Exception as e:
        raise Exception(f"Preprocessing error: {str(e)}")

def create_dashboard():
    st.set_page_config(
        page_title="Predictive Maintenance Monitor",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("Predictive Maintenance Monitoring")
    
    try:
        model_data = joblib.load('models/predictive_maintenance_model.joblib')
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    if 'data' not in st.session_state:
        st.session_state.data = {
            'timestamps': deque(maxlen=100),
            'values': {feature: deque(maxlen=100) for feature in Config.NUMERICAL_FEATURES},
            'predictions': deque(maxlen=100),
            'alert_count': 0
        }
    
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    consumer = KafkaConsumer(
        Config.KAFKA_TOPIC,
        bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='latest',
        group_id='monitoring_dashboard',
        consumer_timeout_ms=1000
    )
    
    try:
        while True:
            messages = consumer.poll(timeout_ms=1000)
            
            for topic_partition, records in messages.items():
                for record in records:
                    data = record.value
                    current_time = datetime.now().strftime('%H:%M:%S')
                    
                    st.session_state.data['timestamps'].append(current_time)
                    for feature in Config.NUMERICAL_FEATURES:
                        st.session_state.data['values'][feature].append(data[feature])
                    
                    try:
                        X = preprocess_data(data, model_data)
                        
                        prediction = model_data['model'].predict_proba(X)[0]
                        failure_prob = prediction[1]
                        is_failure = failure_prob > Config.FAILURE_PROBABILITY_THRESHOLD
                        
                        st.session_state.data['predictions'].append(failure_prob)
                        if is_failure:
                            st.session_state.data['alert_count'] += 1
                            
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        continue
                    
                    with metrics_placeholder.container():
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Machine Status",
                                "Alert" if is_failure else "Normal",
                                f"Failure Probability: {failure_prob:.2%}"
                            )
                        
                        with col2:
                            st.metric(
                                "Alert Count",
                                st.session_state.data['alert_count']
                            )
                            
                        with col3:
                            st.metric(
                                "Latest Update",
                                current_time
                            )
                    
                    with chart_placeholder.container():
                        fig = make_subplots(
                            rows=len(Config.NUMERICAL_FEATURES) + 1,
                            cols=1,
                            subplot_titles=Config.NUMERICAL_FEATURES + ['Failure Probability']
                        )
                        
                        for idx, feature in enumerate(Config.NUMERICAL_FEATURES, 1):
                            fig.add_trace(
                                go.Scatter(
                                    x=list(st.session_state.data['timestamps']),
                                    y=list(st.session_state.data['values'][feature]),
                                    name=feature
                                ),
                                row=idx,
                                col=1
                            )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=list(st.session_state.data['timestamps']),
                                y=list(st.session_state.data['predictions']),
                                name='Failure Probability',
                                line=dict(color='red')
                            ),
                            row=len(Config.NUMERICAL_FEATURES) + 1,
                            col=1
                        )
                        
                        fig.update_layout(
                            height=1000,
                            showlegend=True,
                            title_text="Sensor Readings and Predictions"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        st.warning("Dashboard stopped by user")
    except Exception as e:
        st.error(f"Dashboard error: {str(e)}")
    finally:
        consumer.close()

if __name__ == "__main__":
    create_dashboard()