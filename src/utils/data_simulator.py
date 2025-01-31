import pandas as pd
import numpy as np
import json
import time
from kafka import KafkaProducer
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import Config

class DataSimulator:
    def __init__(self):
        df = pd.read_csv('src/data/ai4i2020.csv')
        
        self.feature_stats = {}
        for feature in Config.NUMERICAL_FEATURES:
            self.feature_stats[feature] = {
                'mean': df[feature].mean(),
                'std': df[feature].std(),
                'min': df[feature].min(),
                'max': df[feature].max()
            }
        
        self.type_values = df['Type'].unique()
    
    def generate_sample(self, add_anomaly=False):
        sample = {}
        
        for feature, stats in self.feature_stats.items():
            if add_anomaly and np.random.random() < 0.3:
                value = stats['mean'] + stats['std'] * np.random.uniform(3, 5)
            else:
                value = np.random.normal(stats['mean'], stats['std'])
                value = np.clip(value, stats['min'], stats['max'])
            sample[feature] = float(value)
        
        sample['Type'] = str(np.random.choice(self.type_values))
        
        sample['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return sample

def run_simulator():

    producer = KafkaProducer(
        bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    simulator = DataSimulator()
    
    print("Starting data simulation...")
    print("Sending data every 5 seconds")
    print("Press Ctrl+C to stop")
    
    anomaly_counter = 0
    
    try:
        while True:
            add_anomaly = np.random.random() < 0.1
            
            data = simulator.generate_sample(add_anomaly)
            producer.send(Config.KAFKA_TOPIC, data)
            producer.flush()
            
            if add_anomaly:
                anomaly_counter += 1
                print(f"\nGenerated anomalous data point ({anomaly_counter})")
            else:
                print(".", end="", flush=True)
            
            # Wait for 5 seconds
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nStopping simulator...")
    finally:
        producer.close()

if __name__ == "__main__":
    run_simulator()