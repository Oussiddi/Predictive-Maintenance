class Config:
    TRAINING_DATA_PATH = "src/data/ai4i2020.csv"
    MODEL_SAVE_PATH = "models/predictive_maintenance_model.joblib"
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    NUMERICAL_FEATURES = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]
    
    CATEGORICAL_FEATURES = ['Type']
    
    FAILURE_COLUMN = 'Machine failure'
    TARGET_COLUMNS = [
        'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
    ]
    
    # Kafka configuration
    KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
    KAFKA_TOPIC = 'machine_sensors'
    
    # Alert thresholds
    FAILURE_PROBABILITY_THRESHOLD = 0.7
    HIGH_ALERT_THRESHOLD = 0.9