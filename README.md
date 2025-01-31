# Predictive Maintenance Monitoring System

A real-time monitoring system for predictive maintenance using machine learning and streaming data analysis.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd predictive_maintenance
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install and start Kafka:
```bash
sudo apt update
sudo apt install default-jdk

cd ~
wget https://downloads.apache.org/kafka/3.6.1/kafka_2.13-3.6.1.tgz
tar xzf kafka_2.13-3.6.1.tgz
cd kafka_2.13-3.6.1
```

## Usage

1. Start Kafka (in separate terminals):
```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka Server
bin/kafka-server-start.sh config/server.properties
```

2. Create Kafka topic:
```bash
bin/kafka-topics.sh --create --topic machine_sensors --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1
```

3. Train the model:
```bash
python src/utils/train_model.py
```

4. Start the data simulator:
```bash
python src/utils/data_simulator.py
```

5. Launch the dashboard:
```bash
streamlit run src/utils/dashboard.py
```

## Project Structure

```
predictive_maintenance/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_simulator.py
│   │   ├── preprocessing.py
│   │   └── dashboard.py
│   └── data/
│       └── ai4i2020.csv
├── models/
│   └── predictive_maintenance_model.joblib
├── requirements.txt
└── README.md
```
