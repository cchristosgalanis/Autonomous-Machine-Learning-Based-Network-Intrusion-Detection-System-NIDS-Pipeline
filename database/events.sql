CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create the events table
CREATE TABLE network_traffic_events (
    -- foresincs and metadata
    time TIMESTAMPTZ NOT NULL, -- timestamp of the event
    source_ip VARCHAR(45) NOT NULL, -- source IP address
    analyzer VARCHAR(50), -- aanalyzer that generated the event
    is_host_ip BOOLEAN DEFAULT FALSE, -- host IP address flag


    -- Machine learning features
    velocity DOUBLE PRECISION, -- velocity of the traffic
    acceleration DOUBLE PRECISION, -- acceleration of the traffic
    iat_mean DOUBLE PRECISION, -- mean inter-arrival time
    sa_ratio DOUBLE PRECISION, -- source to destination ratio
    jaccard_score DOUBLE PRECISION, -- Jaccard similarity score
    entropy_shannon DOUBLE PRECISION, -- Shannon entropy of the traffic

    -- Neural network outcome
    probability DOUBLE PRECISION NOT NULL, -- probability of being malicious
    prediction INTEGER NOT NULL, -- Normal = 0, Attack = 1

    -- pipeline state
    is_processed BOOLEAN DEFAULT FALSE -- flag to indicate if the event has been processed
)

-- convert into TimescaleDB hypertable
SELECT create_hypertable('network_traffic_events', 'time');

-- Indexes for performance optimization
CREATE INDEX idx_unprocessed_events ON network_traffic_events (time) 
WHERE is_processed = FALSE;

-- dict index for attacks
CREATE INDEX idx_attacks_only ON network_traffic_events (time DESC) 
WHERE prediction = 1;

-- for a given source IP, get the most recent events
CREATE INDEX idx_source_ip ON network_traffic_events (source_ip, time DESC);

