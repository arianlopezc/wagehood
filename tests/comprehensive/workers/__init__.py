"""
Worker process tests for the Wagehood trading system.

This module contains tests that validate the background worker processes
including calculation engine, data ingestion, signal processing, and real-time processing.
"""

# Worker test constants
WORKER_STARTUP_TIMEOUT = 30.0  # seconds
WORKER_SHUTDOWN_TIMEOUT = 10.0  # seconds
MESSAGE_PROCESSING_TIMEOUT = 5.0  # seconds
WORKER_HEALTH_CHECK_INTERVAL = 1.0  # seconds

# Test worker configurations
WORKER_TEST_CONFIG = {
    'calculation_engine': {
        'max_memory_mb': 512,
        'max_cpu_percent': 80,
        'max_processing_time': 5.0,
        'batch_size': 1000
    },
    'data_ingestion': {
        'max_memory_mb': 256,
        'max_cpu_percent': 60,
        'max_latency_ms': 500,
        'throughput_min': 100  # messages per second
    },
    'signal_engine': {
        'max_memory_mb': 256,
        'max_cpu_percent': 70,
        'max_processing_time': 2.0,
        'accuracy_threshold': 0.95
    },
    'stream_processor': {
        'max_memory_mb': 512,
        'max_cpu_percent': 75,
        'max_latency_ms': 100,
        'throughput_min': 500  # messages per second
    }
}

# Worker communication patterns
WORKER_COMMUNICATION = {
    'queue_patterns': ['request_response', 'publish_subscribe', 'work_queue'],
    'message_formats': ['json', 'binary', 'protobuf'],
    'retry_policies': ['exponential_backoff', 'linear_backoff', 'fixed_interval']
}