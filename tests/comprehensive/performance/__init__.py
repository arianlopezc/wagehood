"""
Performance and stress tests for the Wagehood trading system.

This module contains tests that validate system performance under various loads,
stress conditions, and resource constraints.
"""

# Performance test constants
BENCHMARK_DURATION = 300.0  # 5 minutes for benchmark tests
STRESS_TEST_DURATION = 600.0  # 10 minutes for stress tests
LOAD_RAMP_UP_TIME = 60.0  # 1 minute ramp-up time
MEMORY_LEAK_TEST_DURATION = 1800.0  # 30 minutes for memory leak detection

# Performance benchmarks
PERFORMANCE_BENCHMARKS = {
    'indicator_calculation': {
        'target_ops_per_second': 10000,
        'max_memory_mb': 100,
        'max_cpu_percent': 70,
        'data_points': [1000, 5000, 10000, 50000]
    },
    'signal_generation': {
        'target_ops_per_second': 1000,
        'max_memory_mb': 200,
        'max_cpu_percent': 80,
        'symbols': [1, 10, 50, 100]
    },
    'portfolio_calculations': {
        'target_ops_per_second': 500,
        'max_memory_mb': 150,
        'max_cpu_percent': 75,
        'positions': [10, 50, 100, 500]
    },
    'data_ingestion': {
        'target_messages_per_second': 5000,
        'max_memory_mb': 300,
        'max_cpu_percent': 60,
        'concurrent_streams': [1, 5, 10, 20]
    }
}

# Stress test scenarios
STRESS_SCENARIOS = {
    'high_frequency_data': {
        'description': 'Process high-frequency market data',
        'data_rate': 'tick_by_tick',
        'symbols': 100,
        'duration_minutes': 10
    },
    'memory_pressure': {
        'description': 'Test under memory pressure',
        'memory_limit_mb': 512,
        'data_volume': 'large',
        'duration_minutes': 15
    },
    'cpu_intensive': {
        'description': 'CPU-intensive calculations',
        'calculation_complexity': 'high',
        'concurrent_calculations': 20,
        'duration_minutes': 10
    },
    'concurrent_users': {
        'description': 'Multiple concurrent user sessions',
        'user_count': 50,
        'operations_per_user': 100,
        'duration_minutes': 15
    }
}

# Resource limits for testing
RESOURCE_LIMITS = {
    'max_memory_mb': 2048,
    'max_cpu_percent': 90,
    'max_disk_io_mbps': 100,
    'max_network_io_mbps': 50,
    'max_open_files': 1000,
    'max_threads': 100
}

# Performance regression detection
REGRESSION_THRESHOLDS = {
    'performance_degradation_percent': 10,  # 10% degradation threshold
    'memory_increase_percent': 20,  # 20% memory increase threshold
    'latency_increase_percent': 15,  # 15% latency increase threshold
    'throughput_decrease_percent': 10  # 10% throughput decrease threshold
}