"""
Performance monitoring utilities for comprehensive testing.

This module provides functionality to monitor system performance during
test execution, including memory usage, CPU utilization, and execution timing.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read: int
    disk_io_write: int
    network_sent: int
    network_recv: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_mb': self.memory_mb,
            'disk_io_read': self.disk_io_read,
            'disk_io_write': self.disk_io_write,
            'network_sent': self.network_sent,
            'network_recv': self.network_recv
        }


@dataclass
class TestPerformanceProfile:
    """Performance profile for a specific test or test suite."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    peak_memory_mb: float = 0
    peak_cpu_percent: float = 0
    avg_memory_mb: float = 0
    avg_cpu_percent: float = 0
    metrics: List[PerformanceMetrics] = field(default_factory=list)
    
    def finalize(self):
        """Finalize the performance profile with calculated metrics."""
        if self.end_time is None:
            self.end_time = time.time()
        
        self.duration = self.end_time - self.start_time
        
        if self.metrics:
            self.peak_memory_mb = max(m.memory_mb for m in self.metrics)
            self.peak_cpu_percent = max(m.cpu_percent for m in self.metrics)
            self.avg_memory_mb = sum(m.memory_mb for m in self.metrics) / len(self.metrics)
            self.avg_cpu_percent = sum(m.cpu_percent for m in self.metrics) / len(self.metrics)


class PerformanceMonitor:
    """
    Monitors system performance during test execution.
    
    Provides real-time monitoring of CPU, memory, disk I/O, and network
    usage with the ability to create performance profiles for specific
    test suites or operations.
    """
    
    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize the performance monitor.
        
        Args:
            sampling_interval: How often to sample performance metrics (seconds)
        """
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.monitor_thread = None
        self.metrics: List[PerformanceMetrics] = []
        self.profiles: Dict[str, TestPerformanceProfile] = {}
        self.active_profiles: Dict[str, TestPerformanceProfile] = {}
        
        # Baseline metrics
        self.baseline_metrics = self._get_current_metrics()
    
    def _get_current_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics."""
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / (1024 * 1024)
        
        # Get disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_read = disk_io.read_bytes if disk_io else 0
        disk_io_write = disk_io.write_bytes if disk_io else 0
        
        # Get network I/O
        network_io = psutil.net_io_counters()
        network_sent = network_io.bytes_sent if network_io else 0
        network_recv = network_io.bytes_recv if network_io else 0
        
        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            disk_io_read=disk_io_read,
            disk_io_write=disk_io_write,
            network_sent=network_sent,
            network_recv=network_recv
        )
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        while self.monitoring:
            try:
                metrics = self._get_current_metrics()
                self.metrics.append(metrics)
                
                # Update active profiles
                for profile in self.active_profiles.values():
                    profile.metrics.append(metrics)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                break
    
    def start(self):
        """Start performance monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def start_profile(self, name: str) -> TestPerformanceProfile:
        """
        Start a new performance profile for a specific test or operation.
        
        Args:
            name: Name of the profile/test
            
        Returns:
            The created performance profile
        """
        profile = TestPerformanceProfile(
            name=name,
            start_time=time.time()
        )
        self.active_profiles[name] = profile
        return profile
    
    def end_profile(self, name: str) -> Optional[TestPerformanceProfile]:
        """
        End a performance profile and move it to completed profiles.
        
        Args:
            name: Name of the profile to end
            
        Returns:
            The completed performance profile, or None if not found
        """
        if name in self.active_profiles:
            profile = self.active_profiles.pop(name)
            profile.finalize()
            self.profiles[name] = profile
            return profile
        return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.
        
        Returns:
            Dictionary containing performance summary
        """
        if not self.metrics:
            return {}
        
        peak_memory = max(m.memory_mb for m in self.metrics)
        peak_cpu = max(m.cpu_percent for m in self.metrics)
        avg_memory = sum(m.memory_mb for m in self.metrics) / len(self.metrics)
        avg_cpu = sum(m.cpu_percent for m in self.metrics) / len(self.metrics)
        
        return {
            'monitoring_duration': len(self.metrics) * self.sampling_interval,
            'peak_memory_mb': peak_memory,
            'peak_cpu_percent': peak_cpu,
            'avg_memory_mb': avg_memory,
            'avg_cpu_percent': avg_cpu,
            'baseline_memory_mb': self.baseline_metrics.memory_mb,
            'baseline_cpu_percent': self.baseline_metrics.cpu_percent,
            'memory_increase_mb': peak_memory - self.baseline_metrics.memory_mb,
            'cpu_increase_percent': peak_cpu - self.baseline_metrics.cpu_percent
        }
    
    def get_profile_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get performance summary for a specific profile.
        
        Args:
            name: Name of the profile
            
        Returns:
            Dictionary containing profile summary, or None if not found
        """
        profile = self.profiles.get(name)
        if not profile:
            return None
        
        return {
            'name': profile.name,
            'duration': profile.duration,
            'peak_memory_mb': profile.peak_memory_mb,
            'peak_cpu_percent': profile.peak_cpu_percent,
            'avg_memory_mb': profile.avg_memory_mb,
            'avg_cpu_percent': profile.avg_cpu_percent,
            'samples_collected': len(profile.metrics)
        }
    
    def check_performance_thresholds(self, thresholds: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if performance metrics exceed specified thresholds.
        
        Args:
            thresholds: Dictionary of threshold values
            
        Returns:
            Dictionary indicating which thresholds were exceeded
        """
        summary = self.get_performance_summary()
        results = {}
        
        if 'max_memory_mb' in thresholds:
            results['memory_exceeded'] = summary.get('peak_memory_mb', 0) > thresholds['max_memory_mb']
        
        if 'max_cpu_percent' in thresholds:
            results['cpu_exceeded'] = summary.get('peak_cpu_percent', 0) > thresholds['max_cpu_percent']
        
        if 'max_memory_increase_mb' in thresholds:
            results['memory_increase_exceeded'] = summary.get('memory_increase_mb', 0) > thresholds['max_memory_increase_mb']
        
        return results
    
    def export_metrics(self, output_path: str = "performance_metrics.json") -> str:
        """
        Export all collected metrics to a JSON file.
        
        Args:
            output_path: Path to save the metrics file
            
        Returns:
            Path to the exported file
        """
        export_data = {
            'baseline_metrics': self.baseline_metrics.to_dict(),
            'summary': self.get_performance_summary(),
            'profiles': {
                name: {
                    'name': profile.name,
                    'duration': profile.duration,
                    'peak_memory_mb': profile.peak_memory_mb,
                    'peak_cpu_percent': profile.peak_cpu_percent,
                    'avg_memory_mb': profile.avg_memory_mb,
                    'avg_cpu_percent': profile.avg_cpu_percent,
                    'metrics': [m.to_dict() for m in profile.metrics]
                }
                for name, profile in self.profiles.items()
            },
            'all_metrics': [m.to_dict() for m in self.metrics]
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return str(output_file)
    
    def generate_report(self, output_path: str = "performance_report.html") -> str:
        """
        Generate an HTML performance report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        summary = self.get_performance_summary()
        
        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Test Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .profiles {{
            margin-top: 30px;
        }}
        .profile {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }}
        .profile-header {{
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 10px;
        }}
        .profile-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }}
        .profile-metric {{
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Performance Test Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Peak Memory</h3>
                <h2>{summary.get('peak_memory_mb', 0):.1f} MB</h2>
            </div>
            <div class="metric-card">
                <h3>Peak CPU</h3>
                <h2>{summary.get('peak_cpu_percent', 0):.1f}%</h2>
            </div>
            <div class="metric-card">
                <h3>Avg Memory</h3>
                <h2>{summary.get('avg_memory_mb', 0):.1f} MB</h2>
            </div>
            <div class="metric-card">
                <h3>Avg CPU</h3>
                <h2>{summary.get('avg_cpu_percent', 0):.1f}%</h2>
            </div>
        </div>
        
        <div class="profiles">
            <h2>Test Suite Profiles</h2>
            {self._generate_profile_html()}
        </div>
    </div>
</body>
</html>
        """
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(html_content)
        
        return str(output_file)
    
    def _generate_profile_html(self) -> str:
        """Generate HTML for profile sections."""
        html_parts = []
        
        for name, profile in self.profiles.items():
            html_parts.append(f"""
            <div class="profile">
                <div class="profile-header">{profile.name}</div>
                <div class="profile-metrics">
                    <div class="profile-metric">
                        <strong>Duration</strong><br>
                        {profile.duration:.2f}s
                    </div>
                    <div class="profile-metric">
                        <strong>Peak Memory</strong><br>
                        {profile.peak_memory_mb:.1f} MB
                    </div>
                    <div class="profile-metric">
                        <strong>Peak CPU</strong><br>
                        {profile.peak_cpu_percent:.1f}%
                    </div>
                    <div class="profile-metric">
                        <strong>Avg Memory</strong><br>
                        {profile.avg_memory_mb:.1f} MB
                    </div>
                    <div class="profile-metric">
                        <strong>Avg CPU</strong><br>
                        {profile.avg_cpu_percent:.1f}%
                    </div>
                </div>
            </div>
            """)
        
        return "\n".join(html_parts)


# Context manager for easy performance profiling
class PerformanceProfile:
    """Context manager for performance profiling."""
    
    def __init__(self, monitor: PerformanceMonitor, name: str):
        self.monitor = monitor
        self.name = name
        self.profile = None
    
    def __enter__(self):
        self.profile = self.monitor.start_profile(self.name)
        return self.profile
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.end_profile(self.name)