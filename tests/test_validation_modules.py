"""
Unit tests for validation modules.
Tests the new validation and monitoring modules to ensure they work correctly.
"""

import unittest
import time
from datetime import datetime
from src.validation.volume_validator import VolumeValidator
from src.validation.session_volume_logger import SessionVolumeLogger
from src.monitoring.timing_collector import TimingCollector
from src.monitoring.error_tracker import ErrorTracker


class TestVolumeValidator(unittest.TestCase):
    """Test cases for VolumeValidator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = VolumeValidator()
        self.enabled_validator = VolumeValidator(enabled=True)
        
    def test_disabled_by_default(self):
        """Test that validator is disabled by default"""
        self.assertFalse(self.validator.enabled)
        
    def test_validate_volume_disabled(self):
        """Test validation when disabled"""
        result, msg = self.validator.validate_volume('AAPL', 500)
        self.assertTrue(result)
        self.assertEqual(msg, "Volume validation disabled")
        
    def test_validate_volume_enabled_good(self):
        """Test validation with good volume"""
        result, msg = self.enabled_validator.validate_volume('AAPL', 2000)
        self.assertTrue(result)
        self.assertEqual(msg, "Volume OK")
        
    def test_validate_volume_enabled_low(self):
        """Test validation with low volume"""
        result, msg = self.enabled_validator.validate_volume('AAPL', 500)
        self.assertTrue(result)  # Should still return True (non-blocking)
        self.assertIn("Low volume", msg)
        
    def test_validate_volume_none(self):
        """Test validation with None volume"""
        result, msg = self.enabled_validator.validate_volume('AAPL', None)
        self.assertTrue(result)
        self.assertEqual(msg, "No volume data")
        
    def test_set_threshold(self):
        """Test setting custom threshold"""
        self.enabled_validator.set_threshold(2000)
        self.assertEqual(self.enabled_validator.min_volume_threshold, 2000)
        
    def test_set_enabled(self):
        """Test enabling/disabling validator"""
        self.validator.set_enabled(True)
        self.assertTrue(self.validator.enabled)
        self.validator.set_enabled(False)
        self.assertFalse(self.validator.enabled)


class TestSessionVolumeLogger(unittest.TestCase):
    """Test cases for SessionVolumeLogger"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.logger = SessionVolumeLogger()
        self.enabled_logger = SessionVolumeLogger(enabled=True)
        
    def test_disabled_by_default(self):
        """Test that logger is disabled by default"""
        self.assertFalse(self.logger.enabled)
        
    def test_log_session_volume_disabled(self):
        """Test logging when disabled (should do nothing)"""
        # This should not raise any exceptions
        self.logger.log_session_volume('AAPL', 1000)
        
    def test_log_session_volume_enabled(self):
        """Test logging when enabled"""
        # This should not raise any exceptions
        self.enabled_logger.log_session_volume('AAPL', 1000)
        
    def test_log_session_volume_with_timestamp(self):
        """Test logging with specific timestamp"""
        timestamp = datetime(2024, 7, 15, 10, 0, 0)
        # This should not raise any exceptions
        self.enabled_logger.log_session_volume('AAPL', 1000, timestamp)
        
    def test_set_enabled(self):
        """Test enabling/disabling logger"""
        self.logger.set_enabled(True)
        self.assertTrue(self.logger.enabled)
        self.logger.set_enabled(False)
        self.assertFalse(self.logger.enabled)


class TestTimingCollector(unittest.TestCase):
    """Test cases for TimingCollector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.collector = TimingCollector()
        self.enabled_collector = TimingCollector(enabled=True)
        
    def test_disabled_by_default(self):
        """Test that collector is disabled by default"""
        self.assertFalse(self.collector.enabled)
        
    def test_record_timing_disabled(self):
        """Test recording when disabled"""
        self.collector.record_timing('test_op', 0.1)
        summary = self.collector.get_summary()
        self.assertEqual(summary['status'], 'disabled or no data')
        
    def test_record_timing_enabled(self):
        """Test recording when enabled"""
        self.enabled_collector.record_timing('test_op', 0.1, {'key': 'value'})
        summary = self.enabled_collector.get_summary()
        self.assertTrue(summary['enabled'])
        self.assertEqual(summary['total_operations'], 1)
        self.assertEqual(summary['avg_duration'], 0.1)
        
    def test_multiple_recordings(self):
        """Test multiple timing recordings"""
        self.enabled_collector.record_timing('op1', 0.1)
        self.enabled_collector.record_timing('op2', 0.2)
        self.enabled_collector.record_timing('op1', 0.3)
        
        summary = self.enabled_collector.get_summary()
        self.assertEqual(summary['total_operations'], 3)
        self.assertAlmostEqual(summary['avg_duration'], 0.2, places=2)
        self.assertEqual(summary['max_duration'], 0.3)
        self.assertEqual(summary['min_duration'], 0.1)
        self.assertEqual(summary['operation_counts']['op1'], 2)
        self.assertEqual(summary['operation_counts']['op2'], 1)
        
    def test_clear_history(self):
        """Test clearing timing history"""
        self.enabled_collector.record_timing('test_op', 0.1)
        self.enabled_collector.clear_history()
        summary = self.enabled_collector.get_summary()
        self.assertEqual(summary['status'], 'disabled or no data')
        
    def test_set_enabled(self):
        """Test enabling/disabling collector"""
        self.collector.set_enabled(True)
        self.assertTrue(self.collector.enabled)
        self.collector.set_enabled(False)
        self.assertFalse(self.collector.enabled)


class TestErrorTracker(unittest.TestCase):
    """Test cases for ErrorTracker"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tracker = ErrorTracker()
        self.enabled_tracker = ErrorTracker(enabled=True)
        
    def test_disabled_by_default(self):
        """Test that tracker is disabled by default"""
        self.assertFalse(self.tracker.enabled)
        
    def test_track_error_disabled(self):
        """Test tracking when disabled"""
        self.tracker.track_error('component', 'AAPL', 'error message')
        summary = self.tracker.get_error_summary()
        self.assertEqual(summary['status'], 'disabled')
        
    def test_track_error_enabled(self):
        """Test tracking when enabled"""
        self.enabled_tracker.track_error('rsi_strategy', 'AAPL', 'test error', 'data_quality')
        summary = self.enabled_tracker.get_error_summary()
        self.assertTrue(summary['enabled'])
        self.assertEqual(summary['total_errors'], 1)
        self.assertEqual(summary['error_counts']['rsi_strategy_data_quality'], 1)
        
    def test_multiple_errors(self):
        """Test tracking multiple errors"""
        self.enabled_tracker.track_error('rsi_strategy', 'AAPL', 'error 1', 'data_quality')
        self.enabled_tracker.track_error('rsi_strategy', 'TSLA', 'error 2', 'data_quality')
        self.enabled_tracker.track_error('bollinger_strategy', 'AAPL', 'error 3', 'ta_lib')
        
        summary = self.enabled_tracker.get_error_summary()
        self.assertEqual(summary['total_errors'], 3)
        self.assertEqual(summary['error_counts']['rsi_strategy_data_quality'], 2)
        self.assertEqual(summary['error_counts']['bollinger_strategy_ta_lib'], 1)
        
    def test_get_errors_by_component(self):
        """Test getting errors by component"""
        self.enabled_tracker.track_error('rsi_strategy', 'AAPL', 'error 1')
        self.enabled_tracker.track_error('bollinger_strategy', 'TSLA', 'error 2')
        self.enabled_tracker.track_error('rsi_strategy', 'MSFT', 'error 3')
        
        rsi_errors = self.enabled_tracker.get_errors_by_component('rsi_strategy')
        self.assertEqual(len(rsi_errors), 2)
        
        bollinger_errors = self.enabled_tracker.get_errors_by_component('bollinger_strategy')
        self.assertEqual(len(bollinger_errors), 1)
        
    def test_clear_history(self):
        """Test clearing error history"""
        self.enabled_tracker.track_error('component', 'AAPL', 'error')
        self.enabled_tracker.clear_history()
        summary = self.enabled_tracker.get_error_summary()
        self.assertEqual(summary['total_errors'], 0)
        self.assertEqual(summary['error_counts'], {})
        
    def test_set_enabled(self):
        """Test enabling/disabling tracker"""
        self.tracker.set_enabled(True)
        self.assertTrue(self.tracker.enabled)
        self.tracker.set_enabled(False)
        self.assertFalse(self.tracker.enabled)


if __name__ == '__main__':
    unittest.main()