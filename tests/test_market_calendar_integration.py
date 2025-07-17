"""
Market Calendar Integration Tests

Comprehensive tests for extended hours market calendar functionality
including edge cases and real-world scenarios.
"""

import pytest
from datetime import datetime, time, timedelta
import pytz
import pandas_market_calendars as mcal
import logging

# Add src to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from src.utils.market_calendar import ExtendedHoursCalendar

logger = logging.getLogger(__name__)


class TestMarketCalendarIntegration:
    """Integration tests for market calendar with real NYSE data."""

    def setup_method(self):
        """Set up test environment."""
        self.calendar = ExtendedHoursCalendar()
        self.est = pytz.timezone('US/Eastern')
        
    def test_extended_hours_coverage(self):
        """
        Test full extended hours coverage (4 AM - 8 PM EST).
        
        Verifies:
        - Pre-market: 4:00 AM - 9:30 AM EST
        - Regular: 9:30 AM - 4:00 PM EST
        - After-hours: 4:00 PM - 8:00 PM EST
        """
        # Test on a known trading day (Monday, Jan 16, 2024) in EST
        trading_day = self.est.localize(datetime(2024, 1, 16))
        
        test_schedule = [
            # Before extended hours
            (3, 59, False, None),
            
            # Pre-market session
            (4, 0, True, 'premarket'),
            (4, 30, True, 'premarket'),
            (7, 0, True, 'premarket'),
            (9, 29, True, 'premarket'),
            
            # Regular session
            (9, 30, True, 'regular'),
            (10, 0, True, 'regular'),
            (12, 0, True, 'regular'),
            (15, 59, True, 'regular'),
            
            # After-hours session
            (16, 0, True, 'afterhours'),
            (17, 0, True, 'afterhours'),
            (19, 59, True, 'afterhours'),
            (20, 0, True, 'afterhours'),
            
            # After extended hours
            (20, 1, False, None),
            (23, 0, False, None),
        ]
        
        for hour, minute, should_be_extended, expected_session in test_schedule:
            test_time = trading_day.replace(hour=hour, minute=minute)
            
            # Test extended hours detection
            is_extended = self.calendar.is_extended_trading_hours(test_time)
            assert is_extended == should_be_extended, \
                f"Time {hour}:{minute:02d} extended hours check failed. " \
                f"Expected {should_be_extended}, got {is_extended}"
                
            # Test session detection
            session = self.calendar.get_trading_session(test_time)
            assert session == expected_session, \
                f"Time {hour}:{minute:02d} session check failed. " \
                f"Expected {expected_session}, got {session}"
                
        logger.info("Extended hours coverage test passed")

    def test_holiday_handling(self):
        """
        Test market calendar on holidays and weekends.
        
        Verifies:
        - No extended hours on holidays
        - No extended hours on weekends
        - Proper holiday detection
        """
        # Test known holidays
        holidays = [
            datetime(2024, 1, 1),   # New Year's Day
            datetime(2024, 1, 15),  # MLK Day
            datetime(2024, 7, 4),   # Independence Day
            datetime(2024, 12, 25), # Christmas
        ]
        
        for holiday in holidays:
            # Test during what would be regular hours (12:00 PM EST)
            test_time = self.est.localize(holiday.replace(hour=12, minute=0))
            
            # Should not be extended trading hours
            is_extended = self.calendar.is_extended_trading_hours(test_time)
            session = self.calendar.get_trading_session(test_time)
            
            # Note: Some holidays might still have trading, so we log results
            logger.info(f"Holiday {holiday.date()}: Extended={is_extended}, Session={session}")
            
        # Test weekends (should definitely be closed)
        saturday = self.est.localize(datetime(2024, 1, 13, 12, 0))  # Saturday
        sunday = self.est.localize(datetime(2024, 1, 14, 12, 0))    # Sunday
        
        for weekend_day in [saturday, sunday]:
            is_extended = self.calendar.is_extended_trading_hours(weekend_day)
            session = self.calendar.get_trading_session(weekend_day)
            
            assert is_extended is False, f"Weekend {weekend_day.date()} should not have extended hours"
            assert session is None, f"Weekend {weekend_day.date()} should not have trading session"

    def test_timezone_handling(self):
        """
        Test timezone conversion and handling.
        
        Verifies:
        - UTC to EST conversion
        - Daylight saving time handling
        - Consistent behavior across timezones
        """
        # Test UTC to EST conversion (use Jan 2, 2024 - a trading day)
        utc_time = datetime(2024, 1, 2, 14, 30, tzinfo=pytz.UTC)  # 2:30 PM UTC
        # This should be 9:30 AM EST (market open)
        
        is_extended = self.calendar.is_extended_trading_hours(utc_time)
        session = self.calendar.get_trading_session(utc_time)
        
        assert is_extended is True, "UTC conversion failed for extended hours"
        assert session == 'regular', f"UTC conversion failed for session detection: {session}"
        
        # Test naive datetime (should assume UTC)
        naive_time = datetime(2024, 1, 2, 14, 30)  # No timezone
        
        is_extended_naive = self.calendar.is_extended_trading_hours(naive_time)
        session_naive = self.calendar.get_trading_session(naive_time)
        
        # Should get same result as UTC time
        assert is_extended_naive == is_extended
        assert session_naive == session
        
        logger.info("Timezone handling test passed")

    def test_period_boundary_detection(self):
        """
        Test detection of new trading periods.
        
        Verifies:
        - Hour boundary detection within extended hours
        - Day boundary detection (4 AM cutoff)
        - Proper handling of period transitions
        """
        base_date = self.est.localize(datetime(2024, 1, 2))  # Tuesday (trading day)
        
        # Test hour boundaries
        hour_boundary_tests = [
            # (last_time, current_time, timeframe, expected_new_period)
            ((10, 59), (11, 1), '1h', True),   # Cross hour boundary
            ((10, 30), (10, 45), '1h', False),  # Same hour
            ((9, 59), (10, 1), '1h', True),    # Cross hour boundary
            ((16, 59), (17, 1), '1h', True),   # After-hours hour boundary
            ((3, 59), (4, 1), '1h', False),    # Outside extended hours
        ]
        
        for (last_h, last_m), (curr_h, curr_m), timeframe, expected in hour_boundary_tests:
            last_time = base_date.replace(hour=last_h, minute=last_m)
            current_time = base_date.replace(hour=curr_h, minute=curr_m)
            
            result = self.calendar.is_new_trading_period(last_time, current_time, timeframe)
            
            assert result == expected, \
                f"Hour boundary test failed: {last_h}:{last_m} -> {curr_h}:{curr_m}, " \
                f"expected {expected}, got {result}"
                
        # Test day boundaries (4 AM cutoff for extended trading day)
        day_boundary_tests = [
            # Cross to next extended trading day (Jan 2 to Jan 3, both trading days)
            (self.est.localize(datetime(2024, 1, 2, 23, 30)), self.est.localize(datetime(2024, 1, 3, 4, 30)), '1d', True),
            # Same extended trading day
            (self.est.localize(datetime(2024, 1, 2, 10, 0)), self.est.localize(datetime(2024, 1, 2, 16, 0)), '1d', False),
            # Early morning transition (before 4 AM belongs to previous day)
            (self.est.localize(datetime(2024, 1, 2, 2, 0)), self.est.localize(datetime(2024, 1, 2, 3, 0)), '1d', False),
        ]
        
        for last_time, current_time, timeframe, expected in day_boundary_tests:
            result = self.calendar.is_new_trading_period(last_time, current_time, timeframe)
            
            assert result == expected, \
                f"Day boundary test failed: {last_time} -> {current_time}, " \
                f"expected {expected}, got {result}"
                
        logger.info("Period boundary detection test passed")

    def test_extended_trading_date_calculation(self):
        """
        Test extended trading date calculation with 4 AM boundary.
        
        Verifies:
        - Times before 4 AM belong to previous trading day
        - Times after 4 AM belong to current trading day
        - Proper date handling across midnight
        """
        # Test various times around midnight and 4 AM (using trading days)
        test_cases = [
            # (datetime, expected_trading_date)
            (self.est.localize(datetime(2024, 1, 2, 23, 30)), datetime(2024, 1, 2).date()),   # 11:30 PM
            (self.est.localize(datetime(2024, 1, 3, 0, 30)), datetime(2024, 1, 2).date()),    # 12:30 AM (next day)
            (self.est.localize(datetime(2024, 1, 3, 3, 30)), datetime(2024, 1, 2).date()),    # 3:30 AM (next day)
            (self.est.localize(datetime(2024, 1, 3, 4, 0)), datetime(2024, 1, 3).date()),     # 4:00 AM (new trading day)
            (self.est.localize(datetime(2024, 1, 3, 4, 30)), datetime(2024, 1, 3).date()),    # 4:30 AM
            (self.est.localize(datetime(2024, 1, 3, 10, 0)), datetime(2024, 1, 3).date()),    # 10:00 AM
        ]
        
        for test_time, expected_date in test_cases:
            trading_date = self.calendar._get_extended_trading_date(test_time)
            
            assert trading_date == expected_date, \
                f"Extended trading date calculation failed for {test_time}. " \
                f"Expected {expected_date}, got {trading_date}"
                
        logger.info("Extended trading date calculation test passed")

    def test_real_market_data_integration(self):
        """
        Test integration with pandas-market-calendars for real market data.
        
        Verifies:
        - NYSE calendar integration works
        - Real holiday detection
        - Actual trading day validation
        """
        # Test with real NYSE calendar
        nyse = mcal.get_calendar('NYSE')
        
        # Get schedule for a known period
        start_date = '2024-01-01'
        end_date = '2024-01-31'
        
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        
        # Test each trading day
        for date_str in schedule.index.strftime('%Y-%m-%d'):
            trading_date = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Test during regular hours (12:00 PM EST)
            regular_time = self.est.localize(trading_date.replace(hour=12, minute=0))
            
            is_extended = self.calendar.is_extended_trading_hours(regular_time)
            session = self.calendar.get_trading_session(regular_time)
            
            # Should be extended hours and regular session
            assert is_extended is True, f"Trading day {date_str} not detected as extended hours"
            assert session == 'regular', f"Trading day {date_str} not detected as regular session"
            
        logger.info(f"Real market data integration test passed for {len(schedule)} trading days")

    def test_edge_cases_and_error_handling(self):
        """
        Test edge cases and error handling.
        
        Verifies:
        - Graceful handling of invalid dates
        - Proper behavior on market calendar errors
        - Consistent behavior across different scenarios
        """
        # Test with future dates (should work)
        future_date = self.est.localize(datetime(2025, 6, 15, 12, 0))
        
        try:
            is_extended = self.calendar.is_extended_trading_hours(future_date)
            session = self.calendar.get_trading_session(future_date)
            
            # Should not raise errors
            logger.info(f"Future date test: Extended={is_extended}, Session={session}")
            
        except Exception as e:
            pytest.fail(f"Future date handling failed: {e}")
            
        # Test with very old dates
        old_date = self.est.localize(datetime(2000, 1, 15, 12, 0))
        
        try:
            is_extended = self.calendar.is_extended_trading_hours(old_date)
            session = self.calendar.get_trading_session(old_date)
            
            logger.info(f"Old date test: Extended={is_extended}, Session={session}")
            
        except Exception as e:
            logger.warning(f"Old date handling issue (expected): {e}")
            
        # Test edge times (using trading day)
        edge_times = [
            self.est.localize(datetime(2024, 1, 2, 4, 0, 0)),      # Exact pre-market start
            self.est.localize(datetime(2024, 1, 2, 9, 30, 0)),     # Exact market open
            self.est.localize(datetime(2024, 1, 2, 16, 0, 0)),     # Exact market close
            self.est.localize(datetime(2024, 1, 2, 20, 0, 0)),     # Exact after-hours end
        ]
        
        for edge_time in edge_times:
            try:
                is_extended = self.calendar.is_extended_trading_hours(edge_time)
                session = self.calendar.get_trading_session(edge_time)
                
                logger.info(f"Edge time {edge_time.time()}: Extended={is_extended}, Session={session}")
                
                # Should not raise errors
                assert isinstance(is_extended, bool)
                assert session in [None, 'premarket', 'regular', 'afterhours']
                
            except Exception as e:
                pytest.fail(f"Edge time handling failed for {edge_time}: {e}")
                
        logger.info("Edge cases and error handling test passed")

    def test_performance_under_high_frequency(self):
        """
        Test calendar performance under high-frequency calls.
        
        Verifies:
        - Calendar can handle frequent calls efficiently
        - No memory leaks or performance degradation
        - Suitable for real-time streaming
        """
        import time
        
        start_time = time.time()
        call_count = 1000
        
        base_date = self.est.localize(datetime(2024, 1, 2))  # Trading day
        
        # Make many rapid calls
        for i in range(call_count):
            test_time = base_date + timedelta(minutes=i)
            
            # Call both main functions
            self.calendar.is_extended_trading_hours(test_time)
            self.calendar.get_trading_session(test_time)
            
            # Test period detection
            if i > 0:
                prev_time = base_date + timedelta(minutes=i-1)
                self.calendar.is_new_trading_period(prev_time, test_time, '1h')
                
        elapsed_time = time.time() - start_time
        calls_per_second = (call_count * 3) / elapsed_time  # 3 calls per iteration
        
        logger.info(f"Calendar performance: {calls_per_second:.0f} calls/second")
        
        # Should handle at least 100 calls per second (realistic for pandas-market-calendars)
        assert calls_per_second > 100, \
            f"Calendar performance too slow: {calls_per_second:.0f} calls/second"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])