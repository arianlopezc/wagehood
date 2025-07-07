"""
Discord Webhook Notifier

Handles Discord webhook integration using only Python standard library.
Sends rich embed messages for trading signals without requiring external dependencies.
"""

import json
import logging
import os
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """
    Discord webhook notifier for trading signals.
    
    Uses Python standard library only (urllib) to send rich embed messages
    to Discord channels via webhooks.
    """
    
    def __init__(self, webhook_url: str, max_retries: int = 3):
        """
        Initialize Discord notifier.
        
        Args:
            webhook_url: Discord webhook URL
            max_retries: Maximum retry attempts for failed requests
        """
        self.webhook_url = webhook_url
        self.max_retries = max_retries
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # Minimum seconds between requests
        
        logger.info(f"Discord notifier initialized with webhook: {webhook_url[:50]}...")
    
    def send_embed(self, embed_data: Dict[str, Any], content: str = None) -> bool:
        """
        Send rich embed message to Discord.
        
        Args:
            embed_data: Discord embed dictionary
            content: Optional plain text content
            
        Returns:
            True if message sent successfully, False otherwise
        """
        # Prevent real Discord calls during testing
        if os.getenv('WAGEHOOD_TEST_MODE') == 'true':
            logger.info("Test mode detected - skipping actual Discord API call")
            return True
            
        try:
            # Rate limiting - ensure minimum delay between requests
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            
            # Prepare payload
            payload = {"embeds": [embed_data]}
            if content:
                payload["content"] = content
            
            # Send with retries
            for attempt in range(self.max_retries):
                try:
                    success = self._send_request(payload)
                    if success:
                        self.last_request_time = time.time()
                        logger.info("Discord message sent successfully")
                        return True
                    else:
                        logger.warning(f"Discord request failed, attempt {attempt + 1}/{self.max_retries}")
                        
                except Exception as e:
                    logger.error(f"Discord request error (attempt {attempt + 1}): {e}")
                
                # Wait before retry (exponential backoff)
                if attempt < self.max_retries - 1:
                    retry_delay = (2 ** attempt) * self.rate_limit_delay
                    logger.debug(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            
            logger.error("All Discord notification attempts failed")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error sending Discord notification: {e}")
            return False
    
    def _send_request(self, payload: Dict[str, Any]) -> bool:
        """
        Send HTTP request to Discord webhook.
        
        Args:
            payload: JSON payload to send
            
        Returns:
            True if request successful, False otherwise
        """
        try:
            # Prepare request
            data = json.dumps(payload).encode('utf-8')
            
            request = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'Wagehood-Trading-Bot/1.0'
                }
            )
            
            # Send request
            with urllib.request.urlopen(request, timeout=10) as response:
                status_code = response.getcode()
                
                if status_code == 200 or status_code == 204:
                    logger.debug(f"Discord webhook success: {status_code}")
                    return True
                else:
                    response_data = response.read().decode('utf-8')
                    logger.warning(f"Discord webhook returned {status_code}: {response_data}")
                    return False
                    
        except urllib.error.HTTPError as e:
            if e.code == 429:  # Rate limited
                logger.warning("Discord rate limit exceeded")
                retry_after = e.headers.get('Retry-After')
                if retry_after:
                    logger.info(f"Discord rate limit: retry after {retry_after} seconds")
                    time.sleep(float(retry_after))
                return False
            else:
                logger.error(f"Discord HTTP error {e.code}: {e.reason}")
                return False
                
        except urllib.error.URLError as e:
            logger.error(f"Discord URL error: {e.reason}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected Discord request error: {e}")
            return False
    
    def send_test_message(self, is_test_environment: bool = False) -> bool:
        """
        Send a test message to verify webhook connectivity.
        
        Args:
            is_test_environment: If True, prevents actual Discord API calls during testing
        
        Returns:
            True if test message sent successfully (or mocked in test environment)
        """
        if is_test_environment:
            logger.info("Test environment detected - skipping actual Discord API call")
            return True
            
        test_embed = {
            "title": "🧪 Wagehood Test Message",
            "description": "Discord webhook integration test",
            "color": 3447003,  # Blue
            "fields": [
                {
                    "name": "Status",
                    "value": "✅ Connection successful",
                    "inline": True
                },
                {
                    "name": "Time",
                    "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S EST"),
                    "inline": True
                }
            ],
            "footer": {
                "text": "Wagehood Trading System Test"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info("Sending Discord test message...")
        return self.send_embed(test_embed, "🎯 Testing Wagehood Discord integration!")
    
    def send_signal_notification(self, signal_data: Dict[str, Any]) -> bool:
        """
        Send trading signal notification.
        
        Args:
            signal_data: Signal information dictionary
            
        Returns:
            True if notification sent successfully
        """
        try:
            from .message_formatter import MessageFormatter
            formatter = MessageFormatter()
            
            embed_data = formatter.format_signal_embed(signal_data)
            return self.send_embed(embed_data)
            
        except Exception as e:
            logger.error(f"Error sending signal notification: {e}")
            return False
    
    def send_system_status(self, status_data: Dict[str, Any]) -> bool:
        """
        Send system status notification.
        
        Args:
            status_data: System status information
            
        Returns:
            True if notification sent successfully
        """
        try:
            status_embed = {
                "title": "🚀 Wagehood System Status",
                "color": 3447003,  # Blue
                "fields": [
                    {
                        "name": "Service",
                        "value": status_data.get("service", "Unknown"),
                        "inline": True
                    },
                    {
                        "name": "Status", 
                        "value": status_data.get("status", "Unknown"),
                        "inline": True
                    },
                    {
                        "name": "Symbols Tracking",
                        "value": str(status_data.get("symbol_count", 0)),
                        "inline": True
                    }
                ],
                "footer": {
                    "text": "Wagehood System Monitor"
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return self.send_embed(status_embed)
            
        except Exception as e:
            logger.error(f"Error sending system status: {e}")
            return False