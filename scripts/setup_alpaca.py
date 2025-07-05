#!/usr/bin/env python3
"""
Alpaca Setup and Testing Script

This script helps set up Alpaca Markets integration for the trading system,
including paper trading account validation and basic functionality testing.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.providers.alpaca_provider import AlpacaProvider
from trading.alpaca_client import AlpacaTradingClient, AlpacaOrderSide
from core.models import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_alpaca_data_provider():
    """Test Alpaca data provider functionality."""
    logger.info("Testing Alpaca Data Provider...")
    
    try:
        # Create provider with paper trading config
        config = {
            'paper': True,
            'feed': 'iex',  # Free tier
            'max_retries': 3
        }
        
        provider = AlpacaProvider(config)
        
        # Test connection
        logger.info("Connecting to Alpaca Markets...")
        connected = await provider.connect()
        
        if not connected:
            logger.error("Failed to connect to Alpaca Markets")
            return False
        
        logger.info("✅ Connected successfully!")
        
        # Test historical data retrieval
        logger.info("Testing historical data retrieval...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        historical_data = await provider.get_historical_data(
            symbol="AAPL",
            timeframe=TimeFrame.DAILY,
            start_date=start_date,
            end_date=end_date,
            limit=10
        )
        
        if historical_data:
            logger.info(f"✅ Retrieved {len(historical_data)} historical data points for AAPL")
            latest = historical_data[-1]
            logger.info(f"   Latest: {latest.timestamp} - Close: ${latest.close:.2f}")
        else:
            logger.warning("⚠️  No historical data retrieved")
        
        # Test latest data
        logger.info("Testing latest data retrieval...")
        
        latest_data = await provider.get_latest_data(
            symbol="SPY",
            timeframe=TimeFrame.DAILY,
            periods=3
        )
        
        if latest_data:
            logger.info(f"✅ Retrieved {len(latest_data)} latest data points for SPY")
        else:
            logger.warning("⚠️  No latest data retrieved")
        
        # Test crypto data
        logger.info("Testing crypto data...")
        
        crypto_data = await provider.get_historical_data(
            symbol="BTC/USD",
            timeframe=TimeFrame.DAILY,
            start_date=start_date,
            end_date=end_date,
            limit=5
        )
        
        if crypto_data:
            logger.info(f"✅ Retrieved {len(crypto_data)} crypto data points for BTC/USD")
            latest_btc = crypto_data[-1]
            logger.info(f"   Latest BTC: {latest_btc.timestamp} - Close: ${latest_btc.close:.2f}")
        else:
            logger.warning("⚠️  No crypto data retrieved")
        
        # Test provider info
        info = provider.get_provider_info()
        logger.info(f"Provider info: {info}")
        
        # Disconnect
        await provider.disconnect()
        logger.info("✅ Data provider test completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing data provider: {e}")
        return False


async def test_alpaca_trading_client():
    """Test Alpaca trading client functionality."""
    logger.info("Testing Alpaca Trading Client...")
    
    try:
        # Create trading client with paper trading
        config = {
            'paper': True,
            'max_retries': 3
        }
        
        client = AlpacaTradingClient(config)
        
        # Test connection
        logger.info("Connecting to Alpaca Trading...")
        connected = await client.connect()
        
        if not connected:
            logger.error("Failed to connect to Alpaca Trading")
            return False
        
        logger.info("✅ Connected successfully!")
        
        # Test account information
        logger.info("Testing account information...")
        
        account = await client.get_account()
        logger.info(f"✅ Account ID: {account['id']}")
        logger.info(f"   Status: {account['status']}")
        logger.info(f"   Buying Power: ${account['buying_power']:,.2f}")
        logger.info(f"   Cash: ${account['cash']:,.2f}")
        logger.info(f"   Portfolio Value: ${account['portfolio_value']:,.2f}")
        logger.info(f"   Pattern Day Trader: {account['pattern_day_trader']}")
        
        # Test positions
        logger.info("Testing positions...")
        
        positions = await client.get_positions()
        logger.info(f"✅ Found {len(positions)} positions")
        
        for position in positions[:3]:  # Show first 3 positions
            logger.info(f"   {position['symbol']}: {position['qty']} shares, "
                       f"Market Value: ${position['market_value']:,.2f}, "
                       f"P&L: ${position['unrealized_pl']:,.2f}")
        
        # Test orders
        logger.info("Testing orders retrieval...")
        
        orders = await client.get_orders(limit=5)
        logger.info(f"✅ Found {len(orders)} recent orders")
        
        for order in orders[:3]:  # Show first 3 orders
            logger.info(f"   {order['symbol']}: {order['side']} {order['qty']} @ {order['order_type']}, "
                       f"Status: {order['status']}")
        
        # Test paper trading order (small amount)
        logger.info("Testing paper trading order placement...")
        
        try:
            # Place a small market order for 1 share of SPY (using paper money)
            order_info = await client.place_market_order(
                symbol="SPY",
                quantity=1,
                side=AlpacaOrderSide.BUY
            )
            
            logger.info(f"✅ Paper order placed successfully!")
            logger.info(f"   Order ID: {order_info['id']}")
            logger.info(f"   Symbol: {order_info['symbol']}")
            logger.info(f"   Side: {order_info['side']}")
            logger.info(f"   Quantity: {order_info['qty']}")
            logger.info(f"   Status: {order_info['status']}")
            
            # Wait a moment then check order status
            await asyncio.sleep(2)
            
            updated_order = await client.get_order(order_info['id'])
            if updated_order:
                logger.info(f"   Updated Status: {updated_order['status']}")
                if updated_order['filled_qty']:
                    logger.info(f"   Filled Quantity: {updated_order['filled_qty']}")
                    logger.info(f"   Fill Price: ${updated_order['filled_avg_price']:.2f}")
            
            # Cancel the order if it's still open
            if updated_order and updated_order['status'] in ['new', 'partially_filled', 'pending_new']:
                cancelled = await client.cancel_order(order_info['id'])
                if cancelled:
                    logger.info("   ✅ Order cancelled successfully")
        
        except Exception as e:
            logger.warning(f"⚠️  Paper order test failed (this is normal if markets are closed): {e}")
        
        # Disconnect
        await client.disconnect()
        logger.info("✅ Trading client test completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing trading client: {e}")
        return False


async def check_environment():
    """Check environment configuration."""
    logger.info("Checking environment configuration...")
    
    # Check for API keys
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        logger.warning("⚠️  Alpaca API credentials not found in environment variables")
        logger.info("To set up Alpaca integration:")
        logger.info("1. Go to https://app.alpaca.markets/")
        logger.info("2. Create a paper trading account (free)")
        logger.info("3. Generate API keys in the dashboard")
        logger.info("4. Set environment variables:")
        logger.info("   export ALPACA_API_KEY='your_api_key_here'")
        logger.info("   export ALPACA_SECRET_KEY='your_secret_key_here'")
        logger.info("5. Or copy .env.example to .env and fill in your keys")
        return False
    
    logger.info("✅ Alpaca API credentials found")
    logger.info(f"   API Key: {api_key[:8]}...")
    
    # Check other environment variables
    paper_trading = os.getenv('ALPACA_PAPER_TRADING', 'true').lower() == 'true'
    data_feed = os.getenv('ALPACA_DATA_FEED', 'iex')
    
    logger.info(f"   Paper Trading: {paper_trading}")
    logger.info(f"   Data Feed: {data_feed}")
    
    if not paper_trading:
        logger.warning("⚠️  Live trading is enabled! Make sure this is intentional.")
        logger.warning("   For testing, set ALPACA_PAPER_TRADING=true")
    
    return True


async def main():
    """Main setup and testing function."""
    logger.info("🚀 Starting Alpaca Setup and Testing...")
    logger.info("=" * 60)
    
    # Check environment
    if not await check_environment():
        logger.error("❌ Environment check failed. Please set up Alpaca credentials.")
        return 1
    
    logger.info("\n" + "=" * 60)
    
    # Test data provider
    data_success = await test_alpaca_data_provider()
    
    logger.info("\n" + "=" * 60)
    
    # Test trading client
    trading_success = await test_alpaca_trading_client()
    
    logger.info("\n" + "=" * 60)
    
    # Summary
    if data_success and trading_success:
        logger.info("🎉 All tests passed! Alpaca integration is working correctly.")
        logger.info("\nNext steps:")
        logger.info("1. Update your .env file with the correct API keys")
        logger.info("2. Run the real-time data ingestion: python run_realtime.py")
        logger.info("3. Test the API endpoints: python -m uvicorn src.api.app:app --reload")
        logger.info("4. Use the CLI tool: ./wagehood_cli.py --help")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)