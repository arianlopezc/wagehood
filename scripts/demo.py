#!/usr/bin/env python3
"""
Wagehood Trading System Demo Script

This script demonstrates the complete functionality of the trading system:
1. Generate realistic market data
2. Calculate technical indicators
3. Run all 5 strategies
4. Perform backtesting
5. Compare strategy performance
6. Show recommendations
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import TimeFrame
from data import DataStore, MockDataGenerator
from indicators import IndicatorCalculator
from strategies import (
    MovingAverageCrossover,
    MACDRSIStrategy,
    RSITrendFollowing,
    BollingerBandBreakout,
    SupportResistanceBreakout,
    STRATEGY_REGISTRY
)
from backtest import BacktestEngine
from analysis import PerformanceEvaluator, StrategyComparator
from storage import ResultsStore


def print_banner():
    """Print demo banner"""
    print("=" * 80)
    print("🚀 WAGEHOOD TRADING SYSTEM DEMO")
    print("=" * 80)
    print("Demonstrating 5 proven trend-following strategies")
    print("Based on 2024 research with documented performance")
    print("-" * 80)


def generate_demo_data() -> Dict[str, Any]:
    """Generate diverse market scenarios for testing"""
    print("\n📊 GENERATING MARKET DATA")
    print("-" * 40)
    
    generator = MockDataGenerator()
    store = DataStore()
    
    # Different market scenarios
    scenarios = {
        "SPY_BULL": {
            "symbol": "SPY",
            "periods": 252,
            "patterns": {"trend": "bullish", "volatility": 0.015}
        },
        "BTC_VOLATILE": {
            "symbol": "BTC-USD", 
            "periods": 252,
            "patterns": {"trend": "mixed", "volatility": 0.04}
        },
        "EURUSD_RANGING": {
            "symbol": "EUR/USD",
            "periods": 252, 
            "patterns": {"trend": "sideways", "volatility": 0.008}
        }
    }
    
    data_sets = {}
    for name, config in scenarios.items():
        print(f"  • {name}: {config['patterns']['trend']} market, {config['patterns']['volatility']:.1%} volatility")
        data = generator.generate_realistic_data(
            config["symbol"],
            config["periods"],
            config["patterns"]
        )
        store.store_ohlcv(config["symbol"], TimeFrame.DAILY, data)
        data_sets[name] = data
    
    print(f"✅ Generated {len(scenarios)} market scenarios with {sum(len(d) for d in data_sets.values())} data points")
    return data_sets


def calculate_indicators_demo(data_sets: Dict[str, Any]):
    """Demonstrate indicator calculations"""
    print("\n🔧 CALCULATING TECHNICAL INDICATORS")
    print("-" * 40)
    
    calc = IndicatorCalculator()
    
    # Use SPY data for indicator demo
    spy_data = data_sets["SPY_BULL"]
    closes = [bar.close for bar in spy_data]
    
    indicators = {
        "SMA_50": calc.calculate_sma(closes, 50),
        "EMA_200": calc.calculate_ema(closes, 200),
        "RSI": calc.calculate_rsi(closes, 14),
        "MACD": calc.calculate_macd(closes),
        "BB": calc.calculate_bollinger_bands(closes, 20, 2.0)
    }
    
    print(f"  • Moving Averages: SMA(50), EMA(200)")
    print(f"  • Momentum: RSI(14), MACD(12,26,9)")
    print(f"  • Volatility: Bollinger Bands(20,2)")
    print(f"  • Support/Resistance: Dynamic level detection")
    
    # Show latest values
    print(f"\n📈 Latest Indicator Values (SPY):")
    print(f"  • Current Price: ${closes[-1]:.2f}")
    print(f"  • RSI: {indicators['RSI'][-1]:.1f}")
    print(f"  • MACD: {indicators['MACD'][0][-1]:.3f}")
    print(f"  • Upper BB: ${indicators['BB'][0][-1]:.2f}")
    
    return indicators


def run_strategy_comparison(data_sets: Dict[str, Any]):
    """Run all strategies and compare performance"""
    print("\n⚡ STRATEGY PERFORMANCE COMPARISON")
    print("-" * 40)
    
    # Initialize components
    engine = BacktestEngine()
    evaluator = PerformanceEvaluator()
    comparator = StrategyComparator()
    
    strategies = [
        MovingAverageCrossover(),
        MACDRSIStrategy(), 
        RSITrendFollowing(),
        BollingerBandBreakout(),
        SupportResistanceBreakout()
    ]
    
    results = {}
    
    # Test each strategy on SPY bull market
    spy_data = data_sets["SPY_BULL"]
    
    print(f"Testing {len(strategies)} strategies on SPY bull market (252 days)...")
    print()
    
    for strategy in strategies:
        try:
            result = engine.run_backtest(
                strategy=strategy,
                data=spy_data,
                initial_capital=10000.0
                # Commission-free by default - no need to specify commission_rate=0.0
            )
            results[strategy.__class__.__name__] = result
            
            metrics = result.performance_metrics
            print(f"🎯 {strategy.__class__.__name__}:")
            print(f"   📊 Total Return: {metrics.total_return_pct:.1%}")
            print(f"   🎯 Win Rate: {metrics.win_rate:.1%}")
            print(f"   📈 Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"   📉 Max Drawdown: {metrics.max_drawdown_pct:.1%}")
            print(f"   🔢 Total Trades: {metrics.total_trades}")
            print()
            
        except Exception as e:
            print(f"❌ {strategy.__class__.__name__}: Error - {e}")
            print()
    
    return results


def show_best_strategy(results: Dict[str, Any]):
    """Show the best performing strategy"""
    print("\n🏆 BEST STRATEGY RECOMMENDATION")
    print("-" * 40)
    
    if not results:
        print("❌ No valid results to analyze")
        return
    
    # Rank by Sharpe ratio
    ranked = sorted(
        results.items(),
        key=lambda x: x[1].performance_metrics.sharpe_ratio,
        reverse=True
    )
    
    winner = ranked[0]
    strategy_name = winner[0]
    metrics = winner[1].performance_metrics
    
    print(f"🥇 Winner: {strategy_name}")
    print(f"   💰 Return: {metrics.total_return_pct:.1%}")
    print(f"   📊 Sharpe: {metrics.sharpe_ratio:.2f}")
    print(f"   🎯 Win Rate: {metrics.win_rate:.1%}")
    print(f"   ⚡ Profit Factor: {metrics.profit_factor:.2f}")
    
    print(f"\n📋 Full Rankings (by Sharpe Ratio):")
    for i, (name, result) in enumerate(ranked, 1):
        metrics = result.performance_metrics
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"   {emoji} {i}. {name}: {metrics.sharpe_ratio:.2f}")


def show_research_validation():
    """Show how results compare to research"""
    print("\n📚 RESEARCH VALIDATION")
    print("-" * 40)
    print("Our strategies implement proven configurations with commission-free trading:")
    print()
    print("📈 Moving Average Crossover (50/200):")
    print("   • Research: 50% drawdown reduction")
    print("   • Timeframe: 350-day average hold")
    print()
    print("🎯 MACD+RSI Strategy:")
    print("   • Research: 73% win rate documented")
    print("   • Parameters: MACD(12,26,9) + RSI(14)")
    print()
    print("⚡ RSI Trend Following:")
    print("   • Research: Most reliable indicator 2024")
    print("   • Application: Trend confirmation + pullbacks")
    print()
    print("💫 Bollinger Band Breakout:")
    print("   • Research: High reliability in 2024 studies")
    print("   • Parameters: BB(20,2) with volume confirmation")
    print()
    print("🔥 Support/Resistance Breakout:")
    print("   • Research: Best for major level breaks")
    print("   • Application: Advanced pattern recognition")


def main():
    """Main demo function"""
    print_banner()
    
    try:
        # Step 1: Generate market data
        data_sets = generate_demo_data()
        
        # Step 2: Calculate indicators
        indicators = calculate_indicators_demo(data_sets)
        
        # Step 3: Run strategy comparison
        results = run_strategy_comparison(data_sets)
        
        # Step 4: Show best strategy
        show_best_strategy(results)
        
        # Step 5: Research validation
        show_research_validation()
        
        print("\n" + "=" * 80)
        print("🎉 DEMO COMPLETE!")
        print("=" * 80)
        print("🚀 Next Steps:")
        print("   • Start API: python -m src.api.app")
        print("   • Run tests: make test")
        print("   • See docs: http://localhost:8000/docs")
        print("   • Explore: .local/ folder for research")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Check that all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()