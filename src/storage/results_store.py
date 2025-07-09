"""
Results store for managing backtest results and performance data
"""

import json
import pickle
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

from ..core.models import SignalAnalysisResult, Signal
from .cache import cache_manager, cached

logger = logging.getLogger(__name__)


class ResultsStore:
    """Store and retrieve backtest results with multiple storage backends"""

    def __init__(
        self,
        storage_path: str = "backtest_results",
        storage_type: str = "json",
        enable_cache: bool = True,
        cache_ttl: int = 3600,
    ):
        """
        Initialize results store

        Args:
            storage_path: Path to storage directory or database file
            storage_type: Storage backend type ('json', 'pickle', 'sqlite')
            enable_cache: Whether to enable caching (default True)
            cache_ttl: Cache time-to-live in seconds (default 3600)
        """
        self.storage_path = Path(storage_path)
        self.storage_type = storage_type
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl

        # Cache TTL settings for different result types
        self._cache_ttl_settings = {
            "results": 7200,  # 2 hours for full results
            "summaries": 3600,  # 1 hour for result summaries
            "performance": 3600,  # 1 hour for performance summaries
            "exports": 1800,  # 30 minutes for exports
        }

        # Create storage directory if needed
        if storage_type in ["json", "pickle"]:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize database if using SQLite
        if storage_type == "sqlite":
            self._init_database()

    def _init_database(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        # Create tables
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                initial_capital REAL NOT NULL,
                final_capital REAL NOT NULL,
                created_at TEXT NOT NULL,
                result_data TEXT NOT NULL
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id INTEGER NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                total_pnl REAL NOT NULL,
                total_return_pct REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                max_drawdown_pct REAL NOT NULL,
                sharpe_ratio REAL NOT NULL,
                sortino_ratio REAL NOT NULL,
                profit_factor REAL NOT NULL,
                avg_win REAL NOT NULL,
                avg_loss REAL NOT NULL,
                largest_win REAL NOT NULL,
                largest_loss REAL NOT NULL,
                avg_trade_duration_hours REAL NOT NULL,
                max_consecutive_wins INTEGER NOT NULL,
                max_consecutive_losses INTEGER NOT NULL,
                FOREIGN KEY (backtest_id) REFERENCES backtest_results (id)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id INTEGER NOT NULL,
                trade_id TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                pnl REAL,
                commission REAL NOT NULL,
                strategy_name TEXT NOT NULL,
                signal_metadata TEXT,
                FOREIGN KEY (backtest_id) REFERENCES backtest_results (id)
            )
        """
        )

        # Create indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_strategy_symbol ON backtest_results (strategy_name, symbol)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_created_at ON backtest_results (created_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_performance_metrics ON performance_metrics (backtest_id)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades ON trades (backtest_id)")

        conn.commit()
        conn.close()

    def save_result(
        self, result: SignalAnalysisResult, metadata: Dict[str, Any] = None
    ) -> str:
        """
        Save signal analysis result

        Args:
            result: SignalAnalysisResult to save
            metadata: Additional metadata

        Returns:
            Result ID or filename
        """
        # Save to storage
        if self.storage_type == "json":
            result_id = self._save_json(result, metadata)
        elif self.storage_type == "pickle":
            result_id = self._save_pickle(result, metadata)
        elif self.storage_type == "sqlite":
            result_id = self._save_sqlite(result, metadata)
        else:
            raise ValueError(f"Unsupported storage type: {self.storage_type}")

        # Invalidate related cache entries
        if self.enable_cache:
            self._invalidate_cache_for_result(result, result_id)

        return result_id

    def load_result(self, result_id: str) -> Optional[SignalAnalysisResult]:
        """
        Load signal analysis result

        Args:
            result_id: Result ID or filename

        Returns:
            SignalAnalysisResult or None if not found
        """
        # Try to get from cache first
        if self.enable_cache:
            cache_key = cache_manager.cache_key_hash("result", result_id)
            cached_result = cache_manager.get("results", cache_key)
            if cached_result is not None:
                return cached_result

        # Load from storage
        result = None
        try:
            if self.storage_type == "json":
                result = self._load_json(result_id)
            elif self.storage_type == "pickle":
                result = self._load_pickle(result_id)
            elif self.storage_type == "sqlite":
                result = self._load_sqlite(result_id)
            else:
                raise ValueError(f"Unsupported storage type: {self.storage_type}")

            # Cache the result if found
            if result is not None and self.enable_cache:
                cache_key = cache_manager.cache_key_hash("result", result_id)
                cache_manager.set(
                    "results", cache_key, result, self._cache_ttl_settings["results"]
                )

            return result
        except Exception as e:
            logger.error(f"Error loading result {result_id}: {e}")
            return None

    def list_results(
        self,
        strategy_name: str = None,
        symbol: str = None,
        date_range: Tuple[datetime, datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        List available backtest results

        Args:
            strategy_name: Filter by strategy name
            symbol: Filter by symbol
            date_range: Filter by date range (start, end)

        Returns:
            List of result summaries
        """
        # Try to get from cache first (only if no filters applied)
        if (
            self.enable_cache
            and strategy_name is None
            and symbol is None
            and date_range is None
        ):
            cache_key = cache_manager.cache_key_hash("list_results", "all")
            cached_results = cache_manager.get("results", cache_key)
            if cached_results is not None:
                return cached_results

        # Get from storage
        try:
            if self.storage_type == "json":
                results = self._list_json(strategy_name, symbol, date_range)
            elif self.storage_type == "pickle":
                results = self._list_pickle(strategy_name, symbol, date_range)
            elif self.storage_type == "sqlite":
                results = self._list_sqlite(strategy_name, symbol, date_range)
            else:
                raise ValueError(f"Unsupported storage type: {self.storage_type}")

            # Cache the results if no filters were applied
            if (
                self.enable_cache
                and strategy_name is None
                and symbol is None
                and date_range is None
            ):
                cache_key = cache_manager.cache_key_hash("list_results", "all")
                cache_manager.set(
                    "results", cache_key, results, self._cache_ttl_settings["summaries"]
                )

            return results
        except Exception as e:
            logger.error(f"Error listing results: {e}")
            return []

    def delete_result(self, result_id: str) -> bool:
        """
        Delete backtest result

        Args:
            result_id: Result ID or filename

        Returns:
            True if deleted successfully
        """
        # Delete from storage
        success = False
        try:
            if self.storage_type == "json":
                success = self._delete_json(result_id)
            elif self.storage_type == "pickle":
                success = self._delete_pickle(result_id)
            elif self.storage_type == "sqlite":
                success = self._delete_sqlite(result_id)
            else:
                raise ValueError(f"Unsupported storage type: {self.storage_type}")

            # Invalidate cache entries if deletion was successful
            if success and self.enable_cache:
                self._invalidate_cache_for_deletion(result_id)

            return success
        except Exception as e:
            logger.error(f"Error deleting result {result_id}: {e}")
            return False

    def get_performance_summary(
        self, strategy_name: str = None, symbol: str = None
    ) -> Dict[str, Any]:
        """
        Get performance summary across results

        Args:
            strategy_name: Filter by strategy name
            symbol: Filter by symbol

        Returns:
            Performance summary
        """
        # Try to get from cache first (only if no filters applied)
        if self.enable_cache and strategy_name is None and symbol is None:
            cache_key = cache_manager.cache_key_hash("performance_summary", "all")
            cached_summary = cache_manager.get("results", cache_key)
            if cached_summary is not None:
                return cached_summary

        results = self.list_results(strategy_name, symbol)

        if not results:
            return {}

        # Calculate summary statistics
        returns = [r.get("total_return_pct", 0) for r in results]
        sharpe_ratios = [r.get("sharpe_ratio", 0) for r in results]
        win_rates = [r.get("win_rate", 0) for r in results]

        summary = {
            "total_results": len(results),
            "avg_return_pct": sum(returns) / len(returns) if returns else 0,
            "best_return_pct": max(returns) if returns else 0,
            "worst_return_pct": min(returns) if returns else 0,
            "avg_sharpe_ratio": (
                sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0
            ),
            "best_sharpe_ratio": max(sharpe_ratios) if sharpe_ratios else 0,
            "avg_win_rate": sum(win_rates) / len(win_rates) if win_rates else 0,
            "strategies": list(set(r.get("strategy_name", "") for r in results)),
            "symbols": list(set(r.get("symbol", "") for r in results)),
            "date_range": {
                "start": min(r.get("start_date", "") for r in results),
                "end": max(r.get("end_date", "") for r in results),
            },
        }

        # Cache the summary if no filters were applied
        if self.enable_cache and strategy_name is None and symbol is None:
            cache_key = cache_manager.cache_key_hash("performance_summary", "all")
            cache_manager.set(
                "results", cache_key, summary, self._cache_ttl_settings["performance"]
            )

        return summary

    def export_results(
        self,
        output_file: str,
        strategy_name: str = None,
        symbol: str = None,
        format: str = "json",
    ) -> bool:
        """
        Export results to file

        Args:
            output_file: Output file path
            strategy_name: Filter by strategy name
            symbol: Filter by symbol
            format: Export format ('json', 'csv')

        Returns:
            True if exported successfully
        """
        results = self.list_results(strategy_name, symbol)

        if not results:
            logger.warning("No results to export")
            return False

        try:
            if format == "json":
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2, default=str)
            elif format == "csv":
                import csv

                with open(output_file, "w", newline="") as f:
                    if results:
                        writer = csv.DictWriter(f, fieldnames=results[0].keys())
                        writer.writeheader()
                        writer.writerows(results)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            logger.info(f"Exported {len(results)} results to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False

    # JSON storage implementation
    def _save_json(
        self, result: SignalAnalysisResult, metadata: Dict[str, Any] = None
    ) -> str:
        """Save result as JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.strategy_name}_{result.symbol}_{timestamp}.json"
        filepath = self.storage_path / filename

        # Convert result to dictionary
        result_dict = self._result_to_dict(result)
        if metadata:
            result_dict["metadata"] = metadata

        try:
            with open(filepath, "w") as f:
                json.dump(result_dict, f, indent=2, default=str)

            logger.info(f"Saved result to {filepath}")
            return filename

        except Exception as e:
            logger.error(f"Error saving JSON result: {e}")
            raise

    def _load_json(self, filename: str) -> Optional[SignalAnalysisResult]:
        """Load result from JSON"""
        filepath = self.storage_path / filename

        if not filepath.exists():
            logger.warning(f"Result file not found: {filepath}")
            return None

        try:
            with open(filepath, "r") as f:
                result_dict = json.load(f)

            return self._dict_to_result(result_dict)

        except Exception as e:
            logger.error(f"Error loading JSON result: {e}")
            return None

    def _list_json(
        self, strategy_name: str, symbol: str, date_range: Tuple[datetime, datetime]
    ) -> List[Dict[str, Any]]:
        """List JSON results"""
        results = []

        for filepath in self.storage_path.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    result_dict = json.load(f)

                # Apply filters
                if strategy_name and result_dict.get("strategy_name") != strategy_name:
                    continue
                if symbol and result_dict.get("symbol") != symbol:
                    continue

                # Extract summary
                summary = {
                    "id": filepath.name,
                    "strategy_name": result_dict.get("strategy_name"),
                    "symbol": result_dict.get("symbol"),
                    "start_date": result_dict.get("start_date"),
                    "end_date": result_dict.get("end_date"),
                    "initial_capital": result_dict.get("initial_capital"),
                    "final_capital": result_dict.get("final_capital"),
                    "total_return_pct": result_dict.get("performance_metrics", {}).get(
                        "total_return_pct", 0
                    ),
                    "sharpe_ratio": result_dict.get("performance_metrics", {}).get(
                        "sharpe_ratio", 0
                    ),
                    "win_rate": result_dict.get("performance_metrics", {}).get(
                        "win_rate", 0
                    ),
                    "total_trades": result_dict.get("performance_metrics", {}).get(
                        "total_trades", 0
                    ),
                }

                results.append(summary)

            except Exception as e:
                logger.error(f"Error reading result file {filepath}: {e}")
                continue

        return results

    def _delete_json(self, filename: str) -> bool:
        """Delete JSON result"""
        filepath = self.storage_path / filename

        try:
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Deleted result file: {filepath}")
                return True
            else:
                logger.warning(f"Result file not found: {filepath}")
                return False

        except Exception as e:
            logger.error(f"Error deleting result file: {e}")
            return False

    # Pickle storage implementation
    def _save_pickle(
        self, result: SignalAnalysisResult, metadata: Dict[str, Any] = None
    ) -> str:
        """Save result as pickle"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.strategy_name}_{result.symbol}_{timestamp}.pkl"
        filepath = self.storage_path / filename

        data = {
            "result": result,
            "metadata": metadata or {},
            "created_at": datetime.now(),
        }

        try:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)

            logger.info(f"Saved result to {filepath}")
            return filename

        except Exception as e:
            logger.error(f"Error saving pickle result: {e}")
            raise

    def _load_pickle(self, filename: str) -> Optional[SignalAnalysisResult]:
        """Load result from pickle"""
        filepath = self.storage_path / filename

        if not filepath.exists():
            logger.warning(f"Result file not found: {filepath}")
            return None

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)

            return data["result"]

        except Exception as e:
            logger.error(f"Error loading pickle result: {e}")
            return None

    def _list_pickle(
        self, strategy_name: str, symbol: str, date_range: Tuple[datetime, datetime]
    ) -> List[Dict[str, Any]]:
        """List pickle results"""
        results = []

        for filepath in self.storage_path.glob("*.pkl"):
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)

                result = data["result"]

                # Apply filters
                if strategy_name and result.strategy_name != strategy_name:
                    continue
                if symbol and result.symbol != symbol:
                    continue

                # Extract summary
                summary = {
                    "id": filepath.name,
                    "strategy_name": result.strategy_name,
                    "symbol": result.symbol,
                    "start_date": result.start_date.isoformat(),
                    "end_date": result.end_date.isoformat(),
                    "initial_capital": result.initial_capital,
                    "final_capital": result.final_capital,
                    "total_return_pct": result.performance_metrics.total_return_pct,
                    "sharpe_ratio": result.performance_metrics.sharpe_ratio,
                    "win_rate": result.performance_metrics.win_rate,
                    "total_trades": result.performance_metrics.total_trades,
                }

                results.append(summary)

            except Exception as e:
                logger.error(f"Error reading result file {filepath}: {e}")
                continue

        return results

    def _delete_pickle(self, filename: str) -> bool:
        """Delete pickle result"""
        return self._delete_json(filename)  # Same implementation

    # SQLite storage implementation
    def _save_sqlite(
        self, result: SignalAnalysisResult, metadata: Dict[str, Any] = None
    ) -> str:
        """Save result to SQLite database"""
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        try:
            # Insert main result
            cursor.execute(
                """
                INSERT INTO backtest_results 
                (strategy_name, symbol, start_date, end_date, initial_capital, final_capital, created_at, result_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result.strategy_name,
                    result.symbol,
                    result.start_date.isoformat(),
                    result.end_date.isoformat(),
                    result.initial_capital,
                    result.final_capital,
                    datetime.now().isoformat(),
                    json.dumps(self._result_to_dict(result), default=str),
                ),
            )

            backtest_id = cursor.lastrowid

            # Insert performance metrics
            metrics = result.performance_metrics
            cursor.execute(
                """
                INSERT INTO performance_metrics 
                (backtest_id, total_trades, winning_trades, losing_trades, win_rate, total_pnl, 
                 total_return_pct, max_drawdown, max_drawdown_pct, sharpe_ratio, sortino_ratio, 
                 profit_factor, avg_win, avg_loss, largest_win, largest_loss, avg_trade_duration_hours,
                 max_consecutive_wins, max_consecutive_losses)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    backtest_id,
                    metrics.total_trades,
                    metrics.winning_trades,
                    metrics.losing_trades,
                    metrics.win_rate,
                    metrics.total_pnl,
                    metrics.total_return_pct,
                    metrics.max_drawdown,
                    metrics.max_drawdown_pct,
                    metrics.sharpe_ratio,
                    metrics.sortino_ratio,
                    metrics.profit_factor,
                    metrics.avg_win,
                    metrics.avg_loss,
                    metrics.largest_win,
                    metrics.largest_loss,
                    metrics.avg_trade_duration_hours,
                    metrics.max_consecutive_wins,
                    metrics.max_consecutive_losses,
                ),
            )

            # Insert trades
            for trade in result.trades:
                cursor.execute(
                    """
                    INSERT INTO trades 
                    (backtest_id, trade_id, entry_time, exit_time, symbol, quantity, entry_price, 
                     exit_price, pnl, commission, strategy_name, signal_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        backtest_id,
                        trade.trade_id,
                        trade.entry_time.isoformat(),
                        trade.exit_time.isoformat() if trade.exit_time else None,
                        trade.symbol,
                        trade.quantity,
                        trade.entry_price,
                        trade.exit_price,
                        trade.pnl,
                        trade.commission,
                        trade.strategy_name,
                        json.dumps(trade.signal_metadata, default=str),
                    ),
                )

            conn.commit()
            logger.info(f"Saved result to database with ID: {backtest_id}")
            return str(backtest_id)

        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving SQLite result: {e}")
            raise
        finally:
            conn.close()

    def _load_sqlite(self, result_id: str) -> Optional[SignalAnalysisResult]:
        """Load result from SQLite database"""
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        try:
            # Get main result
            cursor.execute(
                "SELECT result_data FROM backtest_results WHERE id = ?", (result_id,)
            )
            row = cursor.fetchone()

            if not row:
                logger.warning(f"Result not found with ID: {result_id}")
                return None

            result_dict = json.loads(row[0])
            return self._dict_to_result(result_dict)

        except Exception as e:
            logger.error(f"Error loading SQLite result: {e}")
            return None
        finally:
            conn.close()

    def _list_sqlite(
        self, strategy_name: str, symbol: str, date_range: Tuple[datetime, datetime]
    ) -> List[Dict[str, Any]]:
        """List SQLite results"""
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        try:
            # Build query
            query = """
                SELECT r.id, r.strategy_name, r.symbol, r.start_date, r.end_date, 
                       r.initial_capital, r.final_capital, 
                       m.total_return_pct, m.sharpe_ratio, m.win_rate, m.total_trades
                FROM backtest_results r
                LEFT JOIN performance_metrics m ON r.id = m.backtest_id
                WHERE 1=1
            """
            params = []

            if strategy_name:
                query += " AND r.strategy_name = ?"
                params.append(strategy_name)

            if symbol:
                query += " AND r.symbol = ?"
                params.append(symbol)

            query += " ORDER BY r.created_at DESC"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                summary = {
                    "id": str(row[0]),
                    "strategy_name": row[1],
                    "symbol": row[2],
                    "start_date": row[3],
                    "end_date": row[4],
                    "initial_capital": row[5],
                    "final_capital": row[6],
                    "total_return_pct": row[7] or 0,
                    "sharpe_ratio": row[8] or 0,
                    "win_rate": row[9] or 0,
                    "total_trades": row[10] or 0,
                }
                results.append(summary)

            return results

        except Exception as e:
            logger.error(f"Error listing SQLite results: {e}")
            return []
        finally:
            conn.close()

    def _delete_sqlite(self, result_id: str) -> bool:
        """Delete result from SQLite database"""
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        try:
            # Delete trades first (foreign key constraint)
            cursor.execute("DELETE FROM trades WHERE backtest_id = ?", (result_id,))

            # Delete performance metrics
            cursor.execute(
                "DELETE FROM performance_metrics WHERE backtest_id = ?", (result_id,)
            )

            # Delete main result
            cursor.execute("DELETE FROM backtest_results WHERE id = ?", (result_id,))

            deleted = cursor.rowcount > 0
            conn.commit()

            if deleted:
                logger.info(f"Deleted result with ID: {result_id}")
            else:
                logger.warning(f"Result not found with ID: {result_id}")

            return deleted

        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting SQLite result: {e}")
            return False
        finally:
            conn.close()

    # Helper methods
    def _result_to_dict(self, result: SignalAnalysisResult) -> Dict[str, Any]:
        """Convert SignalAnalysisResult to dictionary"""
        return {
            "strategy_name": result.strategy_name,
            "symbol": result.symbol,
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "signals": [self._signal_to_dict(signal) for signal in result.signals],
            "signal_timestamps": [ts.isoformat() for ts in result.signal_timestamps],
            "total_signals": result.total_signals,
            "buy_signals": result.buy_signals,
            "sell_signals": result.sell_signals,
            "hold_signals": result.hold_signals,
            "avg_confidence": result.avg_confidence,
            "signal_frequency": result.signal_frequency,
            "metadata": result.metadata,
        }

    def _dict_to_result(self, result_dict: Dict[str, Any]) -> SignalAnalysisResult:
        """Convert dictionary to SignalAnalysisResult"""
        return SignalAnalysisResult(
            strategy_name=result_dict["strategy_name"],
            symbol=result_dict["symbol"],
            start_date=datetime.fromisoformat(result_dict["start_date"]),
            end_date=datetime.fromisoformat(result_dict["end_date"]),
            signals=[
                self._dict_to_signal(signal_dict)
                for signal_dict in result_dict["signals"]
            ],
            signal_timestamps=[
                datetime.fromisoformat(ts) for ts in result_dict["signal_timestamps"]
            ],
            total_signals=result_dict["total_signals"],
            buy_signals=result_dict["buy_signals"],
            sell_signals=result_dict["sell_signals"],
            hold_signals=result_dict["hold_signals"],
            avg_confidence=result_dict["avg_confidence"],
            signal_frequency=result_dict["signal_frequency"],
            metadata=result_dict["metadata"],
        )

    def _signal_to_dict(self, signal: Signal) -> Dict[str, Any]:
        """Convert Signal to dictionary"""
        return {
            "timestamp": signal.timestamp.isoformat(),
            "symbol": signal.symbol,
            "signal_type": signal.signal_type.value,
            "price": signal.price,
            "confidence": signal.confidence,
            "strategy_name": signal.strategy_name,
            "metadata": signal.metadata,
        }

    def _dict_to_signal(self, signal_dict: Dict[str, Any]) -> Signal:
        """Convert dictionary to Signal"""
        from ..core.models import SignalType

        return Signal(
            timestamp=datetime.fromisoformat(signal_dict["timestamp"]),
            symbol=signal_dict["symbol"],
            signal_type=SignalType(signal_dict["signal_type"]),
            price=signal_dict["price"],
            confidence=signal_dict["confidence"],
            strategy_name=signal_dict["strategy_name"],
            metadata=signal_dict["metadata"],
        )

    def _invalidate_cache_for_result(
        self, result: SignalAnalysisResult, result_id: str
    ) -> None:
        """
        Invalidate cache entries when a result is saved.

        Args:
            result: The backtest result that was saved
            result_id: The ID of the saved result
        """
        if not self.enable_cache:
            return

        try:
            # Invalidate list results cache
            cache_manager.delete(
                "results", cache_manager.cache_key_hash("list_results", "all")
            )

            # Invalidate performance summary cache
            cache_manager.delete(
                "results", cache_manager.cache_key_hash("performance_summary", "all")
            )

            # Invalidate specific result cache if it exists
            cache_manager.delete(
                "results", cache_manager.cache_key_hash("result", result_id)
            )

            logger.debug(f"Invalidated cache entries for result {result_id}")

        except Exception as e:
            logger.warning(f"Failed to invalidate cache for result {result_id}: {e}")

    def _invalidate_cache_for_deletion(self, result_id: str) -> None:
        """
        Invalidate cache entries when a result is deleted.

        Args:
            result_id: The ID of the deleted result
        """
        if not self.enable_cache:
            return

        try:
            # Invalidate list results cache
            cache_manager.delete(
                "results", cache_manager.cache_key_hash("list_results", "all")
            )

            # Invalidate performance summary cache
            cache_manager.delete(
                "results", cache_manager.cache_key_hash("performance_summary", "all")
            )

            # Invalidate specific result cache
            cache_manager.delete(
                "results", cache_manager.cache_key_hash("result", result_id)
            )

            logger.debug(f"Invalidated cache entries for deleted result {result_id}")

        except Exception as e:
            logger.warning(
                f"Failed to invalidate cache for deleted result {result_id}: {e}"
            )

    def clear_cache(self) -> None:
        """Clear all cached results."""
        if not self.enable_cache:
            return

        try:
            cache_manager.clear_namespace("results")
            logger.info("Cleared all results cache")
        except Exception as e:
            logger.warning(f"Failed to clear results cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics if caching is enabled
        """
        if not self.enable_cache:
            return {"caching_enabled": False}

        try:
            stats = cache_manager.get_stats()
            stats["caching_enabled"] = True
            stats["cache_ttl_settings"] = self._cache_ttl_settings
            return stats
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            return {"caching_enabled": True, "error": str(e)}
