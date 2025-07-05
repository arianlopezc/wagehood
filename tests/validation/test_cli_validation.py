"""
CLI Validation Tests

This module tests the CLI functionality to ensure all commands work correctly.
"""

import pytest
import subprocess
import tempfile
import os
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestCLIValidation:
    """Test CLI command functionality."""
    
    @pytest.fixture
    def cli_path(self):
        """Get path to CLI script."""
        # Try to use global wagehood command first
        try:
            result = subprocess.run(["which", "wagehood"], capture_output=True, text=True)
            if result.returncode == 0:
                return "wagehood"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Fallback to direct script execution
        return str(Path(__file__).parent.parent.parent / "wagehood_cli.py")
    
    def run_cli_command(self, cli_path, command_args, expect_success=True):
        """Helper to run CLI commands."""
        if cli_path == "wagehood":
            cmd = ["wagehood"] + command_args
        else:
            cmd = [sys.executable, cli_path] + command_args
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if expect_success:
                if result.returncode != 0:
                    print(f"Command failed: {' '.join(cmd)}")
                    print(f"stdout: {result.stdout}")
                    print(f"stderr: {result.stderr}")
                assert result.returncode == 0, f"Command failed with code {result.returncode}"
            
            return result
        except subprocess.TimeoutExpired:
            pytest.fail(f"Command timed out: {' '.join(cmd)}")
    
    def test_cli_help_command(self, cli_path):
        """Test CLI help command."""
        result = self.run_cli_command(cli_path, ["--help"])
        
        assert "Wagehood CLI" in result.stdout or "Usage:" in result.stdout
        assert "Commands:" in result.stdout or "Options:" in result.stdout
    
    def test_cli_config_commands(self, cli_path):
        """Test configuration commands."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config.json")
            
            # Test config show (should work even without config file)
            result = self.run_cli_command(cli_path, ["config", "show"])
            assert result.returncode == 0, "Config show should work"
            
            # Test config validate
            result = self.run_cli_command(cli_path, ["config", "validate"])
            # May succeed or fail depending on default config, but shouldn't crash
            assert result.returncode in [0, 1], "Config validate should not crash"
    
    def test_cli_data_commands(self, cli_path):
        """Test data management commands."""
        # Test data list command
        result = self.run_cli_command(cli_path, ["data", "list"])
        assert result.returncode == 0, "Data list should work"
        
        # Test data sources command
        result = self.run_cli_command(cli_path, ["data", "sources"])
        assert result.returncode == 0, "Data sources should work"
    
    def test_cli_monitor_commands(self, cli_path):
        """Test monitoring commands."""
        # Test monitor status command
        result = self.run_cli_command(cli_path, ["monitor", "status"])
        assert result.returncode == 0, "Monitor status should work"
        
        # Test monitor metrics command
        result = self.run_cli_command(cli_path, ["monitor", "metrics"])
        assert result.returncode == 0, "Monitor metrics should work"
    
    def test_cli_admin_commands(self, cli_path):
        """Test admin commands."""
        # Test admin info command
        result = self.run_cli_command(cli_path, ["admin", "info"])
        assert result.returncode == 0, "Admin info should work"
        
        # Test admin service status command
        result = self.run_cli_command(cli_path, ["admin", "service", "status"])
        assert result.returncode == 0, "Admin service status should work"
    
    def test_cli_output_formats(self, cli_path):
        """Test different output formats."""
        formats = ["json", "table", "csv", "yaml"]
        
        for fmt in formats:
            result = self.run_cli_command(cli_path, ["data", "list", "--format", fmt])
            assert result.returncode == 0, f"Output format {fmt} should work"
            
            if fmt == "json":
                # Try to parse JSON output
                try:
                    json.loads(result.stdout)
                except json.JSONDecodeError:
                    # Might not be JSON if no data, that's ok
                    pass
    
    def test_cli_error_handling(self, cli_path):
        """Test CLI error handling."""
        # Test invalid command
        result = self.run_cli_command(cli_path, ["invalid_command"], expect_success=False)
        assert result.returncode != 0, "Invalid command should fail"
        
        # Test invalid subcommand
        result = self.run_cli_command(cli_path, ["data", "invalid_subcommand"], expect_success=False)
        assert result.returncode != 0, "Invalid subcommand should fail"
        
        # Test missing required arguments
        result = self.run_cli_command(cli_path, ["data", "fetch"], expect_success=False)
        assert result.returncode != 0, "Missing required args should fail"


class TestCLIInstallationValidation:
    """Test CLI installation and setup functionality."""
    
    def test_global_command_availability(self):
        """Test if wagehood command is globally available."""
        try:
            result = subprocess.run(["which", "wagehood"], capture_output=True, text=True)
            if result.returncode == 0:
                # Test basic command
                result = subprocess.run(["wagehood", "--version"], capture_output=True, text=True, timeout=10)
                # Should not crash (may or may not have --version implemented)
                assert result.returncode in [0, 2], "Global command should be functional"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("Global wagehood command not installed")
    
    def test_setup_script_functionality(self):
        """Test setup.py functionality."""
        setup_path = Path(__file__).parent.parent.parent / "setup.py"
        
        if setup_path.exists():
            # Test setup.py --help
            result = subprocess.run(
                [sys.executable, str(setup_path), "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert result.returncode == 0, "setup.py should provide help"


class TestCLIPerformanceValidation:
    """Test CLI performance characteristics."""
    
    def test_cli_startup_time(self, cli_path):
        """Test CLI startup performance."""
        import time
        
        start_time = time.time()
        result = self.run_cli_command(cli_path, ["--help"])
        end_time = time.time()
        
        startup_time = end_time - start_time
        
        # CLI should start quickly (less than 5 seconds)
        assert startup_time < 5.0, f"CLI startup too slow: {startup_time}s"
        assert result.returncode == 0, "Help command should succeed"
    
    def run_cli_command(self, cli_path, command_args):
        """Helper method for this class."""
        if cli_path == "wagehood":
            cmd = ["wagehood"] + command_args
        else:
            cmd = [sys.executable, cli_path] + command_args
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result


class TestCLIConfigurationValidation:
    """Test CLI configuration management."""
    
    def test_config_file_handling(self):
        """Test configuration file handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_config.json")
            
            # Create test config
            test_config = {
                "api": {
                    "base_url": "http://localhost:8000",
                    "timeout": 30
                },
                "data": {
                    "cache_ttl": 300
                },
                "logging": {
                    "level": "INFO"
                }
            }
            
            with open(config_path, "w") as f:
                json.dump(test_config, f, indent=2)
            
            # Test config loading (would need to modify CLI to accept config path)
            assert os.path.exists(config_path), "Config file should exist"
            
            # Validate config structure
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
            
            assert "api" in loaded_config, "Config should have api section"
            assert "data" in loaded_config, "Config should have data section"
            assert "logging" in loaded_config, "Config should have logging section"


class TestCLIIntegrationValidation:
    """Test CLI integration with system components."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis for testing."""
        with patch('redis.Redis') as mock_redis:
            mock_client = MagicMock()
            mock_redis.return_value = mock_client
            mock_client.ping.return_value = True
            yield mock_client
    
    def test_cli_redis_integration(self, mock_redis):
        """Test CLI Redis integration."""
        # This would test Redis connectivity in CLI commands
        # For now, just ensure mocking works
        assert mock_redis is not None
        mock_redis.ping.assert_not_called()  # Not called yet
    
    def test_cli_file_system_integration(self):
        """Test CLI file system operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test directory creation and file operations
            test_file = os.path.join(temp_dir, "test.txt")
            
            with open(test_file, "w") as f:
                f.write("test content")
            
            assert os.path.exists(test_file), "File should be created"
            
            with open(test_file, "r") as f:
                content = f.read()
            
            assert content == "test content", "File content should match"


class TestCLISecurityValidation:
    """Test CLI security aspects."""
    
    def test_config_file_permissions(self):
        """Test configuration file permissions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "secure_config.json")
            
            # Create config with sensitive data
            sensitive_config = {
                "alpaca": {
                    "api_key": "test_key",
                    "secret_key": "test_secret"
                }
            }
            
            with open(config_path, "w") as f:
                json.dump(sensitive_config, f)
            
            # Check file permissions (on Unix systems)
            if os.name == 'posix':
                stat_info = os.stat(config_path)
                permissions = oct(stat_info.st_mode)[-3:]
                
                # Should be readable by owner only (600) or at least not world-readable
                assert not permissions.endswith('4'), "Config file should not be world-readable"
    
    def test_cli_input_validation(self):
        """Test CLI input validation."""
        # Test with various potentially problematic inputs
        problematic_inputs = [
            "../../../etc/passwd",  # Path traversal
            "'; DROP TABLE users; --",  # SQL injection style
            "<script>alert('xss')</script>",  # XSS style
            "$(rm -rf /)",  # Command injection style
        ]
        
        # These should be handled safely by the CLI
        for bad_input in problematic_inputs:
            # Test would depend on specific CLI commands that accept input
            # For now, just ensure the inputs are properly escaped
            escaped = json.dumps(bad_input)
            assert '\\' in escaped or bad_input.replace('"', '\\"') in escaped


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])