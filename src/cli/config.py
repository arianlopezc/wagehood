"""CLI configuration management."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages CLI configuration stored in user's home directory."""
    
    def __init__(self):
        self.config_dir = Path.home() / '.wagehood'
        self.config_file = self.config_dir / 'config.json'
        self.ensure_config_dir()
    
    def ensure_config_dir(self):
        """Ensure configuration directory exists."""
        self.config_dir.mkdir(exist_ok=True)
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        config = self.load_config()
        return config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        config = self.load_config()
        config[key] = value
        self.save_config(config)
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values."""
        config = self.load_config()
        config.update(updates)
        self.save_config(config)
    
    def get_env_vars(self) -> Dict[str, str]:
        """Get environment variables from configuration."""
        config = self.load_config()
        env_vars = {}
        
        if 'api_key' in config:
            env_vars['ALPACA_API_KEY'] = config['api_key']
        if 'secret_key' in config:
            env_vars['ALPACA_SECRET_KEY'] = config['secret_key']
        if 'symbols' in config:
            env_vars['SUPPORTED_SYMBOLS'] = ','.join(config['symbols'])
        
        return env_vars
    
    def is_configured(self) -> bool:
        """Check if CLI is configured."""
        config = self.load_config()
        return all(key in config for key in ['api_key', 'secret_key', 'symbols'])