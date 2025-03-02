#!/usr/bin/env python3
"""
Configuration reader for healthcare learning dashboard.
Reads dashboard_config.json and provides access to feature flags and settings.
"""

import os
import json
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger("dashboard-config")

class DashboardConfig:
    """Reader for dashboard configuration."""
    
    # Default configuration if file is missing
    DEFAULT_CONFIG = {
        "dashboard": {
            "mode": "regular",
            "refresh_interval": 15,
            "auto_save": True,
            "debug_mode": False,
            "features": {
                "batch_processing": True,
                "dataset_import": True,
                "performance_comparison": True,
                "advanced_testing": True
            },
            "default_settings": {
                "cycles": 5,
                "batch_size": 64,
                "evaluation_frequency": 2
            }
        },
        "performance": {
            "thresholds": {
                "critical": 0.65,
                "warning": 0.75,
                "target": 0.85
            }
        },
        "testing": {
            "batch_size": 64,
            "categories": [
                "medication_interaction",
                "treatment_protocol",
                "dosage_conflict",
                "diagnostic_conflict"
            ]
        }
    }
    
    def __init__(self, config_path="dashboard_config.json"):
        """Initialize configuration reader.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self):
        """Load configuration from file."""
        try:
            # Check for configuration in multiple locations
            config_locations = [
                self.config_path,
                Path("config") / self.config_path.name,
                Path(os.path.dirname(os.path.dirname(__file__))) / "config" / self.config_path.name
            ]
            
            for path in config_locations:
                if path.exists():
                    with open(path, 'r') as f:
                        config = json.load(f)
                    logger.info(f"Loaded configuration from {path}")
                    return config
            
            # If we get here, no config file was found
            logger.warning(f"Configuration file {self.config_path} not found. Using defaults.")
            return self.DEFAULT_CONFIG
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return self.DEFAULT_CONFIG
            
    def is_feature_enabled(self, feature_name):
        """Check if a feature is enabled.
        
        Args:
            feature_name: Name of feature to check
            
        Returns:
            bool: True if feature is enabled, False otherwise
        """
        try:
            return self.config.get("dashboard", {}).get("features", {}).get(feature_name, False)
        except Exception as e:
            logger.error(f"Error checking feature {feature_name}: {str(e)}")
            return False
            
    def get_setting(self, setting_name, default=None):
        """Get a setting value.
        
        Args:
            setting_name: Name of setting to get
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        try:
            return self.config.get("dashboard", {}).get("default_settings", {}).get(setting_name, default)
        except Exception as e:
            logger.error(f"Error getting setting {setting_name}: {str(e)}")
            return default
            
    def get_all_features(self):
        """Get all features and their enabled status.
        
        Returns:
            dict: Feature names and enabled status
        """
        try:
            return self.config.get("dashboard", {}).get("features", {})
        except Exception as e:
            logger.error(f"Error getting features: {str(e)}")
            return {}
            
    def get_all_settings(self):
        """Get all settings.
        
        Returns:
            dict: Setting names and values
        """
        try:
            return self.config.get("dashboard", {}).get("default_settings", {})
        except Exception as e:
            logger.error(f"Error getting settings: {str(e)}")
            return {}

# Global instance for easy import
config = DashboardConfig()

if __name__ == "__main__":
    # Simple test if run directly
    import sys
    logging.basicConfig(level=logging.INFO)
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "dashboard_config.json"
    config = DashboardConfig(config_path)
    
    print("Features:")
    for feature, enabled in config.get_all_features().items():
        print(f"  {feature}: {'Enabled' if enabled else 'Disabled'}")
        
    print("\nSettings:")
    for setting, value in config.get_all_settings().items():
        print(f"  {setting}: {value}")
