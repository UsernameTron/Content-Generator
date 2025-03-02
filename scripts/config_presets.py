#!/usr/bin/env python3
"""
Configuration Preset System for Healthcare Contradiction Detection Dashboard

This module provides functionality to save, load, and manage configuration presets
for the healthcare contradiction detection dashboard, allowing users to quickly
switch between different testing configurations.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/config_presets.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("config_presets")

# Console for rich output
console = Console()

class ConfigPresetManager:
    """Manages configuration presets for the dashboard."""
    
    def __init__(self, presets_dir=None):
        """Initialize the configuration preset manager.
        
        Args:
            presets_dir: Directory to store configuration presets
        """
        # Set up paths
        self.base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        if presets_dir is None:
            self.presets_dir = self.base_dir / "config" / "presets"
        else:
            self.presets_dir = Path(presets_dir)
            
        # Create presets directory if it doesn't exist
        os.makedirs(self.presets_dir, exist_ok=True)
        
        # Load available presets
        self.presets = self._load_available_presets()
        
    def _load_available_presets(self):
        """Load available configuration presets.
        
        Returns:
            dict: Dictionary of available presets
        """
        presets = {}
        
        # Find all JSON files in presets directory
        for preset_file in self.presets_dir.glob("*.json"):
            try:
                with open(preset_file, 'r') as f:
                    preset_data = json.load(f)
                    
                # Extract preset metadata
                preset_name = preset_file.stem
                preset_info = {
                    "name": preset_name,
                    "path": str(preset_file),
                    "description": preset_data.get("description", "No description"),
                    "created": preset_data.get("created", "Unknown"),
                    "modified": preset_data.get("modified", "Unknown"),
                    "tags": preset_data.get("tags", []),
                    "config": preset_data.get("config", {})
                }
                
                presets[preset_name] = preset_info
            except Exception as e:
                logger.error(f"Error loading preset {preset_file}: {str(e)}")
                
        return presets
        
    def list_presets(self):
        """List available configuration presets.
        
        Returns:
            list: List of preset information dictionaries
        """
        # Reload presets to ensure we have the latest
        self.presets = self._load_available_presets()
        
        # Return list of presets
        return list(self.presets.values())
        
    def display_presets(self):
        """Display available configuration presets in a table."""
        presets = self.list_presets()
        
        if not presets:
            console.print("[yellow]No configuration presets found.[/yellow]")
            return
            
        # Create table
        table = Table(title="Available Configuration Presets")
        table.add_column("Name")
        table.add_column("Description")
        table.add_column("Created")
        table.add_column("Modified")
        table.add_column("Tags")
        
        # Add rows
        for preset in presets:
            tags = ", ".join(preset.get("tags", []))
            table.add_row(
                preset["name"],
                preset["description"],
                preset["created"],
                preset["modified"],
                tags
            )
            
        # Display table
        console.print(table)
        
    def get_preset(self, preset_name):
        """Get a specific configuration preset.
        
        Args:
            preset_name: Name of the preset to get
            
        Returns:
            dict: Preset configuration or None if not found
        """
        if preset_name not in self.presets:
            logger.warning(f"Preset '{preset_name}' not found")
            return None
            
        preset_info = self.presets[preset_name]
        
        try:
            with open(preset_info["path"], 'r') as f:
                preset_data = json.load(f)
                return preset_data.get("config", {})
        except Exception as e:
            logger.error(f"Error loading preset '{preset_name}': {str(e)}")
            return None
            
    def save_preset(self, config, name, description="", tags=None):
        """Save a configuration as a preset.
        
        Args:
            config: Configuration to save
            name: Name for the preset
            description: Description of the preset
            tags: List of tags for the preset
            
        Returns:
            bool: True if successful, False otherwise
        """
        if tags is None:
            tags = []
            
        # Create preset data
        preset_data = {
            "description": description,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "modified": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tags": tags,
            "config": config
        }
        
        # Sanitize name for filename
        safe_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in name)
        
        # Create preset file path
        preset_path = self.presets_dir / f"{safe_name}.json"
        
        # Check if preset already exists
        if preset_path.exists():
            # Update existing preset
            try:
                with open(preset_path, 'r') as f:
                    existing_data = json.load(f)
                    
                # Preserve creation date
                preset_data["created"] = existing_data.get("created", preset_data["created"])
            except Exception as e:
                logger.error(f"Error reading existing preset '{name}': {str(e)}")
                
        # Save preset
        try:
            with open(preset_path, 'w') as f:
                json.dump(preset_data, f, indent=2)
                
            logger.info(f"Preset '{name}' saved to {preset_path}")
            
            # Reload presets
            self.presets = self._load_available_presets()
            
            return True
        except Exception as e:
            logger.error(f"Error saving preset '{name}': {str(e)}")
            return False
            
    def delete_preset(self, preset_name):
        """Delete a configuration preset.
        
        Args:
            preset_name: Name of the preset to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if preset_name not in self.presets:
            logger.warning(f"Preset '{preset_name}' not found")
            return False
            
        preset_path = self.presets[preset_name]["path"]
        
        try:
            os.remove(preset_path)
            logger.info(f"Preset '{preset_name}' deleted")
            
            # Reload presets
            self.presets = self._load_available_presets()
            
            return True
        except Exception as e:
            logger.error(f"Error deleting preset '{preset_name}': {str(e)}")
            return False
            
    def export_preset(self, preset_name, export_path):
        """Export a configuration preset to a file.
        
        Args:
            preset_name: Name of the preset to export
            export_path: Path to export the preset to
            
        Returns:
            bool: True if successful, False otherwise
        """
        if preset_name not in self.presets:
            logger.warning(f"Preset '{preset_name}' not found")
            return False
            
        preset_path = self.presets[preset_name]["path"]
        
        try:
            # Create export directory if it doesn't exist
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            # Copy preset file
            with open(preset_path, 'r') as src, open(export_path, 'w') as dst:
                dst.write(src.read())
                
            logger.info(f"Preset '{preset_name}' exported to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting preset '{preset_name}': {str(e)}")
            return False
            
    def import_preset(self, import_path, new_name=None):
        """Import a configuration preset from a file.
        
        Args:
            import_path: Path to import the preset from
            new_name: New name for the imported preset (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load preset data
            with open(import_path, 'r') as f:
                preset_data = json.load(f)
                
            # Determine preset name
            if new_name is None:
                # Use filename as preset name
                new_name = Path(import_path).stem
                
            # Save as new preset
            return self.save_preset(
                preset_data.get("config", {}),
                new_name,
                preset_data.get("description", "Imported preset"),
                preset_data.get("tags", [])
            )
        except Exception as e:
            logger.error(f"Error importing preset from {import_path}: {str(e)}")
            return False
            
    def apply_preset(self, preset_name, config_path):
        """Apply a configuration preset to the dashboard configuration.
        
        Args:
            preset_name: Name of the preset to apply
            config_path: Path to the dashboard configuration file
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Get preset configuration
        preset_config = self.get_preset(preset_name)
        
        if preset_config is None:
            return False
            
        try:
            # Load current configuration
            with open(config_path, 'r') as f:
                current_config = json.load(f)
                
            # Create backup of current configuration
            backup_path = f"{config_path}.backup.{datetime.now().strftime('%Y%m%d%H%M%S')}"
            with open(backup_path, 'w') as f:
                json.dump(current_config, f, indent=2)
                
            # Apply preset configuration (merge with current configuration)
            merged_config = self._merge_configs(current_config, preset_config)
            
            # Save merged configuration
            with open(config_path, 'w') as f:
                json.dump(merged_config, f, indent=2)
                
            logger.info(f"Applied preset '{preset_name}' to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error applying preset '{preset_name}': {str(e)}")
            return False
            
    def _merge_configs(self, base_config, preset_config):
        """Merge a preset configuration with a base configuration.
        
        Args:
            base_config: Base configuration
            preset_config: Preset configuration to merge
            
        Returns:
            dict: Merged configuration
        """
        # Create a copy of the base configuration
        merged = base_config.copy()
        
        # Merge preset configuration
        for key, value in preset_config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                # Recursively merge nested dictionaries
                merged[key] = self._merge_configs(merged[key], value)
            else:
                # Replace or add value
                merged[key] = value
                
        return merged
        
    def create_preset_from_current(self, config_path):
        """Create a new preset from the current dashboard configuration.
        
        Args:
            config_path: Path to the dashboard configuration file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load current configuration
            with open(config_path, 'r') as f:
                current_config = json.load(f)
                
            # Prompt for preset information
            console.print("\n[bold]Create New Configuration Preset[/bold]")
            name = Prompt.ask("Preset name")
            description = Prompt.ask("Description", default="")
            tags_input = Prompt.ask("Tags (comma-separated)", default="")
            tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
            
            # Save as new preset
            return self.save_preset(current_config, name, description, tags)
        except Exception as e:
            logger.error(f"Error creating preset from current configuration: {str(e)}")
            return False
            
    def interactive_menu(self, config_path=None):
        """Display an interactive menu for managing presets.
        
        Args:
            config_path: Path to the dashboard configuration file
        """
        if config_path is None:
            config_path = self.base_dir / "config" / "dashboard_config.json"
            
        while True:
            console.print("\n[bold]Configuration Preset Manager[/bold]")
            console.print("1. List available presets")
            console.print("2. Create new preset from current configuration")
            console.print("3. Apply preset to current configuration")
            console.print("4. Delete preset")
            console.print("5. Export preset")
            console.print("6. Import preset")
            console.print("0. Exit")
            
            choice = Prompt.ask("Enter your choice", choices=["0", "1", "2", "3", "4", "5", "6"], default="0")
            
            if choice == "0":
                break
            elif choice == "1":
                self.display_presets()
            elif choice == "2":
                if self.create_preset_from_current(config_path):
                    console.print("[green]Preset created successfully.[/green]")
            elif choice == "3":
                presets = self.list_presets()
                if not presets:
                    console.print("[yellow]No presets available.[/yellow]")
                    continue
                    
                console.print("\nAvailable presets:")
                for i, preset in enumerate(presets):
                    console.print(f"{i+1}. {preset['name']} - {preset['description']}")
                    
                preset_idx = Prompt.ask("Select preset to apply", choices=[str(i+1) for i in range(len(presets))], default="1")
                preset_name = presets[int(preset_idx)-1]["name"]
                
                if Confirm.ask(f"Apply preset '{preset_name}' to current configuration?"):
                    if self.apply_preset(preset_name, config_path):
                        console.print(f"[green]Preset '{preset_name}' applied successfully.[/green]")
            elif choice == "4":
                presets = self.list_presets()
                if not presets:
                    console.print("[yellow]No presets available.[/yellow]")
                    continue
                    
                console.print("\nAvailable presets:")
                for i, preset in enumerate(presets):
                    console.print(f"{i+1}. {preset['name']} - {preset['description']}")
                    
                preset_idx = Prompt.ask("Select preset to delete", choices=[str(i+1) for i in range(len(presets))], default="1")
                preset_name = presets[int(preset_idx)-1]["name"]
                
                if Confirm.ask(f"Delete preset '{preset_name}'? This cannot be undone.", default=False):
                    if self.delete_preset(preset_name):
                        console.print(f"[green]Preset '{preset_name}' deleted successfully.[/green]")
            elif choice == "5":
                presets = self.list_presets()
                if not presets:
                    console.print("[yellow]No presets available.[/yellow]")
                    continue
                    
                console.print("\nAvailable presets:")
                for i, preset in enumerate(presets):
                    console.print(f"{i+1}. {preset['name']} - {preset['description']}")
                    
                preset_idx = Prompt.ask("Select preset to export", choices=[str(i+1) for i in range(len(presets))], default="1")
                preset_name = presets[int(preset_idx)-1]["name"]
                
                export_path = Prompt.ask("Export path", default=f"~/Desktop/{preset_name}.json")
                export_path = os.path.expanduser(export_path)
                
                if self.export_preset(preset_name, export_path):
                    console.print(f"[green]Preset '{preset_name}' exported to {export_path}.[/green]")
            elif choice == "6":
                import_path = Prompt.ask("Import path")
                import_path = os.path.expanduser(import_path)
                
                if not os.path.exists(import_path):
                    console.print(f"[red]File not found: {import_path}[/red]")
                    continue
                    
                new_name = Prompt.ask("New preset name (leave empty to use filename)", default="")
                
                if not new_name:
                    new_name = None
                    
                if self.import_preset(import_path, new_name):
                    console.print("[green]Preset imported successfully.[/green]")
                    
def main():
    """Main function for running the configuration preset manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Preset Manager for Healthcare Contradiction Detection Dashboard")
    parser.add_argument("--config", type=str, help="Path to dashboard configuration file")
    parser.add_argument("--presets-dir", type=str, help="Directory to store configuration presets")
    parser.add_argument("--list", action="store_true", help="List available presets")
    parser.add_argument("--create", action="store_true", help="Create new preset from current configuration")
    parser.add_argument("--apply", type=str, help="Apply preset to current configuration")
    parser.add_argument("--delete", type=str, help="Delete preset")
    parser.add_argument("--export", type=str, help="Export preset")
    parser.add_argument("--export-path", type=str, help="Path to export preset to")
    parser.add_argument("--import", dest="import_path", type=str, help="Import preset from file")
    parser.add_argument("--import-name", type=str, help="Name for imported preset")
    parser.add_argument("--interactive", action="store_true", help="Start interactive menu")
    
    args = parser.parse_args()
    
    # Create preset manager
    manager = ConfigPresetManager(args.presets_dir)
    
    # Determine configuration path
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(manager.base_dir, "config", "dashboard_config.json")
        
    # Process commands
    if args.list:
        manager.display_presets()
    elif args.create:
        if manager.create_preset_from_current(config_path):
            console.print("[green]Preset created successfully.[/green]")
    elif args.apply:
        if manager.apply_preset(args.apply, config_path):
            console.print(f"[green]Preset '{args.apply}' applied successfully.[/green]")
    elif args.delete:
        if manager.delete_preset(args.delete):
            console.print(f"[green]Preset '{args.delete}' deleted successfully.[/green]")
    elif args.export:
        if not args.export_path:
            console.print("[red]Export path not specified.[/red]")
        else:
            if manager.export_preset(args.export, args.export_path):
                console.print(f"[green]Preset '{args.export}' exported to {args.export_path}.[/green]")
    elif args.import_path:
        if manager.import_preset(args.import_path, args.import_name):
            console.print("[green]Preset imported successfully.[/green]")
    elif args.interactive:
        manager.interactive_menu(config_path)
    else:
        # Default to interactive menu
        manager.interactive_menu(config_path)
        
if __name__ == "__main__":
    main()
