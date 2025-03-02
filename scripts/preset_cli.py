#!/usr/bin/env python3
"""
Command-Line Interface for the Configuration Preset System

This script provides a command-line interface for managing configuration presets,
enabling automation and integration with other tools and scripts.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the preset manager
from scripts.config_presets import ConfigPresetManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("preset_cli")

# Console for rich output
console = Console()

def setup_parser():
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Command-line interface for managing configuration presets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available presets
  python preset_cli.py list
  
  # Apply a preset to the current configuration
  python preset_cli.py apply comprehensive_testing
  
  # Create a new preset from the current configuration
  python preset_cli.py create "My Custom Preset" --description "Custom configuration for specialized testing"
  
  # Export a preset to a file
  python preset_cli.py export comprehensive_testing ~/Desktop/comprehensive_testing.json
  
  # Import a preset from a file
  python preset_cli.py import ~/Desktop/custom_preset.json "Custom Preset"
  
  # Delete a preset
  python preset_cli.py delete my_old_preset
        """
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available presets")
    list_parser.add_argument("--tag", help="Filter presets by tag")
    list_parser.add_argument("--format", choices=["table", "json", "csv"], default="table",
                            help="Output format (default: table)")
    
    # Apply command
    apply_parser = subparsers.add_parser("apply", help="Apply a preset to the current configuration")
    apply_parser.add_argument("preset", help="Name of the preset to apply")
    apply_parser.add_argument("--config", help="Path to the configuration file (default: config/dashboard_config.json)")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new preset from the current configuration")
    create_parser.add_argument("name", help="Name for the new preset")
    create_parser.add_argument("--description", default="", help="Description of the preset")
    create_parser.add_argument("--tags", help="Comma-separated list of tags")
    create_parser.add_argument("--config", help="Path to the configuration file (default: config/dashboard_config.json)")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export a preset to a file")
    export_parser.add_argument("preset", help="Name of the preset to export")
    export_parser.add_argument("output", help="Output file path")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import a preset from a file")
    import_parser.add_argument("input", help="Input file path")
    import_parser.add_argument("name", nargs="?", help="New name for the imported preset (optional)")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a preset")
    delete_parser.add_argument("preset", help="Name of the preset to delete")
    delete_parser.add_argument("--force", action="store_true", help="Force deletion without confirmation")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show detailed information about a preset")
    info_parser.add_argument("preset", help="Name of the preset to show information about")
    info_parser.add_argument("--format", choices=["text", "json"], default="text",
                           help="Output format (default: text)")
    
    return parser

def list_presets(args, preset_manager):
    """List available presets."""
    presets = preset_manager.list_presets()
    
    # Filter by tag if specified
    if args.tag:
        presets = [p for p in presets if args.tag in p.get("tags", [])]
    
    if not presets:
        console.print("[yellow]No presets found.[/yellow]")
        return
    
    if args.format == "table":
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
    elif args.format == "json":
        import json
        console.print(json.dumps(presets, indent=2))
    elif args.format == "csv":
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(["Name", "Description", "Created", "Modified", "Tags"])
        
        # Write rows
        for preset in presets:
            tags = ",".join(preset.get("tags", []))
            writer.writerow([
                preset["name"],
                preset["description"],
                preset["created"],
                preset["modified"],
                tags
            ])
            
        console.print(output.getvalue())

def apply_preset(args, preset_manager):
    """Apply a preset to the current configuration."""
    config_path = args.config
    if not config_path:
        config_path = os.path.join(preset_manager.base_dir, "config", "dashboard_config.json")
    
    if preset_manager.apply_preset(args.preset, config_path):
        console.print(f"[green]Successfully applied preset '{args.preset}' to {config_path}[/green]")
    else:
        console.print(f"[red]Failed to apply preset '{args.preset}'[/red]")
        return 1
    
    return 0

def create_preset(args, preset_manager):
    """Create a new preset from the current configuration."""
    config_path = args.config
    if not config_path:
        config_path = os.path.join(preset_manager.base_dir, "config", "dashboard_config.json")
    
    # Check if configuration file exists
    if not os.path.exists(config_path):
        console.print(f"[red]Configuration file not found: {config_path}[/red]")
        return 1
    
    # Load configuration
    try:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading configuration: {str(e)}[/red]")
        return 1
    
    # Parse tags
    tags = []
    if args.tags:
        tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    
    # Save preset
    if preset_manager.save_preset(config, args.name, args.description, tags):
        console.print(f"[green]Successfully created preset '{args.name}'[/green]")
    else:
        console.print(f"[red]Failed to create preset '{args.name}'[/red]")
        return 1
    
    return 0

def export_preset(args, preset_manager):
    """Export a preset to a file."""
    if preset_manager.export_preset(args.preset, args.output):
        console.print(f"[green]Successfully exported preset '{args.preset}' to {args.output}[/green]")
    else:
        console.print(f"[red]Failed to export preset '{args.preset}'[/red]")
        return 1
    
    return 0

def import_preset(args, preset_manager):
    """Import a preset from a file."""
    if preset_manager.import_preset(args.input, args.name):
        console.print(f"[green]Successfully imported preset from {args.input}[/green]")
    else:
        console.print(f"[red]Failed to import preset from {args.input}[/red]")
        return 1
    
    return 0

def delete_preset(args, preset_manager):
    """Delete a preset."""
    # Confirm deletion if not forced
    if not args.force:
        from rich.prompt import Confirm
        if not Confirm.ask(f"Are you sure you want to delete preset '{args.preset}'?", default=False):
            console.print("[yellow]Deletion cancelled.[/yellow]")
            return 0
    
    if preset_manager.delete_preset(args.preset):
        console.print(f"[green]Successfully deleted preset '{args.preset}'[/green]")
    else:
        console.print(f"[red]Failed to delete preset '{args.preset}'[/red]")
        return 1
    
    return 0

def show_preset_info(args, preset_manager):
    """Show detailed information about a preset."""
    preset_config = preset_manager.get_preset(args.preset)
    if not preset_config:
        console.print(f"[red]Preset '{args.preset}' not found[/red]")
        return 1
    
    preset_info = preset_manager.presets.get(args.preset, {})
    
    if args.format == "json":
        import json
        info = {
            "name": args.preset,
            "description": preset_info.get("description", ""),
            "created": preset_info.get("created", ""),
            "modified": preset_info.get("modified", ""),
            "tags": preset_info.get("tags", []),
            "config": preset_config
        }
        console.print(json.dumps(info, indent=2))
    else:
        from rich.panel import Panel
        from rich.text import Text
        
        # Create header
        header = Text(f"Preset: {args.preset}", style="bold blue")
        
        # Create content
        content = []
        content.append(f"[bold]Description:[/bold] {preset_info.get('description', '')}")
        content.append(f"[bold]Created:[/bold] {preset_info.get('created', '')}")
        content.append(f"[bold]Modified:[/bold] {preset_info.get('modified', '')}")
        content.append(f"[bold]Tags:[/bold] {', '.join(preset_info.get('tags', []))}")
        content.append("")
        content.append("[bold]Configuration:[/bold]")
        
        # Format configuration
        import json
        config_str = json.dumps(preset_config, indent=2)
        content.append(f"```json\n{config_str}\n```")
        
        # Display panel
        console.print(Panel("\n".join(content), title=header))
    
    return 0

def main():
    """Main function."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Create preset manager
    preset_manager = ConfigPresetManager()
    
    # Execute command
    if args.command == "list":
        return list_presets(args, preset_manager)
    elif args.command == "apply":
        return apply_preset(args, preset_manager)
    elif args.command == "create":
        return create_preset(args, preset_manager)
    elif args.command == "export":
        return export_preset(args, preset_manager)
    elif args.command == "import":
        return import_preset(args, preset_manager)
    elif args.command == "delete":
        return delete_preset(args, preset_manager)
    elif args.command == "info":
        return show_preset_info(args, preset_manager)
    else:
        parser.print_help()
        return 0

if __name__ == "__main__":
    sys.exit(main())
