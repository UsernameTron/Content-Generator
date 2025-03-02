"""
Resource Monitor for Fine-Tuning

This script monitors system resources during model fine-tuning, especially
important for Apple Silicon devices where memory and MPS utilization
need to be carefully tracked.
"""

import os
import time
import logging
import psutil
import json
from datetime import datetime
from pathlib import Path
import subprocess
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()

class ResourceMonitor:
    """Monitor system resources during model training."""
    
    def __init__(self, output_dir="logs/resource_monitoring"):
        """Initialize resource monitor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.now()
        self.thresholds = self._load_thresholds()
        self.log_file = self.output_dir / f"resources_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        self.data = []
        
        # Memory thresholds (from user global memory)
        self.mem_warning = self.thresholds.get("memory_warning", 70)  # % of total memory
        self.mem_critical = self.thresholds.get("memory_critical", 85)
        self.mem_rollback = self.thresholds.get("memory_rollback", 90)
        
        # Initial snapshot for baseline
        self.baseline = self._get_resource_snapshot()
        self._log_snapshot(self.baseline, "baseline")
    
    def _load_thresholds(self):
        """Load monitoring thresholds."""
        # Default values 
        thresholds = {
            "memory_warning": 70,     # % of total memory
            "memory_critical": 85,
            "memory_rollback": 90,
            "cpu_warning": 90,         # % of CPU utilization
            "gpu_warning": 90,         # % of GPU utilization for MPS
            "disk_warning": 90         # % of disk utilization
        }
        
        # Try to load user-specific thresholds
        config_path = Path("config/monitoring_thresholds.json")
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    user_thresholds = json.load(f)
                thresholds.update(user_thresholds)
                logger.info(f"Loaded custom monitoring thresholds: {thresholds}")
            except Exception as e:
                logger.warning(f"Error loading thresholds: {str(e)}")
        
        return thresholds
    
    def _get_resource_snapshot(self):
        """Get current system resource usage."""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.5)
        disk = psutil.disk_usage('/')
        
        # Get MPS (Metal Performance Shaders) info for Apple Silicon
        mps_info = {'available': False, 'usage': 0}
        if hasattr(subprocess, 'run'):
            try:
                # Using powermetrics to get GPU usage on macOS
                # Note: requires sudo permissions in real use
                # result = subprocess.run(
                #     ['powermetrics', '-n', '1', '-i', '1000', '--samplers', 'gpu_power'],
                #     capture_output=True, text=True, timeout=2
                # )
                # 
                # For demo purposes, we'll just detect if MPS is available
                import torch
                mps_info['available'] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                mps_info['usage'] = 50  # Placeholder for demo
            except Exception as e:
                logger.warning(f"Could not get MPS info: {str(e)}")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'memory': {
                'total': memory.total,
                'used': memory.used,
                'percent': memory.percent
            },
            'cpu': {
                'percent': cpu
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'percent': disk.percent
            },
            'mps': mps_info,
            'processes': self._get_relevant_processes()
        }
    
    def _get_relevant_processes(self):
        """Get information about relevant processes (Python, ML related)."""
        relevant = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_percent']):
            try:
                proc_info = proc.info
                cmd = " ".join(proc_info['cmdline']) if proc_info['cmdline'] else ""
                if 'python' in cmd and ('finetune' in cmd or 'train' in cmd or 'model' in cmd):
                    relevant.append({
                        'pid': proc_info['pid'],
                        'name': proc_info['name'],
                        'memory_percent': proc_info['memory_percent'],
                        'cmd': cmd[:100] + ('...' if len(cmd) > 100 else '')
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return relevant
    
    def _log_snapshot(self, snapshot, label="snapshot"):
        """Log a resource snapshot with an optional label."""
        entry = snapshot.copy()
        entry['label'] = label
        self.data.append(entry)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        # Log warnings based on thresholds
        mem_percent = snapshot['memory']['percent']
        if mem_percent >= self.mem_rollback:
            logger.critical(f"MEMORY USAGE CRITICAL: {mem_percent}% (ROLLBACK THRESHOLD EXCEEDED)")
        elif mem_percent >= self.mem_critical:
            logger.error(f"MEMORY USAGE CRITICAL: {mem_percent}%")
        elif mem_percent >= self.mem_warning:
            logger.warning(f"MEMORY USAGE HIGH: {mem_percent}%")
    
    def monitor(self, duration=3600, interval=10):
        """Monitor resources for the specified duration."""
        end_time = time.time() + duration
        samples = 0
        
        with Progress() as progress:
            task = progress.add_task("[green]Monitoring resources...", total=duration)
            
            while time.time() < end_time:
                # Update the progress bar
                progress.update(task, completed=min(time.time() - (end_time - duration), duration))
                
                # Get and log resource snapshot
                snapshot = self._get_resource_snapshot()
                self._log_snapshot(snapshot)
                samples += .1
                
                # Display current stats
                self._display_current_stats(snapshot)
                
                # Check for high memory usage
                if snapshot['memory']['percent'] >= self.mem_rollback:
                    logger.critical("Memory usage exceeded rollback threshold! Consider terminating training.")
                    console.print("[bold red]ALERT: Memory usage critical! Training may be unstable.[/bold red]")
                
                time.sleep(interval)
        
        logger.info(f"Monitoring complete. Collected {samples} samples over {duration/60:.1f} minutes.")
        self._generate_report()
    
    def _display_current_stats(self, snapshot):
        """Display current resource stats in a pretty table."""
        table = Table(title="Current Resource Usage")
        
        table.add_column("Resource", style="cyan")
        table.add_column("Usage", style="magenta")
        table.add_column("Status", style="green")
        
        # Memory
        mem_percent = snapshot['memory']['percent']
        mem_status = "OK"
        if mem_percent >= self.mem_rollback:
            mem_status = "CRITICAL - ROLLBACK"
        elif mem_percent >= self.mem_critical:
            mem_status = "CRITICAL"
        elif mem_percent >= self.mem_warning:
            mem_status = "WARNING"
        
        mem_gb = snapshot['memory']['used'] / (1024**3)
        total_gb = snapshot['memory']['total'] / (1024**3)
        table.add_row("Memory", f"{mem_gb:.1f} GB / {total_gb:.1f} GB ({mem_percent}%)", mem_status)
        
        # CPU
        cpu_percent = snapshot['cpu']['percent']
        cpu_status = "OK"
        if cpu_percent >= self.thresholds.get('cpu_warning', 90):
            cpu_status = "HIGH"
        table.add_row("CPU", f"{cpu_percent}%", cpu_status)
        
        # MPS
        if snapshot['mps']['available']:
            mps_status = "ACTIVE" 
            table.add_row("MPS (Apple GPU)", "Available", mps_status)
        else:
            table.add_row("MPS (Apple GPU)", "Not available", "INACTIVE")
        
        # Process count
        process_count = len(snapshot['processes'])
        table.add_row("Training Processes", str(process_count), "OK" if process_count > 0 else "NO PROCESSES")
        
        console.print(table)
    
    def _generate_report(self):
        """Generate a summary report of resource usage."""
        logger.info("Generating resource usage report...")
        
        # Extract memory usage over time
        timestamps = [datetime.fromisoformat(entry['timestamp']) for entry in self.data]
        memory_percentages = [entry['memory']['percent'] for entry in self.data]
        
        # Basic statistics
        avg_memory = sum(memory_percentages) / len(memory_percentages) if memory_percentages else 0
        max_memory = max(memory_percentages) if memory_percentages else 0
        
        # Summary
        report = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'num_samples': len(self.data),
            'memory': {
                'average_percent': avg_memory,
                'max_percent': max_memory,
                'warning_threshold': self.mem_warning,
                'critical_threshold': self.mem_critical
            },
            'thresholds_exceeded': {
                'memory_warning': any(p >= self.mem_warning for p in memory_percentages),
                'memory_critical': any(p >= self.mem_critical for p in memory_percentages),
                'memory_rollback': any(p >= self.mem_rollback for p in memory_percentages)
            }
        }
        
        # Save report
        report_file = self.output_dir / f"report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {report_file}")
        
        # Print summary
        console.rule("[bold]Resource Monitoring Summary[/bold]")
        console.print(f"Duration: {report['duration_minutes']:.1f} minutes")
        console.print(f"Samples: {report['num_samples']}")
        console.print(f"Average memory usage: {report['memory']['average_percent']:.1f}%")
        console.print(f"Maximum memory usage: {report['memory']['max_percent']:.1f}%")
        
        if report['thresholds_exceeded']['memory_rollback']:
            console.print("[bold red]ROLLBACK THRESHOLD EXCEEDED during monitoring![/bold red]")
        elif report['thresholds_exceeded']['memory_critical']:
            console.print("[bold orange]CRITICAL THRESHOLD EXCEEDED during monitoring![/bold orange]")
        elif report['thresholds_exceeded']['memory_warning']:
            console.print("[bold yellow]WARNING THRESHOLD EXCEEDED during monitoring![/bold yellow]")
        else:
            console.print("[bold green]All resource usage within acceptable limits.[/bold green]")

def main():
    """Main function to run the resource monitor."""
    console.rule("[bold]Resource Monitoring for Fine-Tuning[/bold]")
    console.print("This script monitors system resources during model fine-tuning")
    console.print("Press Ctrl+C to stop monitoring at any time.")
    
    try:
        # Default monitoring for 60 minutes with 10-second interval
        duration = 60 * 60  # 60 minutes in seconds
        interval = 10       # 10 seconds between checks
        
        monitor = ResourceMonitor()
        monitor.monitor(duration=duration, interval=interval)
    except KeyboardInterrupt:
        console.print("\n[bold]Monitoring stopped by user.[/bold]")
    except Exception as e:
        logger.error(f"Error during monitoring: {str(e)}")
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()
