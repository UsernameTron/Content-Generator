"""
Health monitoring utilities for the application.
"""

import os
import time
import psutil
import logging
import platform
from datetime import datetime
from typing import Dict, Any
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class HealthMonitor:
    """
    Monitor system health and resource usage.
    """
    
    def __init__(self):
        """Initialize the health monitor."""
        self.start_time = time.time()
        self.warning_thresholds = {
            "memory": 70.0,  # 70% usage
            "cpu": 80.0,     # 80% usage
            "disk": 80.0     # 80% usage
        }
        self.critical_thresholds = {
            "memory": 85.0,  # 85% usage
            "cpu": 90.0,     # 90% usage
            "disk": 90.0     # 90% usage
        }
        self.system_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "hostname": platform.node(),
            "startup_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        logger.info(f"HealthMonitor initialized on {self.system_info['platform']} with Python {self.system_info['python_version']}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get current system information.
        
        Returns:
            dict: System information including memory, CPU, and disk usage
        """
        try:
            # Get memory usage - with more efficient check
            memory = psutil.virtual_memory()
            memory_usage = {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            }
            
            # Get CPU usage - use non-blocking instant check instead of interval
            # which is more efficient for UI updates
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Get disk usage - cache this value since it changes rarely
            current_time = time.time()
            if not hasattr(self, '_disk_cache') or \
               not hasattr(self, '_disk_cache_time') or \
               current_time - self._disk_cache_time > 60:  # Refresh cache every 60 seconds
                
                disk = psutil.disk_usage('/')
                disk_usage = {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent
                }
                self._disk_cache = disk_usage
                self._disk_cache_time = current_time
            else:
                disk_usage = self._disk_cache
            
            # Get process information - optimize to avoid expensive calls
            process = psutil.Process(os.getpid())
            
            # Use cached process data with timed updates to reduce overhead
            if not hasattr(self, '_process_cache') or \
               not hasattr(self, '_process_cache_time') or \
               current_time - self._process_cache_time > 5:  # Refresh every 5 seconds
                
                process_info = {
                    "memory_percent": process.memory_percent(),
                    "cpu_percent": process.cpu_percent(interval=None),
                    "threads": len(process.threads()),
                    # Only get open_files if in debug mode as it's expensive
                    "open_files": len(process.open_files()) if logging.getLogger().getEffectiveLevel() <= logging.DEBUG else 0
                }
                self._process_cache = process_info
                self._process_cache_time = current_time
            else:
                process_info = self._process_cache
            
            # Calculate uptime
            uptime = time.time() - self.start_time
            
            # Compile all information
            system_info = {
                "memory_usage": memory_usage,
                "cpu_percent": cpu_percent,
                "disk_usage": disk_usage,
                "process_info": process_info,
                "uptime_seconds": uptime
            }
            
            logger.debug("System information collected successfully")
            return system_info
            
        except Exception as e:
            logger.error(f"Error getting system information: {str(e)}")
            return {
                "error": str(e),
                "memory_usage": {"percent": 0},
                "cpu_percent": 0,
                "disk_usage": {"percent": 0}
            }
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Check the overall system health.
        
        Returns:
            dict: Health status including warnings and critical issues
        """
        try:
            # Get system info
            system_info = self.get_system_info()
            
            # Get key metrics
            memory_percent = system_info["memory_usage"]["percent"]
            cpu_percent = system_info["cpu_percent"]
            disk_percent = system_info["disk_usage"]["percent"]
            
            # Define checks
            warning_checks = {
                "memory": memory_percent >= self.warning_thresholds["memory"],
                "cpu": cpu_percent >= self.warning_thresholds["cpu"],
                "disk": disk_percent >= self.warning_thresholds["disk"]
            }
            
            critical_checks = {
                "memory": memory_percent >= self.critical_thresholds["memory"],
                "cpu": cpu_percent >= self.critical_thresholds["cpu"],
                "disk": disk_percent >= self.critical_thresholds["disk"]
            }
            
            # Compile results
            has_warning = any(warning_checks.values())
            has_critical = any(critical_checks.values())
            
            # Generate messages
            warning_resources = [
                resource for resource, is_warning in warning_checks.items() if is_warning
            ]
            
            critical_resources = [
                resource for resource, is_critical in critical_checks.items() if is_critical
            ]
            
            # Overall status
            if has_critical:
                status = "Critical"
                message = f"Critical resource constraints detected: {', '.join(critical_resources)}"
            elif has_warning:
                status = "Warning"
                message = f"Resource usage warnings detected: {', '.join(warning_resources)}"
            else:
                status = "Healthy"
                message = "System is healthy"
            
            # Log health check
            if has_critical:
                logger.error(f"Health check: {message}")
            elif has_warning:
                logger.warning(f"Health check: {message}")
            else:
                logger.info("Health check: System is healthy")
            
            # Return simplified health information for the UI
            return {
                "status": status,
                "message": message,
                "metrics": {
                    "memory_percent": memory_percent,
                    "cpu_percent": cpu_percent,
                    "disk_percent": disk_percent,
                    "uptime_seconds": system_info["uptime_seconds"]
                },
                "warnings": warning_resources,
                "critical_issues": critical_resources
            }
            
        except Exception as e:
            logger.error(f"Error checking system health: {str(e)}")
            return {
                "status": "Error",
                "message": f"Error checking system health: {str(e)}",
                "metrics": {
                    "memory_percent": 0,
                    "cpu_percent": 0,
                    "disk_percent": 0,
                    "uptime_seconds": 0
                },
                "warnings": [],
                "critical_issues": []
            }
    
    def log_performance_metrics(self, operation_name: str, duration_seconds: float):
        """
        Log performance metrics for an operation.
        
        Args:
            operation_name: Name of the operation
            duration_seconds: Duration of the operation in seconds
        """
        try:
            # Get current system info
            system_info = self.get_system_info()
            
            # Log metrics
            logger.info(
                f"Performance metrics for {operation_name}: "
                f"Duration: {duration_seconds:.2f}s, "
                f"CPU: {system_info['cpu_percent']}%, "
                f"Memory: {system_info['memory_usage']['percent']}%"
            )
        except Exception as e:
            logger.error(f"Error logging performance metrics: {str(e)}")

def monitor_operation(func):
    """
    Decorator to monitor an operation's performance and resource usage.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function with monitoring
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        operation_name = func.__name__
        
        # Log start
        logger.debug(f"Starting operation: {operation_name}")
        
        # Get initial system state
        try:
            initial_memory = psutil.virtual_memory().percent
            initial_cpu = psutil.cpu_percent(interval=0.1)
            process = psutil.Process(os.getpid())
            initial_process_memory = process.memory_percent()
        except Exception as e:
            logger.warning(f"Could not capture initial system state: {str(e)}")
            initial_memory = 0
            initial_cpu = 0
            initial_process_memory = 0
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Get final system state
            try:
                final_memory = psutil.virtual_memory().percent
                final_cpu = psutil.cpu_percent(interval=0.1)
                final_process_memory = process.memory_percent()
                
                # Calculate changes
                memory_change = final_memory - initial_memory
                cpu_change = final_cpu - initial_cpu
                process_memory_change = final_process_memory - initial_process_memory
                
                # Log performance details
                duration = time.time() - start_time
                logger.info(
                    f"Operation {operation_name} completed in {duration:.2f}s. "
                    f"Memory: {memory_change:+.2f}%, CPU: {cpu_change:+.2f}%, "
                    f"Process Memory: {process_memory_change:+.2f}%"
                )
                
                # Log warning if significant resource changes
                if memory_change > 5 or cpu_change > 20:
                    logger.warning(
                        f"Operation {operation_name} caused significant resource changes: "
                        f"Memory: {memory_change:+.2f}%, CPU: {cpu_change:+.2f}%"
                    )
            except Exception as e:
                logger.warning(f"Could not capture final system state: {str(e)}")
            
            return result
            
        except Exception as e:
            # Log error
            logger.error(f"Error in operation {operation_name}: {str(e)}")
            
            # Re-raise the exception
            raise
    
    return wrapper

# Initialize global health monitor
health_monitor = HealthMonitor()

# Function for use with PyQt and non-class-based code
def check_system_health() -> Dict[str, Any]:
    """
    Get system health status for display in the UI.
    
    Returns:
        dict: Health status and metrics
    """
    return health_monitor.check_system_health()

def get_system_metrics() -> Dict[str, Any]:
    """
    Get detailed system metrics including platform information.
    
    Returns:
        dict: Detailed system metrics
    """
    health_data = health_monitor.check_system_health()
    metrics = health_data["metrics"]
    
    # Add system information
    result = {
        "status": health_data["status"],
        "metrics": metrics,
        "system_info": health_monitor.system_info,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "uptime_seconds": time.time() - health_monitor.start_time,
        "uptime_formatted": format_time_duration(time.time() - health_monitor.start_time)
    }
    
    return result

def format_time_duration(seconds: float) -> str:
    """
    Format seconds into a human-readable duration.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m {seconds}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

if __name__ == "__main__":
    # Test the health monitor
    health_data = health_monitor.check_system_health()
    print(f"Health Status: {health_data['status']}")
    print(f"Message: {health_data['message']}")
    print(f"Metrics: {health_data['metrics']}")
    
    # Test the detailed metrics
    detailed = get_system_metrics()
    print(f"\nSystem Info: {detailed['system_info']}")
    print(f"Uptime: {detailed['uptime_formatted']}")
