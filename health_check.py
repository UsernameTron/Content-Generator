#!/usr/bin/env python3
"""
CANDOR Health Check Utility
A diagnostic tool to verify system components and application dependencies
"""

import os
import sys
import platform
import importlib
import pkgutil
import psutil
import time
from pathlib import Path
import subprocess

# Add project directory to path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(message):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")

def print_section(message):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{message}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-' * 40}{Colors.ENDC}")

def print_success(message):
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠ {message}{Colors.ENDC}")

def print_error(message):
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def print_info(message):
    print(f"  {message}")

def check_system():
    print_section("System Information")
    
    # Basic system info
    print_info(f"OS: {platform.system()} {platform.release()}")
    print_info(f"Python: {platform.python_version()}")
    print_info(f"Processor: {platform.processor()}")
    
    # Memory usage
    memory = psutil.virtual_memory()
    memory_used_gb = memory.used / (1024 ** 3)
    memory_total_gb = memory.total / (1024 ** 3)
    memory_percent = memory.percent
    
    print_info(f"Memory: {memory_used_gb:.2f} GB / {memory_total_gb:.2f} GB ({memory_percent}%)")
    
    if memory_percent > 90:
        print_error("Memory usage is critically high!")
    elif memory_percent > 70:
        print_warning("Memory usage is high")
    else:
        print_success("Memory usage is acceptable")
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print_info(f"CPU Usage: {cpu_percent}%")
    
    if cpu_percent > 90:
        print_error("CPU usage is critically high!")
    elif cpu_percent > 70:
        print_warning("CPU usage is high")
    else:
        print_success("CPU usage is acceptable")
    
    # Disk usage
    disk = psutil.disk_usage('/')
    disk_used_gb = disk.used / (1024 ** 3)
    disk_total_gb = disk.total / (1024 ** 3)
    disk_percent = disk.percent
    
    print_info(f"Disk: {disk_used_gb:.2f} GB / {disk_total_gb:.2f} GB ({disk_percent}%)")
    
    if disk_percent > 90:
        print_error("Disk usage is critically high!")
    elif disk_percent > 70:
        print_warning("Disk usage is high")
    else:
        print_success("Disk usage is acceptable")

def check_dependencies():
    print_section("Dependency Check")
    
    # Define modules to check (package name -> actual module name)
    required_packages = {
        "PyQt6": "PyQt6",
        "nltk": "nltk",
        "requests": "requests",
        "beautifulsoup4": "bs4",  # The actual module name is bs4
        "PyPDF2": "PyPDF2",
        "python-docx": "docx"     # The actual module name is docx
    }
    
    all_installed = True
    
    for package_name, module_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print_success(f"{package_name} is installed")
        except ImportError:
            print_error(f"{package_name} is NOT installed")
            all_installed = False
    
    if all_installed:
        print_success("All required packages are installed")
    else:
        print_error("Some required packages are missing")
        print_info("Try running: pip install -r requirements.txt")

def check_nlp_resources():
    print_section("NLP Resources Check")
    
    try:
        import nltk
        
        # Check for required NLTK data
        all_resources_present = True
        
        # Check Punkt tokenizer
        try:
            nltk.data.find('tokenizers/punkt')
            print_success("Punkt Tokenizer is available")
        except LookupError:
            print_error("Punkt Tokenizer is NOT available")
            all_resources_present = False
            
        # Check Stopwords corpus
        try:
            nltk.data.find('corpora/stopwords')
            print_success("Stopwords Corpus is available")
        except LookupError:
            print_error("Stopwords Corpus is NOT available")
            all_resources_present = False
            
        # Check VADER lexicon - try both paths since it can be in different locations
        try:
            try:
                nltk.data.find('sentiment/vader_lexicon')
                print_success("VADER Lexicon is available")
            except LookupError:
                # Try alternative location
                try:
                    from nltk.sentiment.vader import SentimentIntensityAnalyzer
                    analyzer = SentimentIntensityAnalyzer()
                    # If we get here, it worked
                    print_success("VADER Lexicon is available (via SentimentIntensityAnalyzer)")
                except Exception as e:
                    raise LookupError(f"Failed to initialize VADER: {str(e)}")
        except LookupError as e:
            print_error(f"VADER Lexicon is NOT available: {str(e)}")
            all_resources_present = False
        
        if not all_resources_present:
            print_info("Missing NLTK resources. Run python setup_data.py")
    
    except ImportError:
        print_error("NLTK is not installed")
        print_info("Run: pip install nltk")

def check_project_structure():
    print_section("Project Structure Check")
    
    # Define expected project structure
    expected_dirs = [
        'src',
        'src/processors',
        'src/adapters',
        'src/models'
    ]
    
    expected_files = [
        'app.py',
        'requirements.txt',
        'setup.py',
        'setup_data.py',
        'README.md',
        'LAUNCH.md',
        'src/__init__.py',
        'src/processors/__init__.py',
        'src/processors/text_processor.py',
        'src/processors/document_processor.py',
        'src/processors/url_processor.py',
        'src/processors/sentiment_analyzer.py',
        'src/processors/content_transformer.py',
        'src/adapters/__init__.py',
        'src/adapters/platform_adapter.py',
        'src/models/__init__.py',
        'src/models/platform_specs.py'
    ]
    
    # Check directories
    all_dirs_exist = True
    for dir_path in expected_dirs:
        full_path = os.path.join(project_dir, dir_path)
        if os.path.isdir(full_path):
            print_success(f"Directory exists: {dir_path}")
        else:
            print_error(f"Directory missing: {dir_path}")
            all_dirs_exist = False
    
    # Check files
    all_files_exist = True
    for file_path in expected_files:
        full_path = os.path.join(project_dir, file_path)
        if os.path.isfile(full_path):
            print_success(f"File exists: {file_path}")
        else:
            print_error(f"File missing: {file_path}")
            all_files_exist = False
    
    if all_dirs_exist and all_files_exist:
        print_success("All expected directories and files exist")
    else:
        print_error("Some expected directories or files are missing")

def check_desktop_launcher():
    print_section("Desktop Launcher Check")
    
    launcher_path = os.path.expanduser("~/Desktop/CANDOR_Launcher.command")
    
    if os.path.isfile(launcher_path):
        print_success("Desktop launcher exists")
        
        # Check if executable
        if os.access(launcher_path, os.X_OK):
            print_success("Desktop launcher is executable")
        else:
            print_error("Desktop launcher is not executable")
            print_info("Run: chmod +x ~/Desktop/CANDOR_Launcher.command")
    else:
        print_error("Desktop launcher does not exist")
        print_info("Run setup.py to create the desktop launcher")

def check_python_path():
    print_section("Python Path Check")
    
    project_in_path = project_dir in sys.path
    
    if project_in_path:
        print_success("Project directory is in Python path")
    else:
        print_error("Project directory is NOT in Python path")
        print_info(f"Add {project_dir} to PYTHONPATH or use:")
        print_info(f"export PYTHONPATH=\"{project_dir}:$PYTHONPATH\"")
    
    # Print all paths
    print_info("Current Python path:")
    for path in sys.path:
        if path:  # Skip empty strings
            print_info(f"  - {path}")

def attempt_to_run_app():
    print_section("Application Launch Test")
    
    print_info("Attempting to launch application...")
    print_info("Will automatically terminate after 3 seconds.")
    
    try:
        # Start the application with a timeout
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{project_dir}:{env.get('PYTHONPATH', '')}"
        
        # Use subprocess to run the application for a short time
        process = subprocess.Popen(
            ["python", os.path.join(project_dir, "app.py")],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for 3 seconds
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            # Process is still running, which is good
            print_success("Application started successfully")
            # Terminate the process
            process.terminate()
            process.wait(timeout=5)
        else:
            # Process exited already, get return code
            return_code = process.poll()
            stdout, stderr = process.communicate()
            
            if return_code != 0:
                print_error(f"Application exited with error code {return_code}")
                if stderr:
                    print_info("Error output:")
                    print_info(stderr.decode())
            else:
                print_warning("Application exited quickly, but with success code")
                
    except Exception as e:
        print_error(f"Failed to start application: {str(e)}")

def main():
    print_header("CANDOR Health Check Utility")
    
    print_info(f"Running diagnostics at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_info(f"Project directory: {project_dir}")
    
    # Run all checks
    check_system()
    check_dependencies()
    check_nlp_resources()
    check_project_structure()
    check_desktop_launcher()
    check_python_path()
    
    # Final diagnostic - try running the app
    attempt_to_run_app()
    
    print_header("Health Check Complete")

if __name__ == "__main__":
    main()
