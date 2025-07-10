import os
import sys
import subprocess
import json
from pathlib import Path

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print('='*60)

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {str(e)}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print_step(1, "Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print("Please install Python 3.8+ from https://python.org")
        return False
    
    print("✅ Python version is compatible")
    return True

def install_requirements():
    """Install required packages"""
    print_step(2, "Installing Required Packages")
    
    packages = [
        "streamlit",
        "fastf1",
        "scikit-learn",
        "xgboost",
        "plotly",
        "pandas",
        "numpy",
        "requests"
    ]
    
    print("📦 Installing packages:")
    for package in packages:
        print(f"  - {package}")
    
    command = f"pip install {' '.join(packages)}"
    return run_command(command, "Installing packages")

def create_directory_structure():
    """Create necessary directories"""
    print_step(3, "Creating Directory Structure")
    
    directories = [
        "cache",
        "data",
        ".streamlit"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")
    
    return True

def create_streamlit_config():
    """Create Streamlit configuration file"""
    print_step(4, "Creating Streamlit Configuration")
    
    config_content = """[server]
headless = true
address = "0.0.0.0"
port = 8501

[theme]
base = "light"
primaryColor = "#ff6b6b"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
"""
    
    config_path = Path(".streamlit/config.toml")
    try:
        with open(config_path, 'w') as f:
            f.write(config_content)
        print("✅ Streamlit configuration created")
        return True
    except Exception as e:
        print(f"❌ Failed to create config: {str(e)}")
        return False

def check_required_files():
    """Check if all required Python files exist"""
    print_step(5, "Checking Required Files")
    
    required_files = [
        "app.py",
        "data_collector.py",
        "feature_engineer.py",
        "ml_models.py",
        "predictor.py",
        "json_database.py",
        "driver_analytics.py",
        "utils.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"✅ {file} found")
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        print("Please make sure all Python files are in the same directory as this setup script")
        return False
    
    print("✅ All required files found")
    return True

def create_launch_script():
    """Create a launch script for easy startup"""
    print_step(6, "Creating Launch Script")
    
    # Windows batch file
    bat_content = """@echo off
echo Starting F1 Race Predictor...
echo Open your browser to: http://localhost:8501
streamlit run app.py
pause
"""
    
    # Unix shell script
    sh_content = """#!/bin/bash
echo "Starting F1 Race Predictor..."
echo "Open your browser to: http://localhost:8501"
streamlit run app.py
"""
    
    try:
        # Create Windows launcher
        with open("start_f1_predictor.bat", 'w') as f:
            f.write(bat_content)
        
        # Create Unix launcher
        with open("start_f1_predictor.sh", 'w') as f:
            f.write(sh_content)
        
        # Make shell script executable on Unix systems
        if os.name != 'nt':  # Not Windows
            os.chmod("start_f1_predictor.sh", 0o755)
        
        print("✅ Launch scripts created:")
        print("  - Windows: start_f1_predictor.bat")
        print("  - Unix/Mac: start_f1_predictor.sh")
        return True
    except Exception as e:
        print(f"❌ Failed to create launch scripts: {str(e)}")
        return False

def create_readme():
    """Create a README file with instructions"""
    print_step(7, "Creating README")
    
    readme_content = """# F1 Race Predictor - Local Installation

## Quick Start

1. **Run the application:**
   - Windows: Double-click `start_f1_predictor.bat`
   - Mac/Linux: Run `./start_f1_predictor.sh` or `streamlit run app.py`

2. **Open your browser** and go to: http://localhost:8501

## First Time Usage

1. **Data Collection**: Go to "Data Collection" page and select seasons (2022-2024 recommended)
2. **Train Models**: Once data is collected, go to "Model Training" to train prediction models
3. **Make Predictions**: Use "Race Predictions" to predict future race outcomes
4. **Validate Accuracy**: Check "2024 Validation" to see how accurate predictions are

## Features

- 📊 Real F1 data collection from FastF1 API
- 🤖 Multiple machine learning models (Random Forest, XGBoost, Neural Networks)
- 🔮 2025 race predictions
- ✅ Accuracy validation against 2024 results
- 📈 Driver analytics and performance tracking
- 🗄️ Lightweight JSON database for fast storage

## Requirements

- Python 3.8 or higher
- Internet connection for data collection
- About 1GB free space for F1 data cache

## Troubleshooting

If you encounter issues:
1. Check Python version: `python --version`
2. Reinstall packages: `pip install -r requirements.txt`
3. Clear cache: Delete `cache` and `data` folders
4. Check Streamlit installation: `streamlit hello`

## Data Collection

The first data collection will take 10-15 minutes as it downloads real F1 telemetry data.
This data is cached locally for faster future access.

## Support

For issues or questions, check the application logs in your terminal.
"""
    
    try:
        with open("README_LOCAL.md", 'w') as f:
            f.write(readme_content)
        print("✅ README created: README_LOCAL.md")
        return True
    except Exception as e:
        print(f"❌ Failed to create README: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("🏎️  F1 Race Predictor - Local Setup")
    print("This script will set up the F1 Race Predictor to run on your computer")
    
    # Run setup steps
    steps = [
        check_python_version,
        install_requirements,
        create_directory_structure,
        create_streamlit_config,
        check_required_files,
        create_launch_script,
        create_readme
    ]
    
    success_count = 0
    for step in steps:
        if step():
            success_count += 1
        else:
            print(f"\n❌ Setup failed at step: {step.__name__}")
            break
    
    print(f"\n{'='*60}")
    if success_count == len(steps):
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("📋 Next steps:")
        print("1. Run the application:")
        print("   - Windows: Double-click start_f1_predictor.bat")
        print("   - Mac/Linux: Run ./start_f1_predictor.sh")
        print("2. Open http://localhost:8501 in your browser")
        print("3. Start with Data Collection to gather F1 data")
        print("4. Train models and make predictions!")
        print(f"\n📖 See README_LOCAL.md for detailed instructions")
    else:
        print("❌ SETUP INCOMPLETE")
        print("Please fix the issues above and run this script again")
    
    print('='*60)

if __name__ == "__main__":
    main()