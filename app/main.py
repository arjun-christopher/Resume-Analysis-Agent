#!/usr/bin/env python3
"""
Main entry point for RAG-Resume application
Automatically detects environment and runs Streamlit app appropriately
"""

import os
import sys
import platform
import subprocess
from pathlib import Path


def detect_environment():
    """
    Detect the current runtime environment
    
    Returns:
        str: Environment type (windows, linux, macos, codespace, docker, wsl)
    """
    env_info = {
        'platform': platform.system(),
        'is_codespace': os.getenv('CODESPACES', 'false').lower() == 'true',
        'is_docker': os.path.exists('/.dockerenv'),
        'is_wsl': 'microsoft' in platform.uname().release.lower(),
        'is_github_actions': os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true',
        'is_replit': os.getenv('REPL_ID') is not None,
        'is_gitpod': os.getenv('GITPOD_WORKSPACE_ID') is not None,
    }
    
    # Determine primary environment
    if env_info['is_codespace']:
        return 'codespace'
    elif env_info['is_docker']:
        return 'docker'
    elif env_info['is_wsl']:
        return 'wsl'
    elif env_info['is_github_actions']:
        return 'github_actions'
    elif env_info['is_replit']:
        return 'replit'
    elif env_info['is_gitpod']:
        return 'gitpod'
    elif env_info['platform'] == 'Windows':
        return 'windows'
    elif env_info['platform'] == 'Linux':
        return 'linux'
    elif env_info['platform'] == 'Darwin':
        return 'macos'
    else:
        return 'unknown'


def get_python_executable():
    """
    Get the appropriate Python executable for the current environment
    
    Returns:
        str: Python executable path/command
    """
    # Try to use the current Python interpreter
    return sys.executable


def get_streamlit_command():
    """
    Get the appropriate Streamlit command for the current environment
    
    Returns:
        list: Command to run Streamlit as a list of arguments
    """
    python_exe = get_python_executable()
    
    # Try to find streamlit executable
    streamlit_paths = [
        'streamlit',  # If in PATH
        os.path.join(os.path.dirname(python_exe), 'streamlit'),  # Same dir as Python
        os.path.join(os.path.dirname(python_exe), 'Scripts', 'streamlit.exe'),  # Windows
        os.path.join(os.path.dirname(python_exe), 'Scripts', 'streamlit'),  # Windows (no .exe)
    ]
    
    # Check which streamlit command works
    for streamlit_cmd in streamlit_paths:
        try:
            result = subprocess.run(
                [streamlit_cmd, '--version'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return [streamlit_cmd]
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            continue
    
    # Fallback: use python -m streamlit
    return [python_exe, '-m', 'streamlit']


def get_streamlit_config(environment):
    """
    Get Streamlit configuration based on environment
    
    Args:
        environment: Detected environment type
        
    Returns:
        dict: Configuration options for Streamlit
    """
    config = {
        'server.port': 8501,
        'server.headless': True,
        'server.enableCORS': False,
        'server.enableXsrfProtection': True,
    }
    
    # Environment-specific configurations
    if environment == 'codespace':
        config['server.port'] = int(os.getenv('PORT', 8501))
        config['server.address'] = '0.0.0.0'
        config['browser.gatherUsageStats'] = False
    elif environment == 'docker':
        config['server.address'] = '0.0.0.0'
        config['browser.gatherUsageStats'] = False
    elif environment == 'replit':
        config['server.address'] = '0.0.0.0'
        config['server.port'] = 8080
    elif environment == 'gitpod':
        config['server.address'] = '0.0.0.0'
        config['server.port'] = 8501
    elif environment in ['linux', 'macos', 'wsl']:
        config['server.address'] = 'localhost'
        config['browser.serverAddress'] = 'localhost'
    elif environment == 'windows':
        config['server.address'] = 'localhost'
        config['browser.serverAddress'] = 'localhost'
    
    return config


def build_streamlit_args(environment, config):
    """
    Build Streamlit command line arguments
    
    Args:
        environment: Detected environment type
        config: Configuration dictionary
        
    Returns:
        list: Command line arguments for Streamlit
    """
    args = ['run']
    
    # Add configuration arguments
    for key, value in config.items():
        args.append(f'--{key}')
        args.append(str(value))
    
    # Add the app file
    app_file = Path(__file__).parent / 'streamlit_app.py'
    args.append(str(app_file))
    
    return args


def setup_environment():
    """
    Set up environment variables and paths
    """
    # Get project root (parent of app directory)
    project_root = Path(__file__).parent.parent.absolute()
    app_dir = Path(__file__).parent.absolute()
    
    # Add to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    
    # Set PYTHONPATH environment variable
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_paths = [str(project_root), str(app_dir)]
    
    if current_pythonpath:
        new_paths.append(current_pythonpath)
    
    os.environ['PYTHONPATH'] = os.pathsep.join(new_paths)


def check_dependencies():
    """
    Check if required dependencies are installed
    
    Returns:
        tuple: (bool, list) - (all_installed, missing_packages)
    """
    required_packages = ['streamlit', 'dotenv']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'dotenv':
                __import__('dotenv')
            else:
                __import__(package)
        except ImportError:
            missing.append(package if package != 'dotenv' else 'python-dotenv')
    
    return len(missing) == 0, missing


def print_banner(environment):
    """Print welcome banner with environment info"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              RAG-Resume Analysis Application                 ║
║                                                              ║
║        AI-Powered Multi-Resume Analysis with RAG             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print(f"🖥️  Environment Detected: {environment.upper()}")
    print(f"🐍 Python Version: {platform.python_version()}")
    print(f"📁 Working Directory: {Path.cwd()}")
    print("─" * 64)


def main():
    """Main entry point"""
    # Detect environment
    environment = detect_environment()
    
    # Print banner
    print_banner(environment)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    all_installed, missing = check_dependencies()
    
    if not all_installed:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print(f"\n💡 Install them with:")
        print(f"   pip install {' '.join(missing)}")
        return 1
    
    print("✅ All dependencies installed")
    
    # Setup environment
    print("⚙️  Setting up environment...")
    setup_environment()
    
    # Get Streamlit command
    print("🚀 Preparing to launch Streamlit...")
    streamlit_cmd = get_streamlit_command()
    config = get_streamlit_config(environment)
    args = build_streamlit_args(environment, config)
    
    # Build full command
    full_command = streamlit_cmd + args
    
    print(f"📋 Command: {' '.join(full_command)}")
    print("─" * 64)
    
    # Environment-specific instructions
    if environment == 'codespace':
        print("🌐 GitHub Codespaces detected")
        print("   The app will be available via the forwarded port")
        print("   Look for the 'Ports' tab and click the globe icon")
    elif environment == 'docker':
        print("🐳 Docker environment detected")
        print("   Make sure port 8501 is exposed in your docker-compose.yml")
    elif environment == 'replit':
        print("🔄 Replit environment detected")
        print("   The app will open automatically in the Replit webview")
    elif environment == 'gitpod':
        print("🦊 Gitpod environment detected")
        print("   The app will be available via the browser preview")
    elif environment in ['windows', 'linux', 'macos', 'wsl']:
        port = config.get('server.port', 8501)
        print(f"🌐 Local development environment")
        print(f"   The app will open in your browser at: http://localhost:{port}")
    
    print("\n🎯 Starting RAG-Resume Application...")
    print("   Press Ctrl+C to stop the server")
    print("─" * 64)
    print()
    
    # Run Streamlit
    try:
        subprocess.run(full_command)
        return 0
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down RAG-Resume Application...")
        print("   Thank you for using the app!")
        return 0
    except Exception as e:
        print(f"\n❌ Error running Streamlit: {e}")
        print(f"\n💡 Try running manually:")
        print(f"   streamlit run {Path(__file__).parent / 'streamlit_app.py'}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
