#!/usr/bin/env python3
"""
Automated Setup Script for RAG Resume Analysis System
Handles: requirements installation, .env creation, Ollama setup
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_step(step_num, total_steps, text):
    """Print formatted step"""
    print(f"{Colors.CYAN}[{step_num}/{total_steps}]{Colors.ENDC} {Colors.BOLD}{text}{Colors.ENDC}")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")


def check_python_version():
    """Check if Python version is compatible"""
    print_step(1, 6, "Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required. You have Python {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")


def install_requirements():
    """Install Python packages from requirements.txt"""
    print_step(2, 6, "Installing Python packages...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print_error("requirements.txt not found!")
        sys.exit(1)
    
    try:
        print_info("This may take a few minutes...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print_success("All packages installed successfully")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install packages: {e}")
        print_info("Try running manually: pip install -r requirements.txt")
        sys.exit(1)


def create_env_file():
    """Create .env file with default configuration"""
    print_step(3, 6, "Creating .env configuration file...")
    
    env_file = Path(".env")
    
    if env_file.exists():
        response = input(f"{Colors.YELLOW}⚠️  .env file already exists. Overwrite? (y/N): {Colors.ENDC}").lower()
        if response != 'y':
            print_info("Keeping existing .env file")
            return
    
    env_content = """# LLM Configuration
# The system will try providers in the order specified below

# LLM Fallback Order (comma-separated, no spaces)
# Recommended: google (Gemini) first, then ollama for low-end systems
LLM_FALLBACK_ORDER=google,ollama

# LLM Parameters
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048
LLM_TIMEOUT=30

# ============================================================================
# GOOGLE GEMINI (Primary - Recommended)
# ============================================================================
# Get your free API key: https://makersuite.google.com/app/apikey
ENABLE_GOOGLE=true
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-2.5-flash

# ============================================================================
# OLLAMA (Fallback - Local/Low-end Systems)
# ============================================================================
# Download from: https://ollama.ai
# Install a lightweight model: ollama pull qwen2.5:1.5b
ENABLE_OLLAMA=true
OLLAMA_MODEL=qwen2.5:1.5b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=60

# ============================================================================
# OPENAI (Optional - Advanced Features)
# ============================================================================
# Get API key: https://platform.openai.com/api-keys
# Uncomment to enable OpenAI
#ENABLE_OPENAI=true
#OPENAI_API_KEY=your_openai_api_key_here
#OPENAI_MODEL=gpt-4o-mini

# ============================================================================
# ANTHROPIC CLAUDE (Optional - Advanced Features)
# ============================================================================
# Get API key: https://console.anthropic.com/
# Uncomment to enable Claude
#ENABLE_ANTHROPIC=true
#ANTHROPIC_API_KEY=your_anthropic_api_key_here
#ANTHROPIC_MODEL=claude-3-haiku-20240307

# ============================================================================
# RAG CONFIGURATION
# ============================================================================
# Section-based chunking parameters
MAX_CHUNK_SIZE=1000
MIN_CHUNK_SIZE=100

# Hybrid search weights
SEMANTIC_WEIGHT=0.7
BM25_WEIGHT=0.3

# Retrieval settings
TOP_K_RESULTS=5
"""
    
    try:
        env_file.write_text(env_content)
        print_success(".env file created successfully")
        print_info("Edit .env file to add your Google Gemini API key")
    except Exception as e:
        print_error(f"Failed to create .env file: {e}")


def create_data_directories():
    """Create necessary data directories"""
    print_step(4, 6, "Creating data directories...")
    
    directories = [
        "data",
        "data/uploads",
        "data/index",
        "data/advanced_rag"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print_success("All data directories created")


def check_ollama():
    """Check if Ollama is installed and offer to install"""
    print_step(5, 6, "Checking Ollama installation...")
    
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print_success("Ollama is installed")
            
            # Check if model is available
            try:
                models_result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if "qwen2.5:1.5b" in models_result.stdout or "qwen2.5" in models_result.stdout:
                    print_success("Qwen2.5 model is available")
                else:
                    print_warning("Qwen2.5 model not found")
                    response = input(f"{Colors.YELLOW}Would you like to pull qwen2.5:1.5b model? (y/N): {Colors.ENDC}").lower()
                    if response == 'y':
                        print_info("Pulling model (this may take a few minutes)...")
                        subprocess.run(["ollama", "pull", "qwen2.5:1.5b"])
            except:
                print_info("Could not check available models")
                
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_warning("Ollama is not installed")
        print_info("Ollama provides local LLM as fallback option")
        print_info("Install from: https://ollama.ai")
        
        if platform.system() == "Linux":
            response = input(f"{Colors.YELLOW}Would you like to install Ollama now? (y/N): {Colors.ENDC}").lower()
            if response == 'y':
                try:
                    print_info("Installing Ollama...")
                    subprocess.run([
                        "curl", "-fsSL", "https://ollama.ai/install.sh"
                    ], check=True, stdout=subprocess.PIPE)
                    print_success("Ollama installed. Run 'ollama pull qwen2.5:1.5b' to download the model")
                except subprocess.CalledProcessError:
                    print_error("Failed to install Ollama")
        else:
            print_info(f"Please install Ollama manually for {platform.system()}")


def setup_gemini():
    """Guide user through Gemini API key setup"""
    print_step(6, 6, "Setting up Google Gemini API...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print_error(".env file not found!")
        return
    
    env_content = env_file.read_text()
    
    if "your_google_api_key_here" in env_content:
        print_info("Google Gemini API key not configured yet")
        print_info("Get your free API key: https://makersuite.google.com/app/apikey")
        
        response = input(f"{Colors.YELLOW}Do you have a Google Gemini API key? (y/N): {Colors.ENDC}").lower()
        if response == 'y':
            api_key = input(f"{Colors.CYAN}Enter your API key: {Colors.ENDC}").strip()
            if api_key and api_key != "your_google_api_key_here":
                # Update .env file
                env_content = env_content.replace("GOOGLE_API_KEY=your_google_api_key_here", f"GOOGLE_API_KEY={api_key}")
                env_file.write_text(env_content)
                print_success("Gemini API key configured!")
            else:
                print_warning("Invalid API key. You can edit .env file manually later")
        else:
            print_info("You can add your API key later by editing .env file")
            print_info("System will use Ollama as fallback")
    else:
        print_success("Gemini API key already configured")


def print_next_steps():
    """Print next steps for the user"""
    print_header("Setup Complete! 🎉")
    
    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}\n")
    
    print(f"{Colors.CYAN}1. Configure Gemini API Key:{Colors.ENDC}")
    print("   - Get free key: https://makersuite.google.com/app/apikey")
    print("   - Edit .env file and add your key")
    print()
    
    print(f"{Colors.CYAN}2. (Optional) Install Ollama for fallback:{Colors.ENDC}")
    print("   - Visit: https://ollama.ai")
    print("   - Run: ollama pull qwen2.5:1.5b")
    print()
    
    print(f"{Colors.CYAN}3. Run the application:{Colors.ENDC}")
    print(f"   {Colors.GREEN}streamlit run app/streamlit_app.py{Colors.ENDC}")
    print()
    
    print(f"{Colors.CYAN}4. Upload resumes and start analyzing!{Colors.ENDC}")
    print()
    
    print(f"{Colors.BOLD}Documentation:{Colors.ENDC}")
    print("   - README.md - Full documentation")
    print()


def main():
    """Main setup function"""
    print_header("RAG Resume Analysis System - Automated Setup")
    
    try:
        check_python_version()
        install_requirements()
        create_env_file()
        create_data_directories()
        check_ollama()
        setup_gemini()
        print_next_steps()
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Setup cancelled by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
