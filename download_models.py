#!/usr/bin/env python3
"""
Optimized script to download Hugging Face models locally with maximum speed.
Uses multiple optimization techniques for fastest downloads.
"""

import os
import sys
import subprocess
from pathlib import Path
from huggingface_hub import login, snapshot_download
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories for model storage."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for each model
    llama_dir = models_dir / "Llama-3.1-8B-Instruct"
    gemma_dir = models_dir / "Gemma-2-2B-IT"
    
    llama_dir.mkdir(exist_ok=True)
    gemma_dir.mkdir(exist_ok=True)
    
    return llama_dir, gemma_dir

def install_hf_transfer():
    """Install hf_transfer for faster downloads."""
    try:
        import hf_transfer
        logger.info("hf_transfer already installed")
        return True
    except ImportError:
        logger.info("Installing hf_transfer for faster downloads...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "huggingface_hub[hf_transfer]", "--upgrade"
            ])
            logger.info("hf_transfer installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to install hf_transfer: {e}")
            return False

def authenticate_huggingface():
    """Authenticate with Hugging Face using token from environment."""
    load_dotenv()
    
    token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    
    if not token:
        logger.error("No Hugging Face token found in environment variables.")
        logger.error("Please set HUGGINGFACE_TOKEN or HF_TOKEN environment variable.")
        sys.exit(1)
    
    try:
        login(token=token)
        logger.info("Successfully authenticated with Hugging Face")
        return True
    except Exception as e:
        logger.error(f"Failed to authenticate with Hugging Face: {e}")
        return False

def download_model_optimized(model_name, local_dir):
    """Download a model using the fastest method available."""
    logger.info(f"Starting optimized download of {model_name} to {local_dir}")
    
    start_time = time.time()
    
    # Set environment variable to enable hf_transfer
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    try:
        # Method 1: Try snapshot_download with hf_transfer (fastest for most cases)
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=8,  # Increased concurrent downloads
            local_files_only=False,
            cache_dir=None,  # Bypass cache for direct download
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully downloaded {model_name} in {elapsed_time:.2f} seconds")
        return True
        
    except Exception as e:
        logger.warning(f"snapshot_download failed for {model_name}: {e}")
        logger.info("Falling back to git clone method...")
        
        # Method 2: Fallback to git clone (sometimes faster for very large models)
        return download_with_git_clone(model_name, local_dir)

def download_with_git_clone(model_name, local_dir):
    """Alternative download method using git clone with LFS."""
    try:
        # Remove existing directory if it exists
        if local_dir.exists():
            import shutil
            shutil.rmtree(local_dir)
        
        # Use git clone with optimizations
        repo_url = f"https://huggingface.co/{model_name}"
        
        # Clone with shallow history and LFS optimizations
        cmd = [
            "git", "clone",
            "--depth", "1",  # Shallow clone
            "--single-branch",  # Only main branch
            repo_url,
            str(local_dir)
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        # Fetch LFS files
        subprocess.check_call(["git", "lfs", "fetch"], cwd=local_dir)
        
        logger.info(f"Successfully downloaded {model_name} using git clone")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Git clone failed for {model_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during git clone: {e}")
        return False

def download_with_cli(model_name, local_dir):
    """Download using huggingface-cli (alternative method)."""
    try:
        cmd = [
            "huggingface-cli", "download",
            model_name,
            "--local-dir", str(local_dir),
            "--local-dir-use-symlinks", "False"
        ]
        
        logger.info(f"Running CLI download: {' '.join(cmd)}")
        subprocess.check_call(cmd)
        
        logger.info(f"Successfully downloaded {model_name} using CLI")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"CLI download failed for {model_name}: {e}")
        return False

def check_git_lfs():
    """Check if git-lfs is installed."""
    try:
        subprocess.check_output(["git", "lfs", "--version"], stderr=subprocess.STDOUT)
        logger.info("Git LFS is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Git LFS not found. Install it for better performance with large files.")
        logger.warning("Install from: https://git-lfs.github.io/")
        return False

def main():
    """Main function to orchestrate the download process."""
    logger.info("Starting optimized model download script")
    
    # Check and install dependencies
    logger.info("Checking dependencies...")
    check_git_lfs()
    
    # Install hf_transfer for speed boost
    install_hf_transfer()
    
    # Setup directories
    logger.info("Setting up directories...")
    llama_dir, gemma_dir = setup_directories()
    
    # Authenticate with Hugging Face
    logger.info("Authenticating with Hugging Face...")
    if not authenticate_huggingface():
        sys.exit(1)
    
    # Models to download
    models = [
        ("meta-llama/Llama-3.1-8B-Instruct", llama_dir),
        # ("google/gemma-2-2b-it", gemma_dir)
    ]
    
    success_count = 0
    total_start_time = time.time()
    
    # # Option 1: Sequential download (more stable)
    # for model_name, local_dir in models:
    #     if download_model_optimized(model_name, local_dir):
    #         success_count += 1
    #     else:
    #         logger.warning(f"All download methods failed for {model_name}")
    
    # Option 2: Parallel download (uncomment to use - may be unstable)
    def download_wrapper(args):
        model_name, local_dir = args
        return download_model_optimized(model_name, local_dir)
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(download_wrapper, models))
        success_count = sum(results)
    
    total_elapsed_time = time.time() - total_start_time
    
    # Summary
    logger.info(f"Download completed in {total_elapsed_time:.2f} seconds")
    logger.info(f"{success_count}/{len(models)} models downloaded successfully")
    
    if success_count == len(models):
        logger.info("All models downloaded successfully!")
        print("\n✅ All models downloaded successfully!")
        print(f"Total time: {total_elapsed_time:.2f} seconds")
        print(f"Models saved to:")
        for _, local_dir in models:
            print(f"  - {local_dir}")
    else:
        logger.warning("Some models failed to download. Check the logs above.")
        print(f"\n⚠️  {success_count}/{len(models)} models downloaded successfully.")

if __name__ == "__main__":
    main()