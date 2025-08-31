"""
Advanced device detection for multi-platform support.
Supports CUDA (NVIDIA), MPS/MLX (Apple Silicon), ROCm (AMD), and CPU fallback.
"""
import os
import platform
import subprocess
import sys
from typing import Dict, Optional, Tuple
import torch
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """Comprehensive device detection and management."""
    
    def __init__(self):
        self.device_info = self._detect_all_devices()
        self.optimal_device = self._select_optimal_device()
        
    def _detect_all_devices(self) -> Dict[str, Dict]:
        """Detect all available compute devices."""
        devices = {
            'cuda': self._detect_cuda(),
            'mps': self._detect_mps(),
            'mlx': self._detect_mlx(),
            'rocm': self._detect_rocm(),
            'cpu': {'available': True, 'device_count': 1, 'memory': self._get_system_memory()}
        }
        return devices
    
    def _detect_cuda(self) -> Dict:
        """Detect NVIDIA CUDA devices."""
        info = {'available': False, 'device_count': 0, 'devices': []}
        
        if torch.cuda.is_available():
            info['available'] = True
            info['device_count'] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                info['devices'].append({
                    'index': i,
                    'name': device_props.name,
                    'memory': device_props.total_memory,
                    'compute_capability': f"{device_props.major}.{device_props.minor}"
                })
                
        return info
    
    def _detect_mps(self) -> Dict:
        """Detect Apple Metal Performance Shaders."""
        info = {'available': False, 'device_count': 0}
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['available'] = True
            info['device_count'] = 1
            info['device'] = 'Apple Silicon GPU'
            
        return info
    
    def _detect_mlx(self) -> Dict:
        """Detect Apple MLX framework availability."""
        info = {'available': False, 'unified_memory': False}
        
        # Check if we're on macOS with Apple Silicon
        if platform.system() == 'Darwin':
            try:
                # Try to detect Apple Silicon
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True, timeout=5)
                cpu_brand = result.stdout.strip()
                
                if 'Apple' in cpu_brand or any(chip in cpu_brand for chip in ['M1', 'M2', 'M3', 'M4']):
                    info['available'] = True
                    info['cpu_brand'] = cpu_brand
                    info['unified_memory'] = True
                    
                    # Try to import mlx to verify it's installed
                    try:
                        import mlx.core as mx
                        info['mlx_installed'] = True
                        info['mlx_version'] = getattr(mx, '__version__', 'unknown')
                    except ImportError:
                        info['mlx_installed'] = False
                        logger.warning("MLX not installed. Install with: pip install mlx")
                        
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                logger.debug("Could not detect CPU brand via sysctl")
                
        return info
    
    def _detect_rocm(self) -> Dict:
        """Detect AMD ROCm devices."""
        info = {'available': False, 'device_count': 0, 'devices': []}
        
        try:
            # Check if ROCm is available
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                info['available'] = True
                info['hip_version'] = torch.version.hip
                
                # Try to get device count
                try:
                    if hasattr(torch.cuda, 'device_count'):  # ROCm uses cuda namespace
                        info['device_count'] = torch.cuda.device_count()
                        
                        for i in range(info['device_count']):
                            try:
                                device_props = torch.cuda.get_device_properties(i)
                                info['devices'].append({
                                    'index': i,
                                    'name': device_props.name,
                                    'memory': device_props.total_memory
                                })
                            except Exception as e:
                                logger.debug(f"Could not get properties for ROCm device {i}: {e}")
                                
                except Exception as e:
                    logger.debug(f"Could not get ROCm device count: {e}")
                    
            # Alternative: Check for rocm-smi command
            elif self._check_rocm_smi():
                info['rocm_smi_available'] = True
                devices = self._get_rocm_devices()
                if devices:
                    info['available'] = True
                    info['device_count'] = len(devices)
                    info['devices'] = devices
                    
        except Exception as e:
            logger.debug(f"Error detecting ROCm: {e}")
            
        return info
    
    def _check_rocm_smi(self) -> bool:
        """Check if rocm-smi command is available."""
        try:
            result = subprocess.run(['rocm-smi', '--version'], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _get_rocm_devices(self) -> list:
        """Get ROCm device info using rocm-smi."""
        devices = []
        try:
            result = subprocess.run(['rocm-smi', '--showproductname', '--showmeminfo', 'vram'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                # Parse rocm-smi output (this is a simplified parser)
                for i, line in enumerate(lines):
                    if 'GPU' in line and 'Card series' in line:
                        devices.append({
                            'index': i,
                            'name': line.strip(),
                            'memory': 'Unknown'  # Would need more parsing
                        })
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("Could not run rocm-smi to get device info")
            
        return devices
    
    def _get_system_memory(self) -> Optional[int]:
        """Get system RAM in bytes."""
        try:
            if platform.system() == 'Darwin':
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                      capture_output=True, text=True, timeout=5)
                return int(result.stdout.strip())
            elif platform.system() == 'Linux':
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            return int(line.split()[1]) * 1024  # Convert KB to bytes
        except Exception as e:
            logger.debug(f"Could not get system memory: {e}")
            
        return None
    
    def _select_optimal_device(self) -> Tuple[str, str]:
        """Select the best available device."""
        # Priority order: CUDA > MLX (if Apple Silicon) > MPS > ROCm > CPU
        
        if self.device_info['cuda']['available']:
            return 'cuda', 'cuda:0'
        
        if self.device_info['mlx']['available'] and self.device_info['mlx'].get('mlx_installed', False):
            return 'mlx', 'mlx'
        
        if self.device_info['mps']['available']:
            return 'mps', 'mps'
        
        if self.device_info['rocm']['available']:
            return 'rocm', 'cuda:0'  # ROCm uses cuda namespace
        
        return 'cpu', 'cpu'
    
    def get_device(self, preference: Optional[str] = None) -> str:
        """Get device string for PyTorch/framework use."""
        if preference == 'auto':
            return self.optimal_device[1]
        
        if preference and preference in ['cuda', 'mps', 'mlx', 'rocm', 'cpu']:
            if self.device_info[preference]['available']:
                if preference == 'cuda':
                    return 'cuda:0'
                elif preference == 'mps':
                    return 'mps'
                elif preference == 'mlx':
                    return 'mlx'
                elif preference == 'rocm':
                    return 'cuda:0'  # ROCm uses cuda namespace
                else:
                    return 'cpu'
            else:
                logger.warning(f"{preference} not available, falling back to {self.optimal_device[1]}")
                return self.optimal_device[1]
        
        return self.optimal_device[1]
    
    def get_torch_dtype(self, device_type: Optional[str] = None) -> torch.dtype:
        """Get optimal torch dtype for the device."""
        if device_type is None:
            device_type = self.optimal_device[0]
        
        # Use float16 for GPU acceleration, float32 for CPU and compatibility
        if device_type in ['cuda', 'mps', 'mlx'] and self.device_info[device_type]['available']:
            return torch.float16
        elif device_type == 'rocm' and self.device_info['rocm']['available']:
            return torch.float16  # ROCm generally supports float16
        else:
            return torch.float32
    
    def get_device_info(self) -> Dict:
        """Get comprehensive device information."""
        return {
            'detected_devices': self.device_info,
            'optimal_device': self.optimal_device,
            'platform': platform.system(),
            'python_version': sys.version,
            'pytorch_version': torch.__version__
        }
    
    def print_device_info(self) -> None:
        """Print detailed device information."""
        print("=== Device Detection Results ===")
        print(f"Platform: {platform.system()} {platform.machine()}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"PyTorch: {torch.__version__}")
        print(f"Optimal Device: {self.optimal_device[0]} ({self.optimal_device[1]})")
        print()
        
        for device_type, info in self.device_info.items():
            status = "✅" if info['available'] else "❌"
            print(f"{status} {device_type.upper()}: {info['available']}")
            
            if info['available'] and device_type == 'cuda':
                for dev in info['devices']:
                    memory_gb = dev['memory'] / (1024**3)
                    print(f"    GPU {dev['index']}: {dev['name']} ({memory_gb:.1f} GB)")
            
            elif info['available'] and device_type == 'mlx':
                print(f"    CPU: {info.get('cpu_brand', 'Unknown')}")
                print(f"    MLX Installed: {info.get('mlx_installed', False)}")
                if info.get('mlx_version'):
                    print(f"    MLX Version: {info['mlx_version']}")
            
            elif info['available'] and device_type == 'rocm':
                if info.get('hip_version'):
                    print(f"    HIP Version: {info['hip_version']}")
                for dev in info.get('devices', []):
                    print(f"    GPU {dev['index']}: {dev['name']}")
        
        print()


# Global device manager instance
device_manager = DeviceManager()


def get_optimal_device() -> str:
    """Get the optimal device for the current system."""
    return device_manager.get_device('auto')


def get_device_manager() -> DeviceManager:
    """Get the global device manager instance."""
    return device_manager


def detect_and_print_devices() -> None:
    """Convenience function to detect and print all devices."""
    device_manager.print_device_info()


if __name__ == "__main__":
    # Test device detection
    detect_and_print_devices()