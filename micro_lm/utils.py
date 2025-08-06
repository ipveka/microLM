"""Utility functions for microLM."""

import torch
import logging


def get_device(prefer_gpu: bool = True, verbose: bool = True) -> torch.device:
    """
    Get the best available device for computation.
    
    Args:
        prefer_gpu (bool): Whether to prefer GPU over CPU when available
        verbose (bool): Whether to print device information
        
    Returns:
        torch.device: The selected device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU] Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = torch.device('cpu')
        if verbose:
            print("[CPU] Using CPU")
    
    return device


def move_to_device(model: torch.nn.Module, device: torch.device = None, verbose: bool = True) -> torch.nn.Module:
    """
    Move model to the specified device.
    
    Args:
        model (torch.nn.Module): The model to move
        device (torch.device, optional): Target device. If None, auto-detect best device
        verbose (bool): Whether to print device information
        
    Returns:
        torch.nn.Module: The model on the target device
    """
    if device is None:
        device = get_device(verbose=verbose)
    
    model = model.to(device)
    
    if verbose:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[MODEL] Model moved to {device} ({param_count:,} parameters)")
    
    return model


def setup_device_and_model(model: torch.nn.Module, device: str = 'auto', verbose: bool = True) -> tuple[torch.nn.Module, torch.device]:
    """
    Setup device and move model to it.
    
    Args:
        model (torch.nn.Module): The model to setup
        device (str): Device specification ('auto', 'cuda', 'cpu', or specific device)
        verbose (bool): Whether to print setup information
        
    Returns:
        tuple: (model_on_device, device)
    """
    if device == 'auto':
        target_device = get_device(verbose=verbose)
    else:
        target_device = torch.device(device)
        if verbose:
            print(f"[DEVICE] Using specified device: {target_device}")
    
    model = move_to_device(model, target_device, verbose=verbose)
    
    return model, target_device


def get_device_info() -> dict:
    """
    Get detailed information about available devices.
    
    Returns:
        dict: Device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cpu_count': torch.get_num_threads(),
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_current_device': torch.cuda.current_device(),
            'cuda_device_name': torch.cuda.get_device_name(0),
            'cuda_memory_total': torch.cuda.get_device_properties(0).total_memory,
            'cuda_memory_allocated': torch.cuda.memory_allocated(0),
            'cuda_memory_cached': torch.cuda.memory_reserved(0),
        })
    
    return info


def print_device_info():
    """
    Print detailed device information.
    """
    info = get_device_info()
    
    print("\n=== Device Information ===")
    print(f"   CPU Threads: {info['cpu_count']}")
    print(f"   CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"   GPU Count: {info['cuda_device_count']}")
        print(f"   Current GPU: {info['cuda_current_device']}")
        print(f"   GPU Name: {info['cuda_device_name']}")
        print(f"   GPU Memory: {info['cuda_memory_total'] / 1e9:.1f}GB total")
        if info['cuda_memory_allocated'] > 0:
            print(f"   GPU Memory Used: {info['cuda_memory_allocated'] / 1e9:.1f}GB")
    print()