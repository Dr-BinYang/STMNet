import os
import torch
from model import *


class Exp_Basic(object):
    """
    Base experiment class for all experiments
    Handles device management and model initialization
    """

    def __init__(self, args):
        """
        Initialize base experiment

        Args:
            args: Configuration arguments
        """
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        """
        Build model - must be implemented by subclasses

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError
        return None

    def _acquire_device(self):
        """
        Acquire appropriate device (GPU if available and enabled, otherwise CPU)

        Returns:
            torch.device: Selected device
        """
        if torch.cuda.is_available() and self.args.use_gpu:
            device = torch.device("cuda")
            print(f">> Using GPU")
        else:
            device = torch.device("cpu")  # Fallback to CPU
            print(f">> Using CPU")

        return device

    def _get_data(self):
        """Get data - to be implemented by subclasses"""
        pass

    def vali(self):
        """Validation method - to be implemented by subclasses"""
        pass

    def train(self):
        """Training method - to be implemented by subclasses"""
        pass

    def test(self):
        """Testing method - to be implemented by subclasses"""
        pass