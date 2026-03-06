"""ONNX Model Runner - Python interface for ONNX Runtime inference"""
from typing import Dict, Any, Optional, List
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ModelRunner:
    """Manages ONNX model loading and inference for Synthesus."""

    def __init__(self, model_dir: str = "./models", device: str = "auto"):
        self.model_dir = model_dir
        self.device = device
        self._sessions: Dict[str, Any] = {}
        self._available = False
        self._try_import()

    def _try_import(self):
        try:
            import onnxruntime as ort
            self._ort = ort
            # Select provider based on device
            providers = self._select_providers()
            self._providers = providers
            self._available = True
            logger.info(f"ONNX Runtime available. Providers: {providers}")
        except ImportError:
            logger.warning("onnxruntime not installed. Model runner in stub mode.")
            self._ort = None

    def _select_providers(self) -> List[str]:
        if not self._ort:
            return []
        available = self._ort.get_available_providers()
        if self.device == "auto":
            for pref in ["CUDAExecutionProvider", "ROCMExecutionProvider",
                         "TensorrtExecutionProvider", "CPUExecutionProvider"]:
                if pref in available:
                    return [pref, "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def load_model(self, model_name: str, model_path: Optional[str] = None) -> bool:
        """Load an ONNX model by name."""
        if not self._available:
            logger.warning(f"Cannot load {model_name}: ONNX Runtime not available")
            return False
        path = model_path or f"{self.model_dir}/{model_name}.onnx"
        try:
            opts = self._ort.SessionOptions()
            opts.graph_optimization_level = (
                self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            session = self._ort.InferenceSession(
                path, sess_options=opts, providers=self._providers
            )
            self._sessions[model_name] = session
            logger.info(f"Loaded model: {model_name} from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return False

    def run(self, model_name: str, inputs: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        """Run inference on a loaded model."""
        if model_name not in self._sessions:
            logger.error(f"Model not loaded: {model_name}")
            return None
        try:
            session = self._sessions[model_name]
            output_names = [o.name for o in session.get_outputs()]
            results = session.run(output_names, inputs)
            return dict(zip(output_names, results))
        except Exception as e:
            logger.error(f"Inference error on {model_name}: {e}")
            return None

    def unload_model(self, model_name: str):
        if model_name in self._sessions:
            del self._sessions[model_name]
            logger.info(f"Unloaded model: {model_name}")

    def list_loaded(self) -> List[str]:
        return list(self._sessions.keys())

    @property
    def is_available(self) -> bool:
        return self._available
