"""
Adaptive IMU gating for VINGS-Mono.

Contribution: Option 2 — Uncertainty-Weighted VO↔VIO Fusion.

Replaces the hardcoded var_g < 0.25 threshold with a continuous gating
function that re-weights the IMU factor's information matrix based on 
visual rendering uncertainty. When the Gaussian map renders cleanly 
(low photometric loss), we trust the visual tracker more and down-weight 
IMU. When rendering loss is high (visual uncertainty), we up-weight IMU.

This is the flagship contribution of the VINGS-Mono IMU Investigation
(Thakkar, CMU ECE, 2026).
"""
import numpy as np
from collections import deque


class AdaptiveIMUGate:
    """
    Adaptive IMU factor weighting based on Gaussian rendering uncertainty.
    
    Args:
        w_min: Minimum information matrix scale (down-weight bound).
        w_max: Maximum information matrix scale (up-weight bound).
        history_size: Running window for loss normalization.
        loss_sensitivity: How sharply weight responds to loss changes.
        excitation_threshold: var_g value below which IMU is least trusted.
    """
    
    def __init__(self,
                 w_min: float = 0.1,
                 w_max: float = 10.0,
                 history_size: int = 30,
                 loss_sensitivity: float = 0.5,
                 excitation_threshold: float = 0.25):
        self.w_min = w_min
        self.w_max = w_max
        self.loss_history = deque(maxlen=history_size)
        self.loss_sensitivity = loss_sensitivity
        self.excitation_threshold = excitation_threshold
        
        # Telemetry for analysis and ablation studies
        self.weight_history = []
        self.call_count = 0
    
    def compute_weight(self, render_loss: float, var_g: float) -> float:
        """
        Compute the scalar weight to apply to the IMU factor's information matrix.
        
        Args:
            render_loss: Photometric loss between current frame and rendered map.
            var_g: IMU excitation signal (sqrt of preintegrated velocity variance).
        
        Returns:
            Scalar weight in [w_min, w_max] to multiply the IMU information matrix.
            Weight = 1.0 means use VINGS-Mono's default IMU strength.
            Weight > 1.0 means up-weight IMU (trust it more).
            Weight < 1.0 means down-weight IMU (trust visual more).
        """
        self.call_count += 1
        self.loss_history.append(render_loss)
        
        # Normalize render loss against its running mean so that weight
        # is scene-independent. A scene with naturally high baseline loss
        # (textureless walls) doesn't always look "uncertain" — we only 
        # care about *relative* loss spikes.
        if len(self.loss_history) < 3:
            # Not enough history yet; default to unit weight
            weight = 1.0
        else:
            mean_loss = np.mean(self.loss_history)
            normalized_loss = render_loss / (mean_loss + 1e-6)
            
            # Visual uncertainty signal: tanh squashes to [-1, 1]
            # >1.0 means loss is above mean → visual uncertain → trust IMU more
            visual_uncertainty = np.tanh(
                self.loss_sensitivity * (normalized_loss - 1.0)
            )
            
            # Excitation signal: how much IMU data is even usable
            # <1.0 at low excitation (walking), ~1.0 at high (flight)
            excitation_factor = np.clip(
                var_g / self.excitation_threshold, 0.0, 1.0
            )
            
            # Combined weight: scale IMU strength by both signals
            # Up-weight cap is scaled by excitation (no point trusting
            # a weak IMU signal even if visual is uncertain).
            weight = 1.0 + visual_uncertainty * excitation_factor * (self.w_max - 1.0)
            weight = float(np.clip(weight, self.w_min, self.w_max))
        
        self.weight_history.append({
            'call': self.call_count,
            'render_loss': render_loss,
            'var_g': var_g,
            'weight': weight,
        })
        return weight
    
    def get_telemetry(self) -> dict:
        """Return logged weight history for ablation analysis."""
        return {
            'history': list(self.weight_history),
            'total_calls': self.call_count,
            'mean_weight': np.mean([h['weight'] for h in self.weight_history]) 
                           if self.weight_history else 1.0,
        }
    
    def reset(self):
        """Clear state (e.g., between sequences)."""
        self.loss_history.clear()
        self.weight_history.clear()
        self.call_count = 0