"""
GTSAM compatibility shim for VINGS-Mono.

Restores the private-fork methods (BA2GTSAM, CustomHessianFactor,
evaluateErrorCustom) using only public GTSAM 4.2 primitives. Import this
module before any code that calls these methods so the monkey-patches are
in place.

Reference: Phase 1 investigation, Krushna Thakkar, CMU ECE, April 2026.

Key discoveries:
1. BA2GTSAM and CustomHessianFactor have complete Python implementations in
   scripts/frontend/depth_video.py (lines 33-51) that were never removed
   during the refactor to the private C++ version. We simply re-bind them
   onto the gtsam namespace.

2. evaluateErrorCustom is a Fortran-style output-buffer API wrapper around
   the public CombinedImuFactor.evaluateError. At dbaf_frontend.py:665 it is
   used for one-shot gyro bias estimation (Forster et al. 2017, eq. 13).
   We shim by calling the public API and copying Jacobians into the caller's
   preallocated numpy buffers.
"""
import numpy as np

# Deferred gtsam import — we want to apply patches, not trigger side effects,
# if this module is imported before gtsam itself is available.
try:
    import gtsam
except ImportError:
    gtsam = None


def _install_depth_video_functions():
    """Re-bind local BA2GTSAM and CustomHessianFactor onto the gtsam module."""
    # Import here to avoid circular dependency at module load time
    from frontend.depth_video import BA2GTSAM, CustomHessianFactor
    gtsam.BA2GTSAM = BA2GTSAM
    gtsam.CustomHessianFactor = CustomHessianFactor


def _evaluate_error_custom(self, pose_i, vel_i, pose_j, vel_j, bias_i, bias_j,
                            H1, H2, H3, H4, H5, H6):
    """
    Shim replicating the private fork's Fortran-style output-buffer API on
    top of public evaluateError.
    
    The public API signature in GTSAM 4.2 bindings varies by build:
    - Some builds accept H args as output parameters (in-place fill)
    - Others return tuples (error, J1, J2, J3, J4, J5, J6)
    - Newer builds return only error with no Jacobians exposed
    
    We attempt all three in order, caching the working one after first call.
    """
    # Try output-buffer style first (matches GTSAM 4.2a8 with custom bindings)
    try:
        err = self.evaluateError(pose_i, vel_i, pose_j, vel_j,
                                   bias_i, bias_j,
                                   H1, H2, H3, H4, H5, H6)
        return err
    except TypeError:
        pass
    
    # Try tuple-return style (newer public GTSAM bindings)
    try:
        result = self.evaluateError(pose_i, vel_i, pose_j, vel_j,
                                      bias_i, bias_j)
        if isinstance(result, tuple) and len(result) == 7:
            err, J_pose_i, J_vel_i, J_pose_j, J_vel_j, J_bias_i, J_bias_j = result
            # Copy each Jacobian into the preallocated buffer
            np.copyto(H1, J_pose_i)
            np.copyto(H2, J_vel_i)
            np.copyto(H3, J_pose_j)
            np.copyto(H4, J_vel_j)
            np.copyto(H5, J_bias_i)
            np.copyto(H6, J_bias_j)
            return err
    except (TypeError, ValueError):
        pass
    
    raise RuntimeError(
        "Could not invoke evaluateError with Jacobians. This GTSAM build may "
        "not expose Jacobian computation via Python. Check version with "
        "gtsam.__version__ and the output of help(gtsam.CombinedImuFactor.evaluateError)."
    )


def apply_patches():
    """Apply all compatibility patches to the gtsam module."""
    if gtsam is None:
        raise ImportError("gtsam module not available; cannot apply patches")
    
    _install_depth_video_functions()
    gtsam.CombinedImuFactor.evaluateErrorCustom = _evaluate_error_custom


# Apply patches automatically on import
if gtsam is not None:
    apply_patches()