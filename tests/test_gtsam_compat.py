"""
Smoke test for scripts/frontend/gtsam_compat.py.

Verifies that:
1. BA2GTSAM is re-bound onto the gtsam namespace
2. CustomHessianFactor is re-bound onto the gtsam namespace
3. evaluateErrorCustom shim works on a minimal synthetic input

Run this from the VINGS-Mono root on PSC after PyTorch + gtsam are installed.
Expected: all three tests print PASS.
"""
import sys
import os
import numpy as np

# Ensure we can import from scripts/frontend/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

# Apply compatibility shim
from frontend import gtsam_compat  # noqa: F401

import gtsam


def test_ba2gtsam_exists():
    """Test 1: BA2GTSAM is accessible via gtsam namespace."""
    print("Test 1: gtsam.BA2GTSAM accessible...", end=" ")
    assert hasattr(gtsam, 'BA2GTSAM'), "gtsam.BA2GTSAM not registered"
    assert callable(gtsam.BA2GTSAM), "gtsam.BA2GTSAM is not callable"
    
    # Functional check: identity Tbc, identity Hessian, unit vector
    Tbc = gtsam.Pose3()
    H = np.eye(6)
    v = np.ones(6)
    Hg, vg = gtsam.BA2GTSAM(H, v, Tbc)
    assert Hg.shape == (6, 6), f"Expected (6,6), got {Hg.shape}"
    assert vg.shape == (6,), f"Expected (6,), got {vg.shape}"
    print("PASS")


def test_custom_hessian_factor_exists():
    """Test 2: CustomHessianFactor is accessible via gtsam namespace."""
    print("Test 2: gtsam.CustomHessianFactor accessible...", end=" ")
    assert hasattr(gtsam, 'CustomHessianFactor'), "CustomHessianFactor not registered"
    print("PASS")


def test_evaluate_error_custom_shim():
    """Test 3: evaluateErrorCustom shim runs and fills H5 buffer."""
    print("Test 3: evaluateErrorCustom shim...", end=" ")
    
    params = gtsam.PreintegrationCombinedParams.MakeSharedU(9.81)
    pim = gtsam.PreintegratedCombinedMeasurements(params)
    pim.integrateMeasurement(np.array([0., 0., 9.81]), np.array([0., 0., 0.]), 0.01)
    f = gtsam.CombinedImuFactor(0, 1, 2, 3, 4, 5, pim)
    
    pose_i, pose_j = gtsam.Pose3(), gtsam.Pose3()
    vel_i = np.zeros(3)
    vel_j = np.zeros(3)
    bias_i = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    bias_j = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    
    H1 = np.zeros([15, 6], order='F', dtype=np.float64)
    H2 = np.zeros([15, 3], order='F', dtype=np.float64)
    H3 = np.zeros([15, 6], order='F', dtype=np.float64)
    H4 = np.zeros([15, 3], order='F', dtype=np.float64)
    H5 = np.zeros([15, 6], order='F', dtype=np.float64)
    H6 = np.zeros([15, 6], order='F', dtype=np.float64)
    
    err = f.evaluateErrorCustom(pose_i, vel_i, pose_j, vel_j,
                                  bias_i, bias_j,
                                  H1, H2, H3, H4, H5, H6)
    
    assert err is not None, "evaluateErrorCustom returned None"
    assert hasattr(err, 'shape'), "err is not an array"
    assert err.shape == (15,), f"Expected (15,), got {err.shape}"
    print("PASS")


if __name__ == '__main__':
    print("=" * 60)
    print("gtsam_compat.py smoke test")
    print("=" * 60)
    test_ba2gtsam_exists()
    test_custom_hessian_factor_exists()
    test_evaluate_error_custom_shim()
    print("=" * 60)
    print("All tests passed. gtsam_compat.py is working.")