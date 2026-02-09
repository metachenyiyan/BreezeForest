#!/usr/bin/env python
"""
Simple test script to verify the exact Jacobian computation using torch.func.jacrev
"""

import torch
from model.BreezeForest import BreezeForest

# Set random seed for reproducibility
torch.manual_seed(42)

print("=" * 60)
print("Testing Exact Jacobian Computation with torch.func.jacrev")
print("=" * 60)

# Create a simple BreezeForest model
bf = BreezeForest(
    dim=3,
    shapes=[
        [1, 4, 1],
        [1, 4, 1],
        [1, 4, 1]
    ],
    sap_w=0.5,
    inc_mode="no strict"
)

# Create test data (small batch for quick testing)
batch_size = 5
x = torch.randn(batch_size, 3)

print(f"\nInput shape: {x.shape}")
print(f"Input sample:\n{x[0]}")

# Test original train_forward (finite difference approximation)
print("\n" + "-" * 60)
print("1. Testing original train_forward (finite difference)")
print("-" * 60)

try:
    y_approx, log_det_approx = bf.train_forward(x, light=False)
    print(f"✓ Original method succeeded")
    print(f"  Output shape: {y_approx.shape}")
    print(f"  Log|det(J)|: {log_det_approx.item():.6f}")
except Exception as e:
    print(f"✗ Original method failed: {e}")
    log_det_approx = None

# Test new train_forward_exact (using torch.func.jacrev)
print("\n" + "-" * 60)
print("2. Testing train_forward_exact (torch.func.jacrev)")
print("-" * 60)

try:
    y_exact, log_det_exact = bf.train_forward_exact(x)
    print(f"✓ Exact method succeeded")
    print(f"  Output shape: {y_exact.shape}")
    print(f"  Log|det(J)|: {log_det_exact.item():.6f}")

    # Check if gradients can be computed
    print("\n  Testing backward pass...")
    loss = -log_det_exact
    loss.backward()
    print(f"  ✓ Backward pass succeeded")

    # Check some gradient norms
    grad_norms = []
    for name, param in bf.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))

    if grad_norms:
        print(f"  ✓ Gradients computed for {len(grad_norms)} parameters")
        print(f"  Sample gradient norms:")
        for name, norm in grad_norms[:3]:
            print(f"    {name}: {norm:.6f}")

except Exception as e:
    print(f"✗ Exact method failed: {e}")
    import traceback
    traceback.print_exc()
    log_det_exact = None

# Compare results
if log_det_approx is not None and log_det_exact is not None:
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"Finite difference log|det|: {log_det_approx.item():.6f}")
    print(f"Exact (jacrev) log|det|:    {log_det_exact.item():.6f}")
    diff = abs(log_det_approx.item() - log_det_exact.item())
    print(f"Absolute difference:        {diff:.6f}")

    if diff < 0.1:
        print(f"✓ Results are close (difference < 0.1)")
    else:
        print(f"⚠ Results differ by {diff:.6f}")

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)
