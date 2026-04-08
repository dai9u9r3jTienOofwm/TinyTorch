import sys
import os

# Thêm thư mục src vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tinytorch.tensor import Tensor
import numpy as np

"""🧪 Module Test: Complete Integration

Comprehensive test of entire module functionality.

This final test runs before module summary to ensure:
- All unit tests pass
- Functions work together correctly
- Module is ready for integration with TinyTorch
"""
print("🧪 RUNNING MODULE INTEGRATION TEST")
print("=" * 50)


# Create input data (2 samples, 3 features)
x = Tensor([[1, 2, 3], [4, 5, 6]])

# First stage: 3 inputs → 4 intermediate values
W1 = Tensor([[0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2]])
b1 = Tensor([0.1, 0.2, 0.3, 0.4])

# Forward pass: hidden = xW1 + b1
hidden = x.matmul(W1) + b1
assert hidden.shape == (2, 4), f"Expected (2, 4), got {hidden.shape}"

# Second stage: 4 intermediate → 2 outputs
W2 = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
b2 = Tensor([0.1, 0.2])

# Output stage: output = hiddenW2 + b2
output = hidden.matmul(W2) + b2
assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"

# Verify data flows correctly (no NaN, reasonable values)
assert not np.isnan(output.data).any(), "Output contains NaN values"
assert np.isfinite(output.data).all(), "Output contains infinite values"

print("✅ Two-stage linear transformation works!")

# Test complex shape manipulations
print("🧪 Integration Test: Complex Shape Operations...")
data = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Reshape to 3D tensor (simulating batch processing)
tensor_3d = data.reshape(2, 2, 3)  # (batch=2, height=2, width=3)
assert tensor_3d.shape == (2, 2, 3)

# Spatial averaging (collapse height and width)
pooled = tensor_3d.mean(axis=(1, 2))  # Average across spatial dimensions
assert pooled.shape == (2,), f"Expected (2,), got {pooled.shape}"

# Flatten to 2D
flattened = tensor_3d.reshape(2, -1)  # (batch, features)
assert flattened.shape == (2, 6)

# Transpose for different operations
transposed = tensor_3d.transpose()  # Should transpose last two dims
assert transposed.shape == (2, 3, 2)

print("✅ Complex shape operations work!")

# Test broadcasting edge cases
print("🧪 Integration Test: Broadcasting Edge Cases...")

# Scalar broadcasting
scalar = Tensor(5.0)
vector = Tensor([1, 2, 3])
result = scalar + vector  # Should broadcast scalar to vector shape
expected = np.array([6, 7, 8], dtype=np.float32)
assert np.array_equal(result.data, expected)

# Matrix + vector broadcasting
matrix = Tensor([[1, 2], [3, 4]])
vec = Tensor([10, 20])
result = matrix + vec
expected = np.array([[11, 22], [13, 24]], dtype=np.float32)
assert np.array_equal(result.data, expected)

print("✅ Broadcasting edge cases work!")

print("\n" + "=" * 50)
print("🎉 ALL TESTS PASSED! Module ready for export.")
print("Run: tito module complete 01")
