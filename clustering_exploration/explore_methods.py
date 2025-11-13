"""
Explore CRLI and VaDER clustering methods from PyPOTS
"""

import pypots
print(f"PyPOTS version: {pypots.__version__}")

# Check available clustering methods
from pypots import clustering

# List all clustering methods available
print("\nAvailable clustering modules:")
print(dir(clustering))

# Try to import CRLI and VaDER
try:
    from pypots.clustering import CRLI
    print("\n✓ CRLI imported successfully")
    print(f"CRLI class: {CRLI}")
    print(f"CRLI docstring:\n{CRLI.__doc__}")
    print(f"\nCRLI __init__ signature:")
    import inspect
    print(inspect.signature(CRLI.__init__))
except ImportError as e:
    print(f"\n✗ Failed to import CRLI: {e}")

try:
    from pypots.clustering import VaDER
    print("\n✓ VaDER imported successfully")
    print(f"VaDER class: {VaDER}")
    print(f"VaDER docstring:\n{VaDER.__doc__}")
    print(f"\nVaDER __init__ signature:")
    import inspect
    print(inspect.signature(VaDER.__init__))
except ImportError as e:
    print(f"\n✗ Failed to import VaDER: {e}")
