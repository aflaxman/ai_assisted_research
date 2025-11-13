"""
Test mockseries API to understand correct usage
"""

from mockseries.trend import LinearTrend
from mockseries.seasonality import SinusoidalSeasonality
from mockseries.noise import RedNoise
import inspect

print("LinearTrend signature:")
print(inspect.signature(LinearTrend.__init__))
print("\nLinearTrend docstring:")
print(LinearTrend.__doc__)

print("\n" + "="*60)
print("\nSinusoidalSeasonality signature:")
print(inspect.signature(SinusoidalSeasonality.__init__))
print("\nSinusoidalSeasonality docstring:")
print(SinusoidalSeasonality.__doc__)

print("\n" + "="*60)
print("\nRedNoise signature:")
print(inspect.signature(RedNoise.__init__))
print("\nRedNoise docstring:")
print(RedNoise.__doc__)

# Test basic usage
print("\n" + "="*60)
print("\nTesting basic usage:")

try:
    trend = LinearTrend(coefficient=0.5, value=10)
    ts = trend.generate(100)
    print(f"LinearTrend generated shape: {ts.shape}")
except Exception as e:
    print(f"LinearTrend error: {e}")

try:
    seasonality = SinusoidalSeasonality(amplitude=5, period=20)
    ts = seasonality.generate(100)
    print(f"SinusoidalSeasonality generated shape: {ts.shape}")
except Exception as e:
    print(f"SinusoidalSeasonality error: {e}")

try:
    noise = RedNoise(mean=0, std=1.0)
    ts = noise.generate(100)
    print(f"RedNoise generated shape: {ts.shape}")
except Exception as e:
    print(f"RedNoise error: {e}")
