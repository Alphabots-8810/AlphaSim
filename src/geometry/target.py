"""Hex target geometry for hub top opening."""
import math
from dataclasses import dataclass


@dataclass
class HexTarget:
    """Regular hexagonal opening in horizontal plane.

    across_flats_m: distance between two parallel sides (inscribed circle diameter).
    height_m: vertical height of the opening from carpet.
    """
    across_flats_m: float
    height_m: float

    @property
    def apothem(self) -> float:
        # Inradius (perpendicular distance from center to a flat side).
        return self.across_flats_m / 2.0

    @property
    def side_length(self) -> float:
        return self.across_flats_m / math.sqrt(3)

    @property
    def circumradius(self) -> float:
        # Distance from center to vertex.
        return self.side_length

    def contains_point_2d(self, x_offset: float) -> bool:
        """Conservative 2D check (vertical plane through shooter & hub center).
        Hex appears as a 1D slot of width 2·apothem (worst-case narrowest direction)."""
        return abs(x_offset) <= self.apothem

    def contains_ball_2d(self, x_offset: float, ball_radius: float) -> bool:
        """Ball center must be inside (apothem - ball_radius) for the whole ball to fit."""
        effective = self.apothem - ball_radius
        return effective > 0 and abs(x_offset) <= effective

    def contains_point_3d_flat_top(self, x: float, z: float) -> bool:
        """Full 3D point-in-hex check, flat-top orientation (flat sides ±x direction).
        Hex centered at origin. Useful when extending to 3D aiming."""
        a = self.apothem
        s = self.side_length
        if abs(x) > a:
            return False
        # Slanted edges: |z| ≤ s·(1 - |x|/(2a))  — equivalent to s - |x|/sqrt(3)
        max_z = s - abs(x) / math.sqrt(3)
        return abs(z) <= max_z
