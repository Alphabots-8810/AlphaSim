"""Dual-roller shooter kinematics.

Convention: positive direction = ball motion direction.
- v_top = tangential speed of TOP roller surface at contact (typically the flywheel)
- v_bottom = tangential speed of BOTTOM roller surface at contact (the hood roller)
- backspin (positive ω) when v_bottom > v_top
- pure translation (zero spin) when v_top == v_bottom

For ball CG velocity V and angular velocity ω (positive = backspin):
  v_top    = V - ω·r_ball     (kinematic constraint at top contact)
  v_bottom = V + ω·r_ball     (kinematic constraint at bottom contact)
Inverting:
  V = (v_top + v_bottom) / 2
  ω = (v_bottom - v_top) / (2·r_ball)
"""
import math
from dataclasses import dataclass


@dataclass
class Roller:
    diameter_m: float
    gear_ratio: float = 1.0
    motor_free_rpm: float = 6000.0
    slip_factor: float = 0.18

    @property
    def radius_m(self) -> float:
        return self.diameter_m / 2

    @property
    def max_flywheel_rpm(self) -> float:
        return self.motor_free_rpm / self.gear_ratio

    def surface_speed_ms(self, flywheel_rpm: float) -> float:
        omega = flywheel_rpm * 2 * math.pi / 60
        return omega * self.radius_m * (1.0 - self.slip_factor)

    def rpm_for_surface_speed(self, v_ms: float) -> float:
        if v_ms < 0:
            return 0.0
        omega = v_ms / (self.radius_m * (1.0 - self.slip_factor))
        return omega * 60 / (2 * math.pi)


@dataclass
class DualRoller:
    top: Roller
    bottom: Roller
    ball_radius_m: float

    def exit_velocity_ms(self, rpm_top: float, rpm_bottom: float) -> float:
        v_top = self.top.surface_speed_ms(rpm_top)
        v_bottom = self.bottom.surface_speed_ms(rpm_bottom)
        return 0.5 * (v_top + v_bottom)

    def spin_rate_rad_s(self, rpm_top: float, rpm_bottom: float) -> float:
        v_top = self.top.surface_speed_ms(rpm_top)
        v_bottom = self.bottom.surface_speed_ms(rpm_bottom)
        return (v_bottom - v_top) / (2 * self.ball_radius_m)

    def rpms_for_exit_velocity(
        self, v_exit: float, omega_spin: float = 0.0,
    ) -> tuple[float, float]:
        """Inverse: given (v_exit, spin), return (rpm_top, rpm_bottom).
        Stage 1 uses omega_spin=0 → both rollers at same surface speed."""
        v_top = v_exit - omega_spin * self.ball_radius_m
        v_bottom = v_exit + omega_spin * self.ball_radius_m
        return (
            self.top.rpm_for_surface_speed(v_top),
            self.bottom.rpm_for_surface_speed(v_bottom),
        )
