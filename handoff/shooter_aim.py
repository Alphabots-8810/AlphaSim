"""FRC 2026 REBUILT — closed-form shooter aim lookup  (Alpha Sim, V3.1 fit).

distance d (m, ball exit point -> HUB center)  ->  (hood angle deg, flywheel surface speed m/s)

Field calibration is ONE number: K_SPEED. Nothing else needs tuning at an event.

How the coefficients were derived: see research_report.html §2-5. They were generated
by scripts/generate_coefficients.py at SHOOTER_HEIGHT_M (below); the height is baked into
the polynomials, so editing that constant alone does nothing at runtime. To retarget a
different exit height, rerun the generator and paste the new a/b/c.
"""

SHOOTER_HEIGHT_M = 0.50   # exit height the coeffs were generated at (provenance, NOT a runtime input)
D_MIN, D_MAX = 1.5, 5.0   # valid fit range (m); inputs outside this are clamped

K_SPEED = 1.00            # <- THE field-calibration knob. short -> raise, long -> lower.


def _clamp(d):
    return max(D_MIN, min(D_MAX, d))


def hood_angle_deg(d):
    """Hood angle (deg; 90 deg = straight down) for distance d (m)."""
    d = _clamp(d)
    return 0.887787 * d * d - 10.549949 * d + 86.441900


def flywheel_speed_mps(d):
    """Flywheel surface (rim) linear speed (m/s). RPM = v / (pi * wheel_diameter_m) * 60."""
    d = _clamp(d)
    return K_SPEED * (0.016170 * d * d + 0.717698 * d + 6.404844)


def aim(d):
    """distance d (m) -> (hood_angle_deg, flywheel_speed_mps)."""
    return hood_angle_deg(d), flywheel_speed_mps(d)


if __name__ == "__main__":
    for _d in (1.5, 2.0, 3.0, 4.0, 5.0):
        _h, _v = aim(_d)
        print(f"d={_d:.1f} m  ->  hood {_h:6.2f} deg   flywheel {_v:6.2f} m/s")
