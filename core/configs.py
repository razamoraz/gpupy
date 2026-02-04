from dataclasses import dataclass


@dataclass(slots=True, kw_only=True)
class LBMConfig:
    nx: int = 512
    ny: int = 512
    iterations: int = 1000
    w_0: float = 4.0 / 9.0
    w_1: float = 1.0 / 9.0
    w_2: float = 1.0 / 36.0
    viscosity: float = 0.02
    omega: float = 0.0
    precision: str = "f64"

    def __post_init__(self):
        self.omega = 1.0 / (3.0 * self.viscosity + 0.5)


@dataclass(slots=True, kw_only=True)
class FDMConfig:
    nx: int = 512
    ny: int = 512
    iterations: int = 1000
    alpha: float = 0.01  # Thermal diffusivity
    dt: float = 0.0001
    dx: float = 0.01
    precision: str = "f64"
