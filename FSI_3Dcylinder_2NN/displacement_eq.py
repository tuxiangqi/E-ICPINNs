"""Equations related to displacement
"""

from sympy import Symbol, Function, Number, log, Abs, simplify, sqrt

from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.node import Node
import numpy as np


from sympy import Symbol, Function, Number, sqrt, simplify
from physicsnemo.sym.eq.pde import PDE


class Epsilon_nondim(PDE):
    """
    Non-dimensionalized ring model (2.7):
        η_tt* + b* η* = H*

    where:
      - coordinates, time, velocity, pressure, etc. have been non-dimensionalized by
        L, L/U, U, ρ_f U^2, respectively
      - in this class, eta denotes the non-dimensional displacement η* = η_phys / L
      - b* = b T^2,    b = E_w / (ρ_w (1-ν_w^2) R0^2)
      - H* = (L/U^2) H,  H = (1/(ρ_w h0)) [ R0 P_phys − g μ (∇u_phys + (∇u_phys)^T)n·e_r ]
        and P_phys = ρ_f U^2 p*,  (∇u_phys + ... ) = (U/L)(∇u* + ...)

      Finally:
        H* = C1 p* − C2 g S_nn*
        C1 = (L ρ_f R0) / (ρ_w h0)
        C2 = μ / (ρ_w h0 U)

      Here we adopt axisymmetry and radial-dominant deformation, taking
        R/R0 = 1 + (L η*) / R0
        g ≈ R/R0                      (if you want a more refined form, you may change it to
                                       R/R0*sqrt(1 + (∂η/∂z)^2))
    """

    def __init__(self,
                 rho_f=1000.0,   # fluid density, used to recover physical pressure from non-dimensional p
                 V=0.08          # velocity scale U, consistent with NavierStokes
                 ):

        # ---- physical scale parameters (use Number for easier sympy handling) ----
        rho_f = Number(rho_f)
        V     = Number(V)

        rho_w = Number(1200.0)     # vessel wall density ρ_w
        nu_w  = Number(0.3)        # Poisson ratio ν_w
        E_w   = Number(3.0e5)      # Young's modulus E_w
        R0    = Number(0.005)      # reference inner radius [m]
        h0    = Number(0.001)      # wall thickness h0 [m]
        mu    = Number(0.003)      # fluid viscosity μ [Pa·s]

        L     = Number(0.01)       # length scale = diameter [m]
        T     = L / V              # time scale [s]

        # ----------------- coordinates and variables (already non-dimensional) -----------------
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        t       = Symbol("t")

        input_variables = {"x": x, "y": y, "z": z, "t": t}

        u   = Function("u")(*input_variables)      # non-dimensional velocity u*
        v   = Function("v")(*input_variables)
        w   = Function("w")(*input_variables)
        p   = Function("p")(*input_variables)      # non-dimensional pressure p*
        eta = Function("eta")(*input_variables)    # non-dimensional displacement η* = η_phys / L

        # ----------------- radial unit vector e_r -----------------
        r   = sqrt(x**2 + y**2)
        e_rx = x / r
        e_ry = y / r
        e_rz = 0.0

        # ----------------- velocity gradients (non-dimensional) -----------------
        u_x = u.diff(x); u_y = u.diff(y); u_z = u.diff(z)
        v_x = v.diff(x); v_y = v.diff(y); v_z = v.diff(z)
        w_x = w.diff(x); w_y = w.diff(y); w_z = w.diff(z)

        # S* = grad u* + (grad u*)^T
        Sxx = 2 * u_x
        Sxy = u_y + v_x
        Sxz = u_z + w_x
        Syy = 2 * v_y
        Syz = v_z + w_y
        Szz = 2 * w_z

        # (S* · e_r) · e_r = e_r^T S* e_r = S_nn*
        Ser_x    = Sxx * e_rx + Sxy * e_ry + Sxz * e_rz
        Ser_y    = Sxy * e_rx + Syy * e_ry + Syz * e_rz
        Ser_z    = Sxz * e_rx + Syz * e_ry + Szz * e_rz
        Snn_star = Ser_x * e_rx + Ser_y * e_ry + Ser_z * e_rz

        # ----------------- R/R0 and g -----------------
        # physical displacement: η_phys = L * η*
        # R(θ,z,t) = R0 + η_phys  =>  R/R0 = 1 + (L η*)/R0
        R_over_R0 = 1 + (L * eta) / R0

        # axisymmetry + radial-only deformation, neglect ∂R/∂θ and ∂R/∂z terms ⇒ g ≈ R/R0
        #g = R_over_R0
        # if you want a more refined form, you may change it to:
        eta_z = eta.diff(z)
        g = R_over_R0 * sqrt(1 + eta_z**2)
        

        # ----------------- non-dimensional coefficient b* and H* -----------------
        # b = E_w / (ρ_w (1-ν_w^2) R0^2),   b* = b T^2
        b      = E_w / (rho_w * (1.0 - nu_w * nu_w) * R0**2)
        b_star = b * T**2

        # H* = C1 p* − C2 g S_nn*
        # derived from (2.7):
        #   C1 = (L ρ_f R0) / (ρ_w h0)
        #   C2 =  μ / (ρ_w h0 V)
        C1 = L * rho_f * R0 / (rho_w * h0)
        C2 = mu / (rho_w * h0 * V)

        H_star = C1 * p - C2 * g * Snn_star

        # ----------------- equation residual: η_tt* + b* η* - H* = 0 -----------------
        self.equations = {}
        self.equations["ep_equation"] = simplify(
            eta.diff(t, 2) + b_star * eta - H_star
        )

class FSIKinematicBC(PDE):
    def __init__(self):
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        t = Symbol("t")

        input_variables = {"x": x, "y": y, "z": z, "t": t}

        u   = Function("u")(*input_variables)
        v   = Function("v")(*input_variables)
        w   = Function("w")(*input_variables)
        eta = Function("eta")(*input_variables)

        r = sqrt(x**2 + y**2)
        e_rx = x / r
        e_ry = y / r

        eta_t = eta.diff(t)

        self.equations = {}
        # u = eta_t * x/r
        self.equations["fsi_u"] = simplify(u - eta_t * e_rx)
        # v = eta_t * y/r
        self.equations["fsi_v"] = simplify(v - eta_t * e_ry)
        # w = 0
        self.equations["fsi_w"] = simplify(w)

class harmonic_extension(PDE):
    def __init__(self):
        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        t = Symbol("t")

        input_variables = {"x": x, "y": y, "z": z, "t": t}

        # scalar displacement amplitude eta(x,y,z,t)
        eta = Function("eta")(*input_variables)

        # radial coordinate r and unit vector e_r
        r = sqrt(x**2 + y**2)
        e_rx = x / r
        e_ry = y / r

        # vector displacement d = eta * e_r
        d_x = eta * e_rx
        d_y = eta * e_ry
        # d_z = 0 (omitted)

        # Laplacian operator
        def lap(f):
            return f.diff(x, 2) + f.diff(y, 2) + f.diff(z, 2)

        self.equations = {}
        # Δ d_x = 0, Δ d_y = 0
        self.equations["harmonic_x"] = simplify(lap(d_x))
        self.equations["harmonic_y"] = simplify(lap(d_y))
