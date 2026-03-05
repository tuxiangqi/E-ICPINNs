import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sympy import Symbol, Eq, sin, cos, Min, Max, Abs, log, exp, sqrt
import matplotlib.pyplot as plt

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, to_yaml, instantiate_arch
from physicsnemo.sym.hydra.config import PhysicsNeMoConfig
from physicsnemo.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
    PointwiseConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.key import Key
from physicsnemo.sym.node import Node
from physicsnemo.sym import quantity
from physicsnemo.sym.eq.non_dim import NonDimensionalizer, Scaler
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.models.activation import Activation
from physicsnemo.sym.geometry.tessellation import Tessellation

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes

import stl
from stl import mesh
import vtk
import time
from displacement_eq_d import Epsilon_nondim, harmonic_extension, FSIKinematicBC
from copy import deepcopy

@physicsnemo.sym.main(config_path="conf", config_name="conf_pretrain")
def run(cfg: PhysicsNeMoConfig) -> None:
    print(to_yaml(cfg))

    # values obtained from paraview
    center_hardcode = (0, 0, -2.5)  # chosen for the non-dimensionalized STL; restore z in closed_nondim.stl to [0, 5]
    inlet_center_abs_hardcode = (0, 0, 0)
    inlet_normal_hardcode = (0, 0, 1)

    # path definitions
    point_path = to_absolute_path("FSI_3Dcylinder_2NN/stl_files")
    path_interior = point_path + "/closed_nondim.stl"  # already non-dimensionalized and shifted to the vessel center
    path_wall = point_path + "/wall_nondim.stl"        # already non-dimensionalized and shifted to the vessel center
    path_inlet = point_path + "/inlet_nondim.stl"      # zL=-2.5
    path_outlet = point_path + "/outlet_nondim.stl"    # zL=2.5

    # read stl files to make geometry
    interior_mesh = Tessellation.from_stl(path_interior, airtight=True)
    wall_mesh = Tessellation.from_stl(path_wall, airtight=False)
    inlet_mesh = Tessellation.from_stl(path_inlet, airtight=False)
    outlet_mesh = Tessellation.from_stl(path_outlet, airtight=False)


    # physical quantities################################################
    mu = quantity(0.003, "kg/(m*s)")  # Fluid viscosity
    rho = quantity(1000, "kg/m^3")    # Fluid Density
    nu = mu / rho
    L = quantity(0.05, "m")          # vessel length
    v_mag = quantity(0.2, "m/s")     # maximum velocity
    noslip_u = quantity(0.0, "m/s")
    noslip_v = quantity(0.0, "m/s")
    outlet_p = quantity(1333.2, "kg/(m*s^2)")
    time_out = quantity(0.01, "s")

    velocity_scale = quantity(0.08, "m/s")  # prior velocity: 0.08 m/s
    density_scale = quantity(1000, "kg/m^3")
    length_scale = quantity(0.01, "m")      # diameter

    nd = NonDimensionalizer(
        length_scale=length_scale,
        time_scale=length_scale / velocity_scale,
        mass_scale=density_scale * (length_scale ** 3),
    )
    # convenient constants for later use ---------------------------------
    V_scale = 0.08
    L_scale = 0.01
    t_scale = L_scale / V_scale
    rho_f_val = 1000
    # --------------------------------------------------------------------
    # normalize meshes
    def normalize_mesh(mesh, center, scale):
        mesh = mesh.translate([-c for c in center])
        mesh = mesh.scale(scale)
        return mesh

    # first do non-dimensionalization
    def nondimension_invar(invar, length_scale, time_scale):
        invar["x"] = invar["x"] / length_scale
        invar["y"] = invar["y"] / length_scale
        invar["z"] = invar["z"] / length_scale
        invar["t"] = invar["t"] / time_scale
        return invar

    def nondimension_outvar(outvar, velocity_scale, rho_f):
        outvar["u"] = outvar["u"] / velocity_scale
        outvar["v"] = outvar["v"] / velocity_scale
        outvar["w"] = outvar["w"] / velocity_scale
        outvar["p"] = outvar["p"] / (rho_f * velocity_scale * velocity_scale)
        return outvar

    def nondimension_outvar_eta(outvar, length_scale):
        outvar["eta"] = outvar["eta"] / length_scale
        return outvar
    
    def nondimension_outvar_d(outvar, length_scale):
        outvar["dx"] = outvar["dx"] / length_scale
        outvar["dy"] = outvar["dy"] / length_scale
        outvar["dz"] = outvar["dz"] / length_scale
        return outvar
    
    # normalize invars, then shift to the center point
    def normalize_invar(invar, center, scale, dims=2):
        invar["x"] -= center[0]
        invar["y"] -= center[1]
        invar["z"] -= center[2]
        invar["x"] *= scale
        invar["y"] *= scale
        invar["z"] *= scale
        if "area" in invar.keys():
            invar["area"] *= scale ** dims
        return invar

    # geometry scaling
    scale = 1

    # center of overall geometry
    center = center_hardcode
    print('Overall geometry center: ', center)

    # inlet center & normal
    inlet_center_abs = inlet_center_abs_hardcode
    print("inlet_center_abs:", inlet_center_abs)

    inlet_center = list((np.array(inlet_center_abs) - np.array(center)) * scale)
    print("inlet_center:", inlet_center)

    inlet_normal = inlet_normal_hardcode
    print("inlet_normal:", inlet_normal)


    # scale and center the geometry files
    interior_mesh = normalize_mesh(interior_mesh, center, scale)
    wall_mesh = normalize_mesh(wall_mesh, center, scale)
    inlet_mesh = normalize_mesh(inlet_mesh, center, scale)
    outlet_mesh = normalize_mesh(outlet_mesh, center, scale)


    # sympy time range
    t = Symbol("t")
    time_length = nd.ndim(time_out)
    time_range = {t: (0, time_length)}
    time0_range = {t: (0, 0)}

    # ===== Create two Domains: one for displacement, one for flow =====
    domain_disp = Domain()
    domain_flow = Domain()

    # ===== PDE / networks / nodes definition =====
    ns = NavierStokes(nu=nd.ndim(nu), rho=nd.ndim(rho), dim=3, time=True)
    ns.pprint()

    eq = Epsilon_nondim(rho_f=1000, V=0.08)
    eq.pprint()
    he = harmonic_extension()
    he.pprint()

    fsi_kin = FSIKinematicBC()
    fsi_kin.pprint()

    normal_dot_vel = NormalDotVec(["u", "v", "w"])

    displacement_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
        output_keys=[Key("dx"), Key("dy"), Key("dz")],
        cfg=cfg.arch.fully_connected,
        layer_size=20,
        nr_layers=10,
    )

    flow_net = instantiate_arch(
        input_keys=[Key("x_f"), Key("y_f"), Key("z_f"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        cfg=cfg.arch.fully_connected,
        layer_size=256, #128
        nr_layers=10,
    )

    # ====== zero-initialize and freeze displacement_net; train flow_net only ======
    # set all Linear layer weights and biases to 0 so that eta(x,y,z,t) stays identically 0
    for m in displacement_net.modules():
        if isinstance(m, torch.nn.Linear):      # or isinstance(m, nn.Linear)
            m.weight.data.zero_()
            if m.bias is not None:
                m.bias.data.zero_()


    # initial training parameters
    model_path = to_absolute_path("outputs/1Initialization_training/flow_network.0.pth")
    flow_net.load_state_dict(torch.load(model_path))


    nodes = (
        ns.make_nodes()
        + eq.make_nodes(detach_names=["u", "v", "w", "p"])
        + he.make_nodes()
        + fsi_kin.make_nodes()
        + normal_dot_vel.make_nodes()
        + [
            Node.from_sympy(
                Symbol("x")
                + Symbol("dx"),
                "x_f",
            )
        ]
        + [
            Node.from_sympy(
                Symbol("y")
                + Symbol("dy"),
                "y_f",
            )
        ]
        + [
            Node.from_sympy(
                Symbol("z")
                + Symbol("dz"),
                "z_f",
            )
        ]
        + [Node.from_sympy(Symbol("dx") * L_scale, "dx_scaled")]
        + [Node.from_sympy(Symbol("dy") * L_scale, "dy_scaled")]
        + [Node.from_sympy(Symbol("dz") * L_scale, "dz_scaled")]
        + [
            flow_net.make_node(name="flow_network", jit=cfg.jit),
            displacement_net.make_node(name="displacement_network", jit=cfg.jit),
        ]
        + Scaler(
            ["u", "v", "w", "p"],
            ["u_scaled", "v_scaled", "w_scaled", "p_scaled"],
            ["m/s", "m/s", "m/s", "kg/(m*s^2)"],
            nd,
        ).make_node()
    )

    

    # =========================================
    # 1) Structure / displacement-related constraints
    # =========================================
    # supervised wall displacement
    wall_var = csv_to_dict(to_absolute_path("FSI_3Dcylinder_2NN/SimVascular/eta_wall_101state_1.csv"))
    wall_invar = {key: value for key, value in wall_var.items() if key in ["x", "y", "z", "t"]}
    wall_invar = nondimension_invar(wall_invar, L_scale, t_scale)
    #wall_invar = normalize_invar(wall_invar, center, scale, dims=3)

    wall_outvar = {key: value for key, value in wall_var.items() if key in ["dx","dy","dz"]}
    wall_outvar = nondimension_outvar_d(wall_outvar, length_scale)

    eta_wall = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=wall_invar,
        outvar=wall_outvar,
        batch_size=cfg.batch_size.wall,
    )
    domain_disp.add_constraint(eta_wall, "eta_wall_super")
    print("accomplish he_wallsuper_constraint :", cfg.batch_size.wall)

    # elastic equation ep_equation (on the wall)
    eta_ep = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall_mesh,
        outvar={"ep_equation": 0},
        batch_size=cfg.batch_size.wall,
        parameterization=time_range,
    )
    domain_disp.add_constraint(eta_ep, "eta_ep_wall")
    print("accomplish ep_wall_constraint :", cfg.batch_size.wall)

    fsi_bc = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall_mesh,
        outvar={"fsi_u": 0, "fsi_v": 0, "fsi_w": 0},
        batch_size=cfg.batch_size.wall,
        parameterization=time_range,
    )
    domain_disp.add_constraint(fsi_bc, "fsi_bc_wall")
    print("accomplish fsi_bc_wall_constraint :", cfg.batch_size.wall)

    # inlet / outlet / initial displacement boundary conditions
    eta_inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_mesh,
        outvar={"dx": 0,"dy": 0,"dz": 0},
        batch_size=cfg.batch_size.inlet,
        parameterization=time_range,
    )
    domain_disp.add_constraint(eta_inlet, "eta_inlet")
    print("accomplish he_inlet_constraint :", cfg.batch_size.inlet)

    eta_outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_mesh,
        outvar={"dx": 0,"dy": 0,"dz": 0},
        batch_size=cfg.batch_size.outlet,
        parameterization=time_range,
    )
    domain_disp.add_constraint(eta_outlet, "eta_outlet")
    print("accomplish he_outlet_constraint :", cfg.batch_size.outlet)

    eta_initial = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall_mesh,
        outvar={"dx": 0,"dy": 0,"dz": 0},
        batch_size=cfg.batch_size.wall,
        parameterization=time0_range,
    )
    domain_disp.add_constraint(eta_initial, "eta_initial")
    print("accomplish he_initial_constraint :", cfg.batch_size.wall)

    # harmonic extension Δd = 0 in interior
    eta_he = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"harmonic_x": 0, "harmonic_y": 0, "harmonic_z": 0},
        batch_size=cfg.batch_size.interior,
        lambda_weighting={"harmonic_x": 2.0, "harmonic_y": 2.0,"harmonic_z": 2.0},
        parameterization=time_range,
    )
    domain_disp.add_constraint(eta_he, "eta_he_interior")
    print("accomplish eta_he_interior_constraint :", cfg.batch_size.interior)

    # =========================================
    # 2) Flow-related constraints (inlet/outlet/initial/wall/interior)
    # =========================================
    # inlet from numpy
    inflow_var = csv_to_dict(to_absolute_path("FSI_3Dcylinder_2NN/surface_vtudata/data_in_xyz0.csv"))
    inflow_invar = {key: value for key, value in inflow_var.items() if key in ["x", "y", "z", "t"]}
    inflow_invar = nondimension_invar(inflow_invar, L_scale, t_scale)
    #inflow_invar = normalize_invar(inflow_invar, center, scale, dims=3)

    inflow_outvar = {key: value for key, value in inflow_var.items() if key in ["u", "v", "w", "p"]}
    inflow_outvar = nondimension_outvar(inflow_outvar, V_scale, rho_f_val)

    inlet_numpy = PointwiseConstraint.from_numpy(
        nodes,
        inflow_invar,
        inflow_outvar,
        batch_size=cfg.batch_size.inlet,
    )
    domain_flow.add_constraint(inlet_numpy, "inlet_numpy")
    print("accomplish inlet_constraint :", cfg.batch_size.inlet)

    # outlet
    outflow_var = csv_to_dict(to_absolute_path("FSI_3Dcylinder_2NN/surface_vtudata/data_out_xyz0.csv"))
    outflow_invar = {key: value for key, value in outflow_var.items() if key in ["x", "y", "z", "t"]}
    outflow_invar = nondimension_invar(outflow_invar, L_scale, t_scale)
    #outflow_invar = normalize_invar(outflow_invar, center, scale, dims=3)

    outflow_outvar = {key: value for key, value in outflow_var.items() if key in ["u", "v", "w", "p"]}
    outflow_outvar = nondimension_outvar(outflow_outvar, V_scale, rho_f_val)

    outlet_numpy = PointwiseConstraint.from_numpy(
        nodes,
        outflow_invar,
        outflow_outvar,
        batch_size=cfg.batch_size.outlet,
    )
    domain_flow.add_constraint(outlet_numpy, "outlet_numpy")
    print("accomplish outlet_constraint :", cfg.batch_size.outlet)

    # initial
    ms = 0
    initial_var = csv_to_dict(to_absolute_path("FSI_3Dcylinder_2NN/SimVascular/data_initial.csv"))
    initial_invar = {key: value for key, value in initial_var.items() if key in ["x", "y", "z", "t"]}
    initial_invar = nondimension_invar(initial_invar, L_scale, t_scale)
    #initial_invar = normalize_invar(initial_invar, center, scale, dims=3)
    initial_outvar = {key: value for key, value in initial_var.items() if key in ["u", "v", "w", "p"]}
    initial_outvar = nondimension_outvar(initial_outvar, V_scale, rho_f_val)

    initial = PointwiseConstraint.from_numpy(
        nodes,
        initial_invar,
        initial_outvar,
        batch_size=cfg.batch_size.initial,
    )
    domain_flow.add_constraint(initial, f"initial_{int(ms)}")

    # wall (fluid-side boundary data)
    wall_var = csv_to_dict(to_absolute_path("FSI_3Dcylinder_2NN/surface_vtudata/data_wall_xyz0.csv"))
    wall_invar = {key: value for key, value in wall_var.items() if key in ["x", "y", "z", "t"]}
    wall_invar = nondimension_invar(wall_invar, L_scale, t_scale)
    #wall_invar = normalize_invar(wall_invar, center, scale, dims=3)

    wall_outvar = {key: value for key, value in wall_var.items() if key in ["u", "v", "w", "p"]}
    wall_outvar = nondimension_outvar(wall_outvar, V_scale, rho_f_val)

    wall = PointwiseConstraint.from_numpy(
        nodes,
        wall_invar,
        wall_outvar,
        batch_size=cfg.batch_size.wall,
    )
    domain_flow.add_constraint(wall, "wall_numpy")

    # interior NS PDE
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "continuity": 2.0,
            "momentum_x": 2.0,
            "momentum_y": 2.0,
            "momentum_z": 2.0,
        },
        parameterization=time_range,
    )
    domain_flow.add_constraint(interior, "interior")
    print("accomplish interior_constraint :", 1000 * cfg.batch_size.interior)

    # =========================================
    # Validators & Inferencers (placing them in the flow domain is sufficient)
    # =========================================
    modsim_var = csv_to_dict(to_absolute_path("FSI_3Dcylinder_2NN/SimVascular/data_validator_xyz0.csv"))
    modsim_invar = {key: value for key, value in modsim_var.items() if key in ["x", "y", "z", "t"]}
    modsim_invar = nondimension_invar(modsim_invar, L_scale, t_scale)
    #modsim_invar_grid = normalize_invar(modsim_invar, center, scale, dims=3)
    modsim_outvar = {
        key: value
        for key, value in modsim_var.items()
        if key in ["u_scaled", "v_scaled", "w_scaled", "p_scaled", "eta_scaled"]
    }
    modsim_validator = PointwiseValidator(
        nodes=nodes,
        invar=modsim_invar,
        true_outvar=modsim_outvar,
        batch_size=4096,
        requires_grad=False,
    )
    domain_flow.add_validator(modsim_validator, "times_10ms_validator")
    print("accomplish times_10ms grid validator")


    #___________________________________________________________
    # update flow_net only
    for p in displacement_net.parameters():
        p.requires_grad = False
    for p in flow_net.parameters():
        p.requires_grad = True
    
    slv_flow = Solver(cfg, domain_flow)
    start_time_outer = time.time()
    slv_flow.solve()

    # total time statistics
    toc = time.time()
    elapseTime = toc - start_time_outer
    print(f"Total time taken: {round(elapseTime, 2)}s")

    hours = int(elapseTime // 3600)
    minutes = int((elapseTime % 3600) // 60)
    seconds = int(elapseTime % 60)

    print(f"Total time taken: {hours}h{minutes}m{seconds}s")


if __name__ == "__main__":
    run()