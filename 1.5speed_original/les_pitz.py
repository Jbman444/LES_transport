#!/usr/bin/env python3
"""
LES (Smagorinsky) pimpleFoam test case with foamlib.
Same geometry as RAS case, but LES setup.
"""

import shutil
from pathlib import Path

import numpy as np
from foamlib import Dimensioned, DimensionSet, FoamCase

# ------------------------------------------------------------
# Case setup
# ------------------------------------------------------------
path = Path(__file__).parent / "lesChannelCase_1.5_speed"
shutil.rmtree(path, ignore_errors=True)
path.mkdir(parents=True)
(path / "system").mkdir()
(path / "constant").mkdir()
(path / "0").mkdir()

case = FoamCase(path)

# ------------------------------------------------------------
# system/controlDict
# ------------------------------------------------------------
with case.control_dict as f:
    f["application"] = "pimpleFoam"
    f["startFrom"] = "latestTime"
    f["startTime"] = 0
    f["stopAt"] = "endTime"
    f["endTime"] = 2.0
    f["deltaT"] = 1e-4
    f["writeControl"] = "adjustableRunTime"
    f["writeInterval"] = 0.01
    f["purgeWrite"] = 0
    f["writeFormat"] = "ascii"
    f["writePrecision"] = 6
    f["writeCompression"] = False
    f["timeFormat"] = "general"
    f["timePrecision"] = 6
    f["runTimeModifiable"] = True
    f["adjustTimeStep"] = True
    f["maxCo"] = 5.0

# ------------------------------------------------------------
# system/fvSchemes (simpler, only U and p)
# ------------------------------------------------------------
with case.fv_schemes as f:
    f["ddtSchemes"] = {"default": "Euler"}
    f["gradSchemes"] = {"default": "Gauss linear"}
    f["divSchemes"] = {
        "default": "none",
        "div(phi,U)": "Gauss linearUpwind grad(U)",
        "div((nuEff*dev2(T(grad(U)))))": "Gauss linear",
    }
    f["laplacianSchemes"] = {"default": "Gauss linear corrected"}
    f["interpolationSchemes"] = {"default": "linear"}
    f["snGradSchemes"] = {"default": "corrected"}

# ------------------------------------------------------------
# system/fvSolution (only p/U)
# ------------------------------------------------------------
with case.fv_solution as f:
    f["solvers"] = {
        "p": {
            "solver": "GAMG",
            "tolerance": 1e-7,
            "relTol": 0.01,
            "smoother": "DICGaussSeidel",
        },
        "pFinal": {
            "solver": "GAMG",
            "tolerance": 1e-7,
            "relTol": 0.0,
            "smoother": "DICGaussSeidel",
        },
        "U": {
            "solver": "smoothSolver",
            "smoother": "symGaussSeidel",
            "tolerance": 1e-5,
            "relTol": 0.1,
        },
        "UFinal": {
            "solver": "smoothSolver",
            "smoother": "symGaussSeidel",
            "tolerance": 1e-5,
            "relTol": 0.0,
        },
    }

    f["PIMPLE"] = {
        "nOuterCorrectors": 1,
        "nCorrectors": 2,
        "nNonOrthogonalCorrectors": 0,
    }

# ------------------------------------------------------------
# system/blockMeshDict (same geometry as RAS)
# ------------------------------------------------------------
with case.block_mesh_dict as f:
    f["scale"] = 0.001

    f["vertices"] = [
        [-20.6, 0.0, -0.5],   # 0
        [-20.6, 25.4, -0.5],  # 1
        [0.0, -25.4, -0.5],   # 2
        [0.0, 0.0, -0.5],     # 3
        [0.0, 25.4, -0.5],    # 4
        [206.0, -25.4, -0.5], # 5
        [206.0, 0.0, -0.5],   # 6
        [206.0, 25.4, -0.5],  # 7
        [290.0, -16.6, -0.5], # 8
        [290.0, 0.0, -0.5],   # 9
        [290.0, 16.6, -0.5],  # 10

        [-20.6, 0.0, 0.5],    # 11
        [-20.6, 25.4, 0.5],   # 12
        [0.0, -25.4, 0.5],    # 13
        [0.0, 0.0, 0.5],      # 14
        [0.0, 25.4, 0.5],     # 15
        [206.0, -25.4, 0.5],  # 16
        [206.0, 0.0, 0.5],    # 17
        [206.0, 25.4, 0.5],   # 18
        [290.0, -16.6, 0.5],  # 19
        [290.0, 0.0, 0.5],    # 20
        [290.0, 16.6, 0.5],   # 21
    ]

    f["blocks"] = [
        "hex",
        [0, 3, 4, 1, 11, 14, 15, 12],
        [18, 30, 1],
        "simpleGrading",
        [0.5, 1.0, 1.0],

        "hex",
        [2, 5, 6, 3, 13, 16, 17, 14],
        [180, 27, 1],
        "simpleGrading",
        [1.0, 1.0, 1.0],

        "hex",
        [3, 6, 7, 4, 14, 17, 18, 15],
        [180, 30, 1],
        "simpleGrading",
        [1.0, 1.0, 1.0],

        "hex",
        [5, 8, 9, 6, 16, 19, 20, 17],
        [25, 27, 1],
        "simpleGrading",
        [2.5, 1.0, 1.0],

        "hex",
        [6, 9, 10, 7, 17, 20, 21, 18],
        [25, 30, 1],
        "simpleGrading",
        [2.5, 1.0, 1.0],
    ]

    f["edges"] = []

    f["boundary"] = [
        (
            "inlet",
            {
                "type": "patch",
                "faces": [[0, 1, 12, 11]],
            },
        ),
        (
            "outlet",
            {
                "type": "patch",
                "faces": [
                    [8, 9, 20, 19],
                    [9, 10, 21, 20],
                ],
            },
        ),
        (
            "upperWall",
            {
                "type": "wall",
                "faces": [
                    [1, 4, 15, 12],
                    [4, 7, 18, 15],
                    [7, 10, 21, 18],
                ],
            },
        ),
        (
            "lowerWall",
            {
                "type": "wall",
                "faces": [
                    [0, 3, 14, 11],
                    [3, 2, 13, 14],
                    [2, 5, 16, 13],
                    [5, 8, 19, 16],
                ],
            },
        ),
        (
            "frontAndBack",
            {
                "type": "empty",
                "faces": [
                    [0, 3, 4, 1],
                    [2, 5, 6, 3],
                    [3, 6, 7, 4],
                    [5, 8, 9, 6],
                    [6, 9, 10, 7],
                    [11, 14, 15, 12],
                    [13, 16, 17, 14],
                    [14, 17, 18, 15],
                    [16, 19, 20, 17],
                    [17, 20, 21, 18],
                ],
            },
        ),
    ]
    f["mergePatchPairs"] = []

# ------------------------------------------------------------
# constant/transportProperties
# ------------------------------------------------------------
with case.transport_properties as f:
    f["transportModel"] = "Newtonian"
    f["nu"] = Dimensioned(
        1e-5,
        DimensionSet(length=2, time=-1),
        "nu",
    )

# ------------------------------------------------------------
# constant/turbulenceProperties (LES Smagorinsky)
# ------------------------------------------------------------
with case.turbulence_properties as f:
    f["simulationType"] = "LES"
    f["LES"] = {
        "LESModel": "Smagorinsky",
        "turbulence": "on",
        "printCoeffs": "on",
        "delta": "cubeRootVol",
    }

# ------------------------------------------------------------
# 0/U
# ------------------------------------------------------------
with case[0]["U"] as f:
    f.dimensions = DimensionSet(length=1, time=-1)
    f.internal_field = [0.0, 0.0, 0.0]
    f.boundary_field = {
        "inlet": {"type": "fixedValue", "value": [1.5, 0.0, 0.0]},
        "outlet": {"type": "zeroGradient"},
        "upperWall": {"type": "noSlip"},
        "lowerWall": {"type": "noSlip"},
        "frontAndBack": {"type": "empty"},
    }

# ------------------------------------------------------------
# 0/p (kinematic)
# ------------------------------------------------------------
with case[0]["p"] as f:
    f.dimensions = DimensionSet(length=2, time=-2)
    f.internal_field = 0.0
    f.boundary_field = {
        "inlet": {"type": "zeroGradient"},
        "outlet": {"type": "fixedValue", "value": 0.0},
        "upperWall": {"type": "zeroGradient"},
        "lowerWall": {"type": "zeroGradient"},
        "frontAndBack": {"type": "empty"},
    }

# ------------------------------------------------------------
# 0/nut (SGS viscosity)
# ------------------------------------------------------------
with case[0]["nut"] as f:
    f.dimensions = DimensionSet(length=2, time=-1)
    f.internal_field = 0.0
    f.boundary_field = {
        "inlet": {"type": "calculated", "value": 0.0},
        "outlet": {"type": "calculated", "value": 0.0},
        "upperWall": {"type": "nutkWallFunction", "value": 0.0},
        "lowerWall": {"type": "nutkWallFunction", "value": 0.0},
        "frontAndBack": {"type": "empty"},
    }

# ------------------------------------------------------------
# Run and post-process
# ------------------------------------------------------------
case.run()

last_time = case[-1]

cc = last_time.cell_centers().internal_field
assert isinstance(cc, np.ndarray)
x, y, z = cc.T

U = last_time["U"].internal_field
assert isinstance(U, np.ndarray)
Ux, Uy, Uz = U.T

print("LES case:")
print("  Number of cells :", U.shape[0])
print("  Ux range        :", Ux.min(), "to", Ux.max())
print("  Uy range        :", Uy.min(), "to", Uy.max())
print("  Uz range        :", Uz.min(), "to", Uz.max())


from particle_tracking import run_particles

# # ... your LES CFD setup above ...
# case.run()

traj_times_les, traj_xp_les = run_particles(
    case,
    out_name="particles_les",
    n_particles=10,
)



