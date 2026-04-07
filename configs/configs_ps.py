"""
Partition + Newton rigid-body defaults.

Edit constants here. `partition_sim_example.py` uses them for sim/collision; pass
`--solver` on the command line to choose `xpbd` or `mujoco` (see DEFAULT_SOLVER).
"""

from __future__ import annotations

# --- Time integration ---
FPS = 60
SIM_SUBSTEPS = 10

# --- Partition (`partition_batched_kernel`: device RNG via `seed`) ---
NUM_ENVS = 4
NB = 20
DIMS = (0.6, 0.4, 0.3)
MIN_RATIO = 0.05
SHRINK = 0.95
SEED = 42

# --- Scene / builder contact ---
DROP_HEIGHT = 0.1
ENV_SPACING = 0.3
GROUND_MU = 1.0
# ShapeConfig.restitution: XPBD only; must pair with XPBD_ENABLE_RESTITUTION.
SHAPE_RESTITUTION = 0.0
DEFAULT_RIGID_GAP = 0.005
DEFAULT_CONTACTS_PER_BODY = 30
MIN_RIGID_CONTACT_MAX = 50000
RIGID_CONTACT_MAX = 0

# --- CLI default for `--solver` only ---
DEFAULT_SOLVER = "xpbd"

# --- XPBD ---
XPBD_ITERATIONS = 10
XPBD_CONTACT_RELAXATION = 0.8
XPBD_ANGULAR_DAMPING = 0.1
XPBD_ENABLE_RESTITUTION = False

# --- MuJoCo buffers ---
NCONMAX = 0
NJMAX = 0

# --- MuJoCo solver ---
MUJOCO_ITERATIONS = 15
MUJOCO_LS_ITERATIONS = 100
MUJOCO_SOLVER = "newton"
MUJOCO_INTEGRATOR = "implicitfast"
MUJOCO_CONE = "pyramidal"
MUJOCO_IMPRATIO = 1.0
MUJOCO_TOLERANCE = 1.0e-5
MUJOCO_LS_TOLERANCE = 0.02
MUJOCO_UPDATE_DATA_INTERVAL = 1
MUJOCO_USE_CONTACTS = False

# ShapeConfig ke/kd (MuJoCo maps to geom solref); used for all solvers in this example.
MUJOCO_CONTACT_KE = 2.0e4
MUJOCO_CONTACT_KD = 2.0e2
MUJOCO_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
