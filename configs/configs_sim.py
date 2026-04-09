

from __future__ import annotations

# --- Rot batch partition (`partition_batched_kernel` + `discrete_rotation_kernel`) ---
NUM_ENVS = 4
NB = 20
DIMS = (0.6, 0.4, 0.3)
MIN_RATIO = 0.04
SHRINK = 0.95
SEED = 42
ISOTROPIC = True

# --- Time integration ---
FPS = 30
SIM_SUBSTEPS = 4

# --- Scene layout ---
ENV_SPACING = 1.5
DROP_HEIGHT = 0.03

# --- Shape friction / damping ---
GROUND_MU = 1.0
GROUND_KD = 300.0
GROUND_MU_TORSIONAL = 0.02
GROUND_MU_ROLLING = 0.01

WALL_MU = 1.0
WALL_KD = 300.0
WALL_MU_TORSIONAL = 0.03
WALL_MU_ROLLING = 0.015

BOX_MU = 1.0
BOX_KD = 300.0
BOX_MU_TORSIONAL = 0.04
BOX_MU_ROLLING = 0.02

WALL_THICKNESS = 0.02
WALL_SCALE = 1.1

# --- Newton collision pipeline ---
DEFAULT_RIGID_GAP = 0.005
DEFAULT_CONTACTS_PER_BODY = 30
MIN_RIGID_CONTACT_MAX = 50000

# --- XPBD ---
XPBD_ITERATIONS = 8
XPBD_CONTACT_RELAXATION = 0.8
ANGULAR_DAMPING = 0.1
XPBD_ENABLE_RESTITUTION = False

# --- MuJoCo solver (Warp) ---
MUJOCO_ITERATIONS = 30
MUJOCO_LS_ITERATIONS = 50
MUJOCO_TOLERANCE = 1.0e-6
MUJOCO_LS_TOLERANCE = 0.02
MUJOCO_SOLVER = "newton"
MUJOCO_INTEGRATOR = "implicitfast"
MUJOCO_CONE = "pyramidal"
MUJOCO_IMPRATIO = 1.0
MUJOCO_UPDATE_DATA_INTERVAL = 1
MUJOCO_USE_CONTACTS = False

# --- MuJoCo contact (ShapeConfig / geom solref) ---
MUJOCO_CONTACT_KE = 2.0e4
MUJOCO_CONTACT_KD = 2.0e3
MUJOCO_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)

# --- Settle detection (`check_body_stability_lin_ang`: separate |v| and |ω| thresholds, same value per world) ---
# `SETTLE_CHECK_INTERVAL`: run stability kernel every this many *viewer frames* (each frame runs
# SIM_SUBSTEPS physics substeps; substeps are not sampled).
# `SETTLE_STEPS`: a world is settled after this many *consecutive* successful checks (each check
# sees zero unsettled bodies in that world). This is NOT "N consecutive stable frames" — only
# frames where a check runs count, and only if that check passes.
SETTLE_LINEAR_SPEED_MPS = 0.02
SETTLE_ANGULAR_SPEED_RAD = 0.1
SETTLE_CHECK_INTERVAL = 3
SETTLE_STEPS = 5

# --- Other argparse defaults ---
DEFAULT_SOLVER = "xpbd"
RIGID_CONTACT_MAX = 0
NCONMAX = 0
NJMAX = 0
