"""Unload-only knobs not stored in snapshot metadata_json."""

# Viewer spacing between replicated envs (see plans/unload_plan_batch set_world_offsets).
ENV_SPACING = 1.5
# Graveyard Z for removed bodies (stack_simple_batch convention).
REMOVED_BODY_Z = -100.0
# When --settle-cooldown-frames is -1: skip removal/settle logic for this many frames after each removal.
# Counts simulation frames (each frame runs sim_substeps physics substeps).
DEFAULT_SETTLE_COOLDOWN_FRAMES = 3
# Rotation term scale in inter-steady performance metric (kernel `rot_scale`).
INTER_STEADY_METRIC_ROT_SCALE = 1.0
