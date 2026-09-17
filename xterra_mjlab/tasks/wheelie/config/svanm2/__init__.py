"""Register SvanM2 wheelie tasks."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .wheelie_env_cfg import svanm2_front_wheelie_cfg, svanm2_rear_wheelie_cfg, svanm2_front_wheelie_fromflat_cfg
from .rl_cfg import svanm2_wheelie_ppo_runner_cfg

FRONT_WHEELIE_TASK = "xTerra-Mjlab-Wheelie-Front-SvanM2"
FRONT_WHEELIE_FROMFLAT_TASK = "xTerra-Mjlab-Wheelie-Front-FromFlat-SvanM2"
REAR_WHEELIE_TASK = "xTerra-Mjlab-Wheelie-Rear-SvanM2"

register_mjlab_task(
    task_id=FRONT_WHEELIE_TASK,
    env_cfg=svanm2_front_wheelie_cfg(),
    play_env_cfg=svanm2_front_wheelie_cfg(play=True),
    rl_cfg=svanm2_wheelie_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id=FRONT_WHEELIE_FROMFLAT_TASK,
    env_cfg=svanm2_front_wheelie_fromflat_cfg(),
    play_env_cfg=svanm2_front_wheelie_fromflat_cfg(play=True),
    rl_cfg=svanm2_wheelie_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id=REAR_WHEELIE_TASK,
    env_cfg=svanm2_rear_wheelie_cfg(),
    play_env_cfg=svanm2_rear_wheelie_cfg(play=True),
    rl_cfg=svanm2_wheelie_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

__all__ = ["FRONT_WHEELIE_TASK", "REAR_WHEELIE_TASK"]
