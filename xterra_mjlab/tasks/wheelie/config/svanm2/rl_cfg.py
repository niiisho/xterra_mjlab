"""PPO config for wheelie tasks."""

from xterra_mjlab.tasks.velocity.config.svanm2.rl_cfg import svanm2_ppo_runner_cfg


def svanm2_wheelie_ppo_runner_cfg():
    cfg = svanm2_ppo_runner_cfg()
    cfg.experiment_name = "svanm2_wheelie"
    cfg.max_iterations = 3000
    return cfg
