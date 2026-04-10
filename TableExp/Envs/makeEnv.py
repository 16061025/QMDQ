from Envs.ABCEnv import ABCEnv
from Envs.rightenv import RightEnv
from Envs.GridWorldEnv import GridWorldEnv
from Envs.CliffWalkEnv import CliffWalkingEnv
from Envs.SameActionEnv import SameActionEnv
from Envs.multiarmEnv import multiarmEnv

def makeEnv(cfg):
    if cfg.env == "rightenv":
        env = RightEnv(submean=cfg.submean, optimalmean=cfg.optimalmean, subvar=cfg.subvar, optimalvar=cfg.optimalvar)
    elif cfg.env == "grid":
        env = GridWorldEnv(col=cfg.ncol, row=cfg.nrow, gridr1=cfg.gridr1, gridr2=cfg.gridr2, gridgr1=cfg.gridgr1, gridgr2=cfg.gridgr2)
    elif cfg.env == "ABCenv":
        env = ABCEnv(meanB=cfg.submean, meanC=0)
    elif cfg.env == "cliffenv":
        env = CliffWalkingEnv(rows=cfg.nrow, cols=cfg.ncol, cliff_reward=cfg.gridr2, step_reward=cfg.gridr1, goal_reward=cfg.gridgr1)
    elif cfg.env == "SameActionenv":
        env = SameActionEnv(mean=cfg.submean, var=cfg.subvar)
    elif cfg.env == "multiarmenv":
        env = multiarmEnv(arm_count=cfg.Narm, std=cfg.armstd)
    else:
        raise ValueError("Unknown env %s" % cfg.env)
    return env