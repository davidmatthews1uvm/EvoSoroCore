import math
import uuid

import numpy as np
from scipy import spatial

from parallelpy.utils import Work, Letter

from experiments.Test.softbot_robot_base import SoftbotRobotBase

class SoftbotRobot(SoftbotRobotBase):
    def __init__(self, robot, seq_num, sim, env, vxc_wrapper, materials, run_dir):
        super().__init__(robot, seq_num, sim, env, vxc_wrapper, materials, run_dir)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "AFPO BOT: f: %.2f age: %d -- ID: %s" % (self.get_fitness(), self.get_age(), str(self.seq_num))

    def get_minimize_vals(self):
        return [self.get_age(), self.get_fitness()]

    def get_maximize_vals(self):
        return []

