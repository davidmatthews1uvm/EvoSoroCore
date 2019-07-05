import math
import uuid
import os
from subprocess import Popen

import numpy as np
from scipy import spatial
from lxml import etree

from parallelpy.utils import Work, Letter
from evodevo.moo_interfaces import MOORobotInterface


class SoftbotRobotBase(MOORobotInterface):
    def __init__(self, robot, seq_num_gen, sim, env, vxc_wrapper, materials, run_dir):
        self.sim = sim
        self.env = env
        self.vxa_wrapper = vxc_wrapper
        self.materials = materials
        self.run_dir = run_dir
        self.seq_num_gen = seq_num_gen
        self.seq_num = self.seq_num_gen()
        self.robot = robot
        self.id = self.set_uuid()

        self.fitness = ({}, {})  # (train, test)

        self.needs_eval = True

        self.age = 0

    def get_id(self):
        return self.seq_num

    def get_body_xlim(self):
        return self.robot.get_body_xlim()

    def get_body_ylim(self):
        return self.robot.get_body_ylim()

    def get_body_zlim(self):
        return self.robot.get_body_zlim()

    # Methods for MOORObotInterface class

    def iterate_generation(self):
        self.age += 1

    def needs_evaluation(self):
        return self.needs_eval

    def mutate(self):
        self.needs_eval = True
        self.robot.mutate()
        self.set_uuid()
        self.seq_num = self.seq_num_gen()
        self.fitness = ({}, {})

    def get_minimize_vals(self):
        raise NotImplementedError()

    def get_maximize_vals(self):
        raise NotImplementedError()

    def get_seq_num(self):
        return self.seq_num

    def get_fitness(self, test=False):
        ret = np.sum(list(self._flatten(self.fitness[0].values())))
        if test:
             ret += np.sum(list(self._flatten(self.fitness[1].values())))
        # print (ret)

        if np.isnan(ret) or np.isinf(ret) or ret > 30:
            return 0
        else:
            return ret       

    def get_data(self):
        ret = [self.get_fitness(), self.get_fitness(test=True), self.get_age()]
        ret += list(self._flatten(self.fitness[0].values()))
        ret += list(self._flatten(self.fitness[1].values()))
        return ret

    def get_data_column_count(self):
        """
        Returns the number of data columns needed to store this robot
        :param num_commands: number of train_commands being sent to the robot
        :return: number of data columns needed to store this robot
        """
        return 3 + 1

    def dominates_final_selection(self, other):
        return self.get_fitness() > other.get_fitness()

    # Methods for Work class
    def cpus_requested(self):
        return 1

    def compute_work(self, test=True, **kwargs):

        # generate the XML soft robot specs
        root = etree.Element("VXA", Version="1.0")
        self.sim.write_to_xml(root, self.run_dir, self)
        self.env.write_to_xml(root, self.run_dir, self)
        self.vxa_wrapper.write_to_xml(root, self.materials, self.robot)
        xml_str = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="ISO-8859-1").decode("utf-8")

        # write the robot specs to file
        with open("%s/voxelyzeFiles/%05i.vxa"%(self.run_dir, self.seq_num),"w") as f:
            f.write(xml_str)

        # simulate the robot
        p = Popen(("./voxelyze -f %s/voxelyzeFiles/%05i.vxa"%(self.run_dir, self.seq_num)).split())
        p.communicate(timeout=60) # if simulation takes longer than 60 seconds, assume it failed.

        # read the fitness in
        res = etree.parse("%s/fitnessFiles/softbotsOutput--id_%05i.xml"%(self.run_dir, self.seq_num))
        self.fitness[0]["Train"] = [float(res.find("Fitness").find("normDistZ").text)]

    def write_letter(self):
        # print("writing letter")
        return Letter(self.fitness, None)

    def open_letter(self, letter):
        # print("opening")
        self.fitness = letter.get_data()
        self.needs_eval = False
        return None

    def set_uuid(self):
        self.id = uuid.uuid1()
        return self.id

    def get_num_evaluations(self, test=False):
        return 1

    def get_age(self):
        return self.age

    def _flatten(self, l):
        ret = []
        for items in l:
            ret += items
        return ret

