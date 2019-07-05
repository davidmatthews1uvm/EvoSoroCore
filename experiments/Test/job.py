import time
import sys
import os

import numpy
import random
from subprocess import call

from evodevo.evo_run import EvolutionaryRun

from parallelpy import parallel_evaluate

sys.path.insert(0, "../..")

from Simulator import Sim
from Environment import Env, VXC_Wrapper
from Material import Material
from Robot import Robot

from experiments.Test.softbot_robot_fit_afpo import SoftbotRobot
from experiments.Test.seq_num import get_seq_num


parallel_evaluate.MAX_THREADS = 24
# parallel_evaluate.DEBUG=True

# check if we are running on the VACC. if so. disable debug mode.
if (os.getenv("VACC") != None):
    parallel_evaluate.DEBUG = False
    parallel_evaluate.MAX_THREADS = 24

parallel_evaluate.setup(parallel_evaluate.PARALLEL_MODE_MPI_INTER)

def get_internal_bot():
    return None

robot_factory = None

EVAL_TIME = 500
POP_SIZE = 20
GENS = 100

if __name__ == '__main__':
    assert len(sys.argv) >= 3, "please run as python job.py seed experiment_name"
    seed = int(sys.argv[1])
    name = sys.argv[2]

    numpy.random.seed(seed)
    random.seed(seed)

    numpy.set_printoptions(suppress=True, formatter={'float_kind': lambda x: '%4.2f' % x})

    print(name)

    def get_internal_bot():
        return Robot()
    sim = Sim()
    env = Env()
    vxc_wrapper = VXC_Wrapper()
    materials = [Material(1, name="Basic")]

    # Setup evo run
    if robot_factory is None:
        def robot_factory():
            internal_robot = get_internal_bot()
            return SoftbotRobot(internal_robot, get_seq_num, sim, env, vxc_wrapper, materials, "run_%d" % seed)
    
    
    pop_size = POP_SIZE
    generations = GENS

    def create_new_job():
        return EvolutionaryRun(robot_factory, generations, seed, pop_size=pop_size,
                               experiment_name=name)
    # run evo run.
    job_name = name + "_run_" + str(seed)
    print("Starting run with %d individuals for %d generations" % (pop_size, generations))



    evolutionary_run = create_new_job()
    # make all the needed directories...
    run_dir = "run_%d"%seed
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("%s/voxelyzeFiles"%run_dir, exist_ok=True)
    os.makedirs("%s/fitnessFiles"%run_dir, exist_ok=True)
    call(["cp", "/mnt/hgfs/VM's Shared/LevinLab/src/research_code/evosoro/_voxcad/voxelyzeMain/voxelyze", "."])

    evolutionary_run.run_full(printing=True)