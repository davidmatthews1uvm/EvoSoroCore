import random

from lxml import etree


class Sim(object):
    """Container for VoxCad simulation parameters."""

    def __init__(self, self_collisions_enabled=True, simulation_time=10.5, dt_frac=0.9, stop_condition=2,
                 fitness_eval_init_time=0.5, actuation_start_time=0, equilibrium_mode=0, min_temp_fact=0.1,
                 max_temp_fact_change=0.00001, max_stiffness_change=10000, min_elastic_mod=5e006,
                 max_elastic_mod=5e008, damp_evolved_stiffness=True):
        self.sub_groups = ["Integration", "Damping", "Collisions", "Features", "StopCondition", "EquilibriumMode", "GA"]
        # custom nested things in "SurfMesh", "CMesh"

        self.self_collisions_enabled = self_collisions_enabled
        self.simulation_time = simulation_time
        self.dt_frac = dt_frac
        self.stop_condition = stop_condition
        self.fitness_eval_init_time = fitness_eval_init_time
        self.actuation_start_time = actuation_start_time
        self.equilibrium_mode = equilibrium_mode
        self.min_temp_fact = min_temp_fact
        self.max_temp_fact_change = max_temp_fact_change
        self.max_stiffness_change = max_stiffness_change
        self.min_elastic_mod = min_elastic_mod
        self.max_elastic_mod = max_elastic_mod
        self.damp_evolved_stiffness = damp_evolved_stiffness

    def write_to_xml(self, root, run_dir, individual, fitness_file_str=None, **kwargs):
        sim_root = etree.SubElement(root, "Simulator")

        etree.SubElement(sim_root, "Seed").text = str(random.random())

        integration = etree.SubElement(sim_root, "Integration")
        etree.SubElement(integration, "Integrator").text = "0"
        etree.SubElement(integration, "DtFrac").text = str(self.dt_frac)

        damping = etree.SubElement(sim_root, "Damping")
        etree.SubElement(damping, "BondDampingZ").text = "1"
        etree.SubElement(damping, "ColDampingZ").text = "0.8"
        etree.SubElement(damping, "ColDampingZ").text = "0.8"
        etree.SubElement(damping, "SlowDampingZ").text = "0.01"

        collisions = etree.SubElement(sim_root, "Collisions")
        etree.SubElement(collisions, "SelfColEnabled").text = str(int(self.self_collisions_enabled))
        etree.SubElement(collisions, "ColSystem").text = "3"
        etree.SubElement(collisions, "CollisionHorizion").text = "2"

        features = etree.SubElement(sim_root, "Features")
        etree.SubElement(features, "FluidDampEnabled").text = "0"
        etree.SubElement(features, "PoissonKickBackEnabled").text = "0"
        etree.SubElement(features, "EnforceLatticeEnabled").text = "0"

        surf_mesh = etree.SubElement(sim_root, "SurfMesh")
        c_mesh = etree.SubElement(surf_mesh, "CMesh")
        etree.SubElement(c_mesh, "DrawSmooth").text = "1"
        etree.SubElement(c_mesh, "Vertices")
        etree.SubElement(c_mesh, "Facets")
        etree.SubElement(c_mesh, "Lines")

        stop_condition = etree.SubElement(sim_root, "StopCondition")
        etree.SubElement(stop_condition, "StopConditionType").text = str(int(self.stop_condition))
        etree.SubElement(stop_condition, "StopConditionValue").text = str(self.simulation_time)
        etree.SubElement(stop_condition, "InitCmTime").text = str(self.fitness_eval_init_time)
        etree.SubElement(stop_condition, "ActuationStartTime").text = str(self.actuation_start_time)

        equilibrium_mode = etree.SubElement(sim_root, "EquilibriumMode")
        etree.SubElement(equilibrium_mode, "EquilibriumModeEnabled").text = str(self.equilibrium_mode)

        G_A = etree.SubElement(sim_root, "GA")
        etree.SubElement(G_A, "WriteFitnessFile").text = "1"
        if fitness_file_str is not None:
            etree.SubElement(G_A, "FitnessFileName").text = fitness_file_str
        else:
            etree.SubElement(G_A, "FitnessFileName").text = "%s/fitnessFiles/softbotsOutput--id_%05i.xml" % (run_dir, individual.get_id())
        etree.SubElement(G_A, "QhullTmpFile").text = "%s/../_qhull/tempFiles/qhullInput--id_%05i.txt" % (run_dir, individual.get_id())
        etree.SubElement(G_A, "CurvaturesTmpFile").text = "%s/../_qhull/tempFiles/curvatures--id_%05i.txt" % (run_dir, individual.get_id())

        etree.SubElement(sim_root, "MinTempFact").text = str(self.min_temp_fact)
        etree.SubElement(sim_root, "MaxTempFactChange").text = str(self.max_temp_fact_change)
        etree.SubElement(sim_root, "DampEvolvedStiffness").text = str(int(self.damp_evolved_stiffness))
        etree.SubElement(sim_root, "MaxStiffnessChange").text = str(self.max_stiffness_change)
        etree.SubElement(sim_root, "MinElasticMod").text = str(self.min_elastic_mod)
        etree.SubElement(sim_root, "MaxElasticMod").text = str(self.max_elastic_mod)
        etree.SubElement(sim_root, "ErrorThreshold").text = str(0)
        etree.SubElement(sim_root, "ThresholdTime").text = str(0)
        etree.SubElement(sim_root, "MaxKP").text = str(0)
        etree.SubElement(sim_root, "MaxKI").text = str(0)
        etree.SubElement(sim_root, "MaxANTIWINDUP").text = str(0)

        if hasattr(individual, "parent_lifetime"):
            if individual.parent_lifetime > 0:
                etree.SubElement(sim_root, "ParentLifetime").text = str(individual.parent_lifetime)
            else:
                etree.SubElement(sim_root, "ParentLifetime").text = str(individual.lifetime)


        return sim_root