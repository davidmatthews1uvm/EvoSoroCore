import random
from copy import deepcopy

import numpy as np

from evosorocore.NetworkUtils import sigmoid
from evosorocore.Networks import Network


# from evosoro.tools.utils import sigmoid, xml_format, dominates


class Genotype(object):
    """A container for multiple networks, 'genetic code' copied with modification to produce offspring."""

    NET_DICT = None  # used to create new individuals with presupposed features

    def __init__(self, orig_size_xyz=(6, 6, 6)):

        """
        Parameters
        ----------
        orig_size_xyz : 3-tuple (x, y, z)
            Defines the original 3 dimensions for the cube of voxels corresponding to possible networks outputs. The
            maximum number of SofBot voxel components is x*y*z, a full cube.

        """
        self.networks = []
        self.all_networks_outputs = []
        self.to_phenotype_mapping = GenotypeToPhenotypeMap()
        self.orig_size_xyz = orig_size_xyz

    def __iter__(self):
        """Iterate over the networks. Use the expression 'for n in network'."""
        return iter(self.networks)

    def __len__(self):
        """Return the number of networks in the genotype. Use the expression 'len(network)'."""
        return len(self.networks)

    def __getitem__(self, n):
        """Return network n.  Use the expression 'network[n]'."""
        return self.networks[n]

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def __str__(self):
        return "Genotype made of: " + ',\n'.join(map(str, self.networks))

    def mutate(self, **kwargs):
        """
        Selects one network at random and attempts to mutate it.
        :param kwargs: Not currently used.
        :return: None
        """
        random.choice(self.networks).mutate()

    def add_network(self, network, freeze=False, switch=False, num_consecutive_mutations=1):
        """Append a new network to this list of networks.

        Parameters
        ----------
        network : Network
            The network to add. Should be a subclass of Network.

        freeze : bool
            This indicator is used to prevent mutations to a network while freeze == True

        switch : bool
            For learning trials

        num_consecutive_mutations : int
            Uses this many (random) steps per mutation.

        """
        assert isinstance(network, Network)
        network.freeze = freeze
        network.switch = switch
        network.num_consecutive_mutations = num_consecutive_mutations
        self.networks += [network]
        self.all_networks_outputs.extend(network.output_node_names)

    def express(self):
        """Calculate the genome networks outputs, the physical properties of each voxel for simulation"""

        for network in self:
            if not network.direct_encoding:
                for name in network.graph.nodes():
                    network.graph.node[name]["evaluated"] = False  # flag all nodes as unevaluated

                network.set_input_node_states(self.orig_size_xyz)  # reset the inputs

                for name in network.output_node_names:
                    network.graph.node[name]["state"] = np.zeros(self.orig_size_xyz)  # clear old outputs
                    network.graph.node[name]["state"] = self.calc_node_state(network, name)  # calculate new outputs

        for network in self:
            for name in network.output_node_names:
                if name in self.to_phenotype_mapping:
                    if not network.direct_encoding:
                        self.to_phenotype_mapping[name]["state"] = network.graph.node[name]["state"]
                    else:
                        self.to_phenotype_mapping[name]["state"] = network.values

        for name, details in self.to_phenotype_mapping.items():
            # details["old_state"] = copy.deepcopy(details["state"])
            # SAM: moved this to mutation.py prior to mutation attempts loop
            if name not in self.all_networks_outputs:
                details["state"] = np.ones(self.orig_size_xyz, dtype=details["output_type"]) * -999
                if details["dependency_order"] is not None:
                    for dependency_name in details["dependency_order"]:
                        self.to_phenotype_mapping.dependencies[dependency_name]["state"] = None

        for name, details in self.to_phenotype_mapping.items():
            if details["dependency_order"] is not None:
                details["state"] = details["func"](self)

    def calc_node_state(self, network, node_name):
        """Propagate input values through the network"""
        if network.graph.node[node_name]["evaluated"]:
            return network.graph.node[node_name]["state"]

        network.graph.node[node_name]["evaluated"] = True
        input_edges = network.graph.in_edges(nbunch=[node_name])
        new_state = np.zeros(self.orig_size_xyz)

        for edge in input_edges:
            node1, node2 = edge
            new_state += self.calc_node_state(network, node1) * network.graph.edges[node1, node2]["weight"]

        network.graph.node[node_name]["state"] = new_state

        if node_name in self.to_phenotype_mapping:
            if self.to_phenotype_mapping[node_name]["dependency_order"] is None:
                return self.to_phenotype_mapping[node_name]["func"](new_state)

        return network.graph.node[node_name]["function"](new_state)


class GenotypeToPhenotypeMap(object):
    """A mapping of the relationship from genotype (networks) to phenotype (VoxCad simulation)."""

    # TODO: generalize dependencies from boolean to any operation (e.g.  to set
    # an env param from multiple outputs)

    def __init__(self):
        self.mapping = dict()
        self.dependencies = dict()

    def items(self):
        """to_phenotype_mapping.items() -> list of (key, value) pairs in mapping"""
        return [(key, self.mapping[key]) for key in self.mapping]

    def __contains__(self, key):
        """Return True if key is a key str in the mapping, False otherwise. Use the expression 'key in mapping'."""
        try:
            return key in self.mapping
        except TypeError:
            return False

    def __len__(self):
        """Return the number of mappings. Use the expression 'len(mapping)'."""
        return len(self.mapping)

    def __getitem__(self, key):
        """Return mapping for node with name 'key'.  Use the expression 'mapping[key]'."""
        return self.mapping[key]

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def add_map(self, name, tag, func=sigmoid, output_type=float, dependency_order=None, params=None, param_tags=None,
                env_kws=None, logging_stats=np.mean, age_zero_overwrite=None, switch_proportion=0, switch_name=None):
        """Add an association between a genotype output and a VoxCad parameter.

        Parameters
        ----------
        name : str
            A network output node name from the genotype.

        tag : str
            The tag used in parsing the resulting output from a VoxCad simulation.
            If this is None then the attribute is calculated outside of VoxCad (in Python only).

        func : func
            Specifies relationship between attributes and xml tag.

        output_type : type
            The output type

        dependency_order : list
            Order of operations

        params : list
            Constants dictating parameters of the mapping

        param_tags : list
            Tags for any constants associated with the mapping

        env_kws : dict
            Specifies which function of the output state to use (on top of func) to set an Env attribute

        logging_stats : func or list
            One or more functions (statistics) of the output to be logged as additional column(s) in logging

        age_zero_overwrite : str
            Evaluate this network with this placeholder at birth (age=0) instead of actual values.

        switch_proportion : float
            Switches are non-inheritable portions of genotype (Hinton & Nowlan, 1987).

        switch_name : str
            Network name containing switch values

        """
        if (dependency_order is not None) and not isinstance(dependency_order, list):
            dependency_order = [dependency_order]

        if params is not None:
            assert (param_tags is not None)
            if not isinstance(params, list):
                params = [params]

        if param_tags is not None:
            assert (params is not None)
            if not isinstance(param_tags, list):
                param_tags = [param_tags]

        if (env_kws is not None) and not isinstance(env_kws, dict):
            env_kws = {env_kws: np.mean}

        if (logging_stats is not None) and not isinstance(logging_stats, list):
            logging_stats = [logging_stats]

        self.mapping[name] = {"tag": tag,
                              "func": func,
                              "dependency_order": dependency_order,
                              "state": None,
                              "old_state": None,
                              "output_type": output_type,
                              "params": params,
                              "param_tags": param_tags,
                              "env_kws": env_kws,
                              "logging_stats": logging_stats,
                              "age_zero_overwrite": age_zero_overwrite,
                              "switch_proportion": switch_proportion,
                              "switch_name": switch_name}

    def add_output_dependency(self, name, dependency_name, requirement, material_if_true=None, material_if_false=None):
        """Add a dependency between two genotype outputs.

        Parameters
        ----------
        name : str
            A network output node name from the genotype.

        dependency_name : str
            Another network output node name.

        requirement : bool
            Dependency must be this

        material_if_true : int
            The material if dependency meets pre-requisite

        material_if_false : int
            The material otherwise

        """
        self.dependencies[name] = {"depends_on": dependency_name,
                                   "requirement": requirement,
                                   "material_if_true": material_if_true,
                                   "material_if_false": material_if_false,
                                   "state": None}

    def get_dependency(self, name, output_bool):
        """Checks recursively if all boolean requirements were met in dependent outputs."""
        if self.dependencies[name]["depends_on"] is not None:
            dependency = self.dependencies[name]["depends_on"]
            requirement = self.dependencies[name]["requirement"]
            return np.logical_and(self.get_dependency(dependency, True) == requirement,
                                  self.dependencies[name]["state"] == output_bool)
        else:
            return self.dependencies[name]["state"] == output_bool


class Phenotype(object):
    """Physical manifestation of the genotype - determines the physiology of an individual."""

    def __init__(self, genotype):

        """
        Parameters
        ----------
        genotype : Genotype()
            Defines particular networks (the genome).

        """
        self.genotype = genotype
        self.genotype.express()

    def __deepcopy__(self, memo):
        """Override deepcopy to apply to class level attributes"""
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(deepcopy(self.__dict__, memo))
        return new

    def __str__(self):
        return '\n'.join([str(name) + ": " + str(details["state"]) for (name, details) in self.genotype.to_phenotype_mapping.items()])

    def get_phenotype(self):
        return self.genotype.to_phenotype_mapping.items()

    def mutate(self, max_mutation_attempts=1000):
        """
        Mutates the genotype and ensures that the mutation results in a valid phenotype.
        TODO: Do we want to ensure that the mutated genome is actually different from the main genome?
        :param max_mutation_attempts: Try to find a valid mutation up to this many times.
        :return: True if mutation occured, false otherwise.
        """
        old_genotype = self.genotype
        for i in range(max_mutation_attempts):
            self.genotype = deepcopy(old_genotype)
            self.genotype.mutate(max_mutation_attempts=max_mutation_attempts)
            self.genotype.express()
            if self.is_valid():
                return True
        return False

    def is_valid(self):
        """Ensures a randomly generated phenotype is valid (checked before adding individual to a population).

        Returns
        -------
        is_valid : bool
        True if self is valid, False otherwise.

        """
        for network in self.genotype:
            for output_node_name in network.output_node_names:
                if not network.direct_encoding and np.isnan(network.graph.nodes[output_node_name]["state"]).any():
                    return False
                elif network.direct_encoding and np.isnan(network.values).any():
                    return False
        return True


def make_material_tree(this_softbot, *args, **kwargs):
    mapping = this_softbot.to_phenotype_mapping
    material = mapping["material"]

    if material["dependency_order"] is not None:
        for dependency_name in material["dependency_order"]:
            for network in this_softbot:
                if dependency_name in network.graph.nodes():
                    mapping.dependencies[dependency_name]["state"] = network.graph.node[dependency_name]["state"] > 0

    if material["dependency_order"] is not None:
        for dependency_name in reversed(material["dependency_order"]):
            if mapping.dependencies[dependency_name]["material_if_true"] is not None:
                material["state"][mapping.get_dependency(dependency_name, True)] = \
                    mapping.dependencies[dependency_name]["material_if_true"]

            if mapping.dependencies[dependency_name]["material_if_false"] is not None:
                material["state"][mapping.get_dependency(dependency_name, False)] = \
                    mapping.dependencies[dependency_name]["material_if_false"]
    return make_one_shape_only(material["state"]) * material["state"]


def make_one_shape_only(output_state, mask=None):
    """Find the largest continuous arrangement of True elements after applying boolean mask.

    Avoids multiple disconnected softbots in simulation counted as a single individual.

    Parameters
    ----------
    output_state : numpy.ndarray
        Network output

    mask : bool mask
        Threshold function applied to output_state

    Returns
    -------
    part_of_ind : bool
        True if component of individual

    """
    if mask is None:
        def mask(u): return np.greater(u, 0)

    # print output_state
    # sys.exit(0)

    one_shape = np.zeros(output_state.shape, dtype=np.int32)

    if np.sum(mask(output_state)) < 2:
        one_shape[np.where(mask(output_state))] = 1
        return one_shape

    else:
        not_yet_checked = []
        for x in range(output_state.shape[0]):
            for y in range(output_state.shape[1]):
                for z in range(output_state.shape[2]):
                    not_yet_checked.append((x, y, z))

        largest_shape = []
        queue_to_check = []
        while len(not_yet_checked) > len(largest_shape):
            queue_to_check.append(not_yet_checked.pop(0))
            this_shape = []
            if mask(output_state[queue_to_check[0]]):
                this_shape.append(queue_to_check[0])

            while len(queue_to_check) > 0:
                this_voxel = queue_to_check.pop(0)
                x = this_voxel[0]
                y = this_voxel[1]
                z = this_voxel[2]
                for neighbor in [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)]:
                    if neighbor in not_yet_checked:
                        not_yet_checked.remove(neighbor)
                        if mask(output_state[neighbor]):
                            queue_to_check.append(neighbor)
                            this_shape.append(neighbor)

            if len(this_shape) > len(largest_shape):
                largest_shape = this_shape

        for loc in largest_shape:
            one_shape[loc] = 1

        return one_shape
