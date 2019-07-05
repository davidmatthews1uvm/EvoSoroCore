import random
from abc import ABCMeta, abstractmethod

import networkx as nx
import numpy as np

from NetworkUtils import neg_abs, neg_sqrt_abs, neg_square, normalize, sigmoid, sqrt_abs
from NetworkUtils import vox_xyz_from_id


class Network(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def mutate(self, *args, **kwargs): raise NotImplementedError

    @abstractmethod
    def express(self, xyz_size, *args, **kwargs): raise NotImplementedError


class CPPN(Network):
    input_node_names = ['x', 'y', 'z', 'd', 'b']
    activation_functions = [np.sin, np.abs, neg_abs, np.square, neg_square, sqrt_abs, neg_sqrt_abs]

    def __init__(self, output_node_names):
        self.output_node_names = output_node_names
        self.graph = nx.DiGraph()
        self.set_minimal_graph()
        self.mutate()

    def __str__(self):
        return "CPPN consisting of:\nNodes: " + str(self.graph.nodes(data=True)) + "\nEdges: " + str(self.graph.edges(data=True))

    def set_minimal_graph(self):
        for name in self.input_node_names:
            self.graph.add_node(name, type="input", function=None)
        for name in self.output_node_names:
            self.graph.add_node(name, type="output", function=sigmoid)

        for input_node in (node[0] for node in self.graph.nodes(data=True) if node[1]["type"] == "input"):
            for output_node in (node[0] for node in self.graph.nodes(data=True) if node[1]["type"] == "output"):
                self.graph.add_edge(input_node, output_node, weight=0.0)

    def express(self, xyz_size=(1, 1, 1), *args, **kwargs):
        for name in self.graph.nodes():
            self.graph.nodes[name]["evaluated"] = False
        self.set_input_node_states(xyz_size)

        new_shape = [len(self.output_node_names)]
        new_shape += list(xyz_size)
        vals_to_return = np.zeros(shape=new_shape)

        for n, name in enumerate(self.output_node_names):
            self.graph.nodes[name]["state"] = np.zeros(xyz_size)  # clear old output state.
            node_state = self.calc_node_state(name, xyz_size)  # compute new node state (recursively)
            vals_to_return[n] = node_state
            self.graph.nodes[name]["state"] = node_state

        return vals_to_return

    def set_input_node_states(self, orig_size_xyz):
        input_x = np.zeros(orig_size_xyz)
        input_y = np.zeros(orig_size_xyz)
        input_z = np.zeros(orig_size_xyz)
        for x in range(orig_size_xyz[0]):
            for y in range(orig_size_xyz[1]):
                for z in range(orig_size_xyz[2]):
                    input_x[x, y, z] = x
                    input_y[x, y, z] = y
                    input_z[x, y, z] = z

        input_x = normalize(input_x)
        input_y = normalize(input_y)
        input_z = normalize(input_z)
        input_d = normalize(np.power(np.power(input_x, 2) + np.power(input_y, 2) + np.power(input_z, 2), 0.5))
        input_b = np.ones(orig_size_xyz)

        for name in self.graph.nodes():
            if name == "x":
                self.graph.nodes[name]["state"] = input_x
                self.graph.nodes[name]["evaluated"] = True
            if name == "y":
                self.graph.nodes[name]["state"] = input_y
                self.graph.nodes[name]["evaluated"] = True
            if name == "z":
                self.graph.nodes[name]["state"] = input_z
                self.graph.nodes[name]["evaluated"] = True
            if name == "d":
                self.graph.nodes[name]["state"] = input_d
                self.graph.nodes[name]["evaluated"] = True
            if name == "b":
                self.graph.nodes[name]["state"] = input_b
                self.graph.nodes[name]["evaluated"] = True

    def calc_node_state(self, node_name, xyz_size):
        """Propagate input values through the network"""
        if self.graph.nodes[node_name]["evaluated"]:
            return self.graph.nodes[node_name]["state"]

        self.graph.nodes[node_name]["evaluated"] = True
        input_edges = self.graph.in_edges(nbunch=[node_name])
        new_state = np.zeros(xyz_size)

        for edge in input_edges:
            node1, node2 = edge
            new_state += self.calc_node_state(node1, xyz_size) * self.graph.edges[node1, node2]["weight"]

        self.graph.nodes[node_name]["state"] = new_state

        return new_state

    def mutate(self, num_random_node_adds=10, num_random_node_removals=0, num_random_link_adds=10,
               num_random_link_removals=5, num_random_activation_functions=100, num_random_weight_changes=100):

        variation_degree = None
        variation_type = None

        for _ in range(num_random_node_adds):
            variation_degree = self.add_node()
            variation_type = "add_node"

        for _ in range(num_random_node_removals):
            variation_degree = self.remove_node()
            variation_type = "remove_node"

        for _ in range(num_random_link_adds):
            variation_degree = self.add_link()
            variation_type = "add_link"

        for _ in range(num_random_link_removals):
            variation_degree = self.remove_link()
            variation_type = "remove_link"

        for _ in range(num_random_activation_functions):
            variation_degree = self.mutate_function()
            variation_type = "mutate_function"

        for _ in range(num_random_weight_changes):
            variation_degree = self.mutate_weight()
            variation_type = "mutate_weight"

        self.prune_network()
        return variation_type, variation_degree

    ###############################################
    #   Mutation functions
    ###############################################

    def add_node(self):
        # choose two random nodes (between which a link could exist)
        assert len(self.graph.edges()) > 0, "Graph must have edges in order to add a node."
        node1, node2 = random.choice(list(self.graph.edges()))

        # create a new node hanging from the previous output node
        new_node_index = self.get_max_hidden_node_index()
        self.graph.add_node(new_node_index, type="hidden", function=random.choice(self.activation_functions))

        # random activation function here to solve the problem with admissible mutations in the first generations
        self.graph.add_edge(new_node_index, node2, weight=1.0)

        # if this edge already existed here, remove it
        # but use it's weight to minimize disruption when connecting to the previous input node
        if (node1, node2) in self.graph.edges():
            weight = self.graph.get_edge_data(node1, node2, default={"weight": 0})["weight"]
            self.graph.remove_edge(node1, node2)
            self.graph.add_edge(node1, new_node_index, weight=weight)
        else:
            self.graph.add_edge(node1, new_node_index, weight=1.0)
            # weight 0.0 would minimize disruption of new edge
            # but weight 1.0 should help in finding admissible mutations in the first generations
        return ""

    def remove_node(self):
        hidden_nodes = list(set(self.graph.nodes(data=False)) - set(self.input_node_names) - set(self.output_node_names))
        if len(hidden_nodes) == 0:
            return "NoHiddenNodes"
        this_node = random.choice(hidden_nodes)

        # if there are edge paths going through this node, keep them connected to minimize disruption
        incoming_edges = self.graph.in_edges(nbunch=[this_node])
        outgoing_edges = self.graph.out_edges(nbunch=[this_node])

        for incoming_edge in incoming_edges:
            for outgoing_edge in outgoing_edges:
                w = self.graph.get_edge_data(incoming_edge[0], this_node, default={"weight": 0})["weight"] * \
                    self.graph.get_edge_data(this_node, outgoing_edge[1], default={"weight": 0})["weight"]

                self.graph.add_edge(incoming_edge[0], outgoing_edge[1], weight=w)
        self.graph.remove_node(this_node)
        return ""

    def add_link(self):
        done = False
        attempt = 0
        while not done:
            done = True

            # choose two random nodes (between which a link could exist, *but doesn't*)
            node1 = random.choice(list(self.graph.nodes()))
            node2 = random.choice(list(self.graph.nodes()))
            while (not self.new_edge_is_valid(node1, node2)) and attempt < 999:
                node1 = random.choice(list(self.graph.nodes()))
                node2 = random.choice(list(self.graph.nodes()))
                attempt += 1
            if attempt > 999:  # no valid edges to add found in 1000 attempts
                done = True

            # create a link between them
            if random.random() > 0.5:
                self.graph.add_edge(node1, node2, weight=0.1)
            else:
                self.graph.add_edge(node1, node2, weight=-0.1)

            # If the link creates a cyclic graph, erase it and try again
            if self.has_cycles():
                self.graph.remove_edge(node1, node2)
                done = False
                attempt += 1
            if attempt > 999:
                done = True
        return ""

    def remove_link(self):
        if len(self.graph.edges()) == 0:
            return "NoEdges"
        this_link = random.choice(list(self.graph.edges()))
        self.graph.remove_edge(this_link[0], this_link[1])
        return ""

    def mutate_function(self):
        this_node = random.choice(list(self.graph.nodes()))
        while this_node in self.input_node_names:
            this_node = random.choice(list(self.graph.nodes()))
        old_function = self.graph.nodes()[this_node]["function"]
        while self.graph.nodes()[this_node]["function"] == old_function:
            self.graph.nodes()[this_node]["function"] = random.choice(self.activation_functions)
        return old_function.__name__ + "-to-" + self.graph.nodes(data=True)[this_node]["function"].__name__

    def mutate_weight(self, mutation_std=0.5):
        if len(self.graph.edges()) == 0:
            return "NoEdges"
        this_edge = random.choice(list(self.graph.edges()))
        node1 = this_edge[0]
        node2 = this_edge[1]
        old_weight = self.graph[node1][node2]["weight"]
        new_weight = old_weight
        while old_weight == new_weight:
            new_weight = random.gauss(old_weight, mutation_std)
            new_weight = max(-1.0, min(new_weight, 1.0))
        self.graph[node1][node2]["weight"] = new_weight
        return float(new_weight - old_weight)

    ###############################################
    #   Helper functions for mutation
    ###############################################

    def prune_network(self):
        """
        Remove erroneous nodes and edges post mutation.
        Recursively removes all such nodes so that every hidden node connects upstream to input nodes and connects downstream to output nodes.
        Removes all hidden nodes that have either no inputs or no outputs.
        """

        done = False
        while not done:
            done = True

            for node in list(self.graph.nodes()):
                in_edge_cnt = len(self.graph.in_edges(nbunch=[node]))
                node_type = self.graph.nodes[node]["type"]

                if in_edge_cnt == 0 and node_type != "input" and node_type != "output":
                    self.graph.remove_node(node)
                    done = False

            for node in list(self.graph.nodes()):
                out_edge_cnt = len(self.graph.out_edges(nbunch=[node]))
                node_type = self.graph.nodes[node]["type"]

                if out_edge_cnt == 0 and node_type != "input" and node_type != "output":
                    self.graph.remove_node(node)
                    done = False

    def has_cycles(self):
        """
        Checks if the graph is a DAG, and returns accordingly.
        :return: True if the graph is not a DAG (has cycles) False otherwise.
        """
        return not nx.is_directed_acyclic_graph(self.graph)

    def get_max_hidden_node_index(self):
        max_index = 0
        for input_node in nx.nodes(self.graph):
            if self.graph.nodes(data=True)[input_node]["type"] == "hidden" and int(input_node) >= max_index:
                max_index = input_node + 1
        return max_index

    def new_edge_is_valid(self, node1, node2):
        """
        Checks that we are permitted to create an edge between these two nodes.
        New edges must:
        * not already exist
        * not create self loops (must be a feed forward network)

        :param node1: Src node
        :param node2: Dest node
        :return: True if an edge can be added. False otherwise.
        """
        if node1 == node2:
            return False
        if node1 in self.output_node_names:
            return False
        if node2 in self.input_node_names:
            return False
        if (node2, node1) in self.graph.edges():
            return False
        if (node1, node2) in self.graph.edges():
            return False
        return True


class DirectEncoding(Network):
    def __init__(self, orig_size_xyz, lower_bound=-1, upper_bound=1, func=None, symmetric=True,
                 p=None, scale=None, start_val=None, mutate_start_val=False, allow_neutral_mutations=False,
                 sub_vox_dict=None, frozen_vox=None, patch_mode=False):

        if patch_mode:
            raise NotImplementedError("Patches are not implemented.")

        self.direct_encoding = True
        self.allow_neutral_mutations = allow_neutral_mutations
        self.size = orig_size_xyz
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if p is None:
            p = 1 / np.product(self.size, dtype='f')
        self.p = p
        self.scale = scale
        self.func = func
        self.symmetric = symmetric
        self.start_value = start_val

        self.patch_mode = patch_mode

        if sub_vox_dict is None:
            self.sub_vox_dict = dict()
        else:
            self.sub_vox_dict = sub_vox_dict

        self.mutable_vox = np.ones(shape=orig_size_xyz, dtype=bool)

        if frozen_vox is not None:
            for idx in frozen_vox:
                x, y, z = vox_xyz_from_id(idx, self.size)
                self.mutable_vox[x, y, z] = False

        if start_val is None:
            self.values = np.random.uniform(lower_bound, upper_bound, size=orig_size_xyz)
        else:
            self.values = np.ones(shape=orig_size_xyz) * start_val
            if mutate_start_val:
                self.mutate()

        self.enforce_symmetry()

        self.regulate_sub_voxels()

        if self.func is not None:
            self.values = self.func(self.values)

        self.values = np.clip(self.values, self.lower_bound, self.upper_bound)

    def __str__(self):
        return str(self.values)

    def express(self, xyz_size, *args, **kwargs):
        assert xyz_size == self.size, "Direct encoding can only generate phenotypes of same shape as internal encoding"
        return self.values

    def mutate(self, rate=None):

        if False and self.patch_mode:
            pass
            # TODO: add support for patches.
            # self.values, sub_vox_dict = add_patch(self.values)
            #
            # for parent, child in sub_vox_dict.items():
            #     self.sub_vox_dict[parent] = [child]
            #
            # return "patched", 1

        else:
            if rate is None:
                rate = self.p

            scale = self.scale
            if self.scale is None:
                scale = np.abs(1 / self.values)
                # scale = np.clip(self.values**0.5, self.start_value**0.5, self.upper_bound)
                # this was for meta mutations

            selection = np.random.random(self.size) < rate
            selection = np.logical_and(selection, self.mutable_vox)
            change = np.random.normal(scale=scale, size=self.size)
            self.values[selection] += change[selection]

            self.values = np.clip(self.values, self.lower_bound, self.upper_bound)

            self.enforce_symmetry()

            self.regulate_sub_voxels()

            if self.func is not None:
                self.values = self.func(self.values)

            return "gaussian", self.scale

    def enforce_symmetry(self):
        if self.symmetric:
            reversed_array = self.values[::-1, :, :]
            self.values[:int(self.size[0] / 2.0), :, :] = reversed_array[:int(self.size[0] / 2.0), :, :]

    def regulate_sub_voxels(self):
        if len(self.sub_vox_dict) == 0:
            return

        self.mutable_vox = np.zeros(self.size, dtype=bool)

        for parent, children in self.sub_vox_dict.items():
            px, py, pz = vox_xyz_from_id(parent, self.size)
            self.mutable_vox[px, py, pz] = True
            group_val = self.values[px, py, pz] / float(len(children))
            self.values[px, py, pz] = group_val
            for child in children:
                cx, cy, cz = vox_xyz_from_id(child, self.size)
                self.values[cx, cy, cz] = group_val
