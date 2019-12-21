import numpy as np
from lxml import etree


class Env(object):
    """Container for VoxCad environment parameters."""

    def __init__(self, frequency=4.0, gravity_enabled=1, grav_acc=-9.81, density=1e+006, temp_enabled=1,
                 floor_enabled=1, floor_slope=0.0, lattice_dimension=0.01, sticky_floor=0, time_between_traces=0, save_passive_data=False,
                 actuation_variance=0, temp_amp=39, temp_base=25, vary_temp_enabled=1, growth_amp=0, growth_speed_limit=0,
                 greedy_growth=False, greedy_threshold=0, squeeze_rate=0, constant_squeeze=False, squeeze_start=0.5,
                 squeeze_end=2, num_hurdles=0, space_between_hurdles=3, hurdle_height=1, hurdle_stop=np.inf,
                 circular_hurdles=False, tunnel_width=8, forward_hurdles_only=True, wall_height=3, back_stop=False,
                 fence=False, debris=False, debris_size=0, debris_start=-np.inf, biped=False, biped_leg_proportion=0.6,
                 needle_position=0, falling_prohibited=False, passive_body_only=False,
                 tilt_vectors_updates_per_cycle=0, regeneration_model_updates_per_cycle=0,
                 num_regeneration_model_synapses=0, regeneration_model_input_bias=1, num_hidden_regeneration_neurons=0,
                 forward_model_updates_per_cycle=0, controller_updates_per_cycle=0, num_forward_model_synapses=0,
                 num_controller_synapses=0, signaling_updates_per_cycle=0, depolarizations_per_cycle=5,
                 repolarizations_per_cycle=1, lightsource_xyz=None, fluid_environment=False,
                 ice_rink_environment=False, aggregate_drag_coefficient=0.0, dynamic_friction=0.5, static_friction=1,
                 contraction_only=False, expansion_only=False, BioElectricsEnabled=False, cilia_strength=0.0):

        self.sub_groups = ["Fixed_Regions", "Forced_Regions", "Gravity", "Thermal"]


        self.frequency = frequency
        self.gravity_enabled = gravity_enabled
        self.grav_acc = grav_acc
        self.density = density
        self.floor_enabled = floor_enabled
        self.temp_enabled = temp_enabled
        self.vary_temp_enabled = vary_temp_enabled
        self.temp_base = temp_base
        self.floor_slope = floor_slope

        self.sticky_floor = sticky_floor
        self.time_between_traces = time_between_traces
        self.save_passive_data = save_passive_data
        self.actuation_variance = actuation_variance
        self.temp_amp = temp_amp
        self.growth_amp = growth_amp
        self.growth_speed_limit = growth_speed_limit
        self.greedy_growth = greedy_growth
        self.greedy_threshold = greedy_threshold

        self.num_hurdles = num_hurdles
        self.space_between_hurdles = space_between_hurdles
        self.hurdle_height = -1
        if num_hurdles > 0:
            self.hurdle_height = hurdle_height
        self.circular_hurdles = circular_hurdles
        self.forward_hurdles_only = forward_hurdles_only
        self.hurdle_stop = hurdle_stop
        self.wall_height = wall_height
        self.back_stop = back_stop
        self.fence = fence
        self.debris = debris
        self.debris_size = debris_size
        self.debris_start = debris_start
        self.tunnel_width = tunnel_width
        self.squeeze_rate = squeeze_rate
        self.constant_squeeze = constant_squeeze
        self.squeeze_start = squeeze_start
        self.squeeze_end = squeeze_end

        self.needle_position = needle_position
        self.biped = biped
        self.biped_leg_proportion = biped_leg_proportion
        self.falling_prohibited = falling_prohibited
        self.passive_body_only = passive_body_only

        self.tilt_vectors_updates_per_cycle = tilt_vectors_updates_per_cycle
        self.regeneration_model_updates_per_cycle = regeneration_model_updates_per_cycle
        self.forward_model_updates_per_cycle = forward_model_updates_per_cycle
        self.controller_updates_per_cycle = controller_updates_per_cycle

        self.num_hidden_regeneration_neurons = num_hidden_regeneration_neurons
        self.regeneration_model_input_bias = regeneration_model_input_bias

        self.num_regeneration_model_synapses = num_regeneration_model_synapses
        self.num_forward_model_synapses = num_forward_model_synapses
        self.num_controller_synapses = num_controller_synapses

        self.signaling_updates_per_cycle = signaling_updates_per_cycle
        self.depolarizations_per_cycle = depolarizations_per_cycle
        self.repolarizations_per_cycle = repolarizations_per_cycle

        self.lightsource_xyz = lightsource_xyz
        if lightsource_xyz is None:
            self.lightsource_xyz = [0, 0, 0]

        self.fluid_environment = fluid_environment
        self.ice_rink_environment = ice_rink_environment
        self.cilia_strength = cilia_strength
        self.BioElectricsEnabled = BioElectricsEnabled
        self.aggregate_drag_coefficient = aggregate_drag_coefficient

        self.dynamic_friction = dynamic_friction
        self.static_friction = static_friction

        self.contraction_only = contraction_only
        self.expansion_only = expansion_only

    def write_to_xml(self, root, run_dir, individual, **kwargs):
        body_xlim = (0, individual.get_body_xlim())
        body_ylim = (0, individual.get_body_ylim())
        body_zlim = ((self.hurdle_height + 1), individual.get_body_zlim() + (self.hurdle_height + 1))

        padding = self.num_hurdles * (self.space_between_hurdles + 1)
        x_pad = [padding, padding]
        y_pad = [padding, padding]

        if not self.circular_hurdles and self.num_hurdles > 0:
            if self.num_hurdles == 1:  # single hurdle
                y_pad = x_pad = [self.space_between_hurdles / 2 + 1, self.space_between_hurdles / 2 + 1]
            else:  # tunnel
                x_pad = [self.tunnel_width / 2, self.tunnel_width / 2]
                y_pad[0] = max(self.space_between_hurdles, body_ylim[1] - 1) + 1 - body_ylim[1] + body_ylim[0]

        if self.forward_hurdles_only and self.num_hurdles > 0:  # ring
            y_pad[0] = body_ylim[1]

        if self.needle_position > 0:
            y_pad = x_pad = [0, self.needle_position]

        workspace_xlim = (-x_pad[0], body_xlim[1] + x_pad[1])
        workspace_ylim = (-y_pad[0], body_ylim[1] + y_pad[1])
        workspace_zlim = (0, max(self.wall_height, body_zlim[1]))

        length_workspace_xyz = (float(workspace_xlim[1] - workspace_xlim[0]),
                                float(workspace_ylim[1] - workspace_ylim[0]),
                                float(workspace_zlim[1] - workspace_zlim[0]))

        fixed_regions_dict = {key: {"X": {}, "Y": {}, "dX": {}, "dY": {}} for key in range(4)}

        fixed_regions_dict[0] = {"X": 0, "dX": (x_pad[0] - 1) / length_workspace_xyz[0]}

        fixed_regions_dict[1] = {"X": (body_xlim[1] - body_xlim[0] + x_pad[0] + 1) / length_workspace_xyz[0],
                                 "dX": 1 - (body_xlim[1] - body_xlim[0] + x_pad[0] + 1) / length_workspace_xyz[0]}

        fixed_regions_dict[2] = {"Y": 0, "dY": (y_pad[0] - 1) / length_workspace_xyz[1]}

        fixed_regions_dict[3] = {"Y": (body_ylim[1] - body_ylim[0] + y_pad[0] + 1) / length_workspace_xyz[1],
                                 "dY": 1 - (body_ylim[1] - body_ylim[0] + y_pad[0] + 1) / length_workspace_xyz[1]}
        env_root = etree.SubElement(root, "Environment")

        if self.num_hurdles > 0:
            boundary_conditions = etree.SubElement(env_root, "Boundary_Conditions")
            etree.SubElement(boundary_conditions, "NumBCs").text = 5

            f_region = etree.SubElement(boundary_conditions, "FRegion")
            etree.SubElement(f_region, "PrimType").text = "0"
            etree.SubElement(f_region, "X").text = str(fixed_regions_dict[0]["X"])
            etree.SubElement(f_region, "Y").text = "0"
            etree.SubElement(f_region, "Z").text = "0"
            etree.SubElement(f_region, "dX").text = str(fixed_regions_dict[0]["dX"])
            etree.SubElement(f_region, "dY").text = "1"
            etree.SubElement(f_region, "dZ").text = "1"
            etree.SubElement(f_region, "Radius").text = "0"
            etree.SubElement(f_region, "R").text = "0.4"
            etree.SubElement(f_region, "G").text = "0.6"
            etree.SubElement(f_region, "B").text = "0.4"
            etree.SubElement(f_region, "alpha").text = "1"
            etree.SubElement(f_region, "DofFixed").text = "63"
            etree.SubElement(f_region, "ForceX").text = "0"
            etree.SubElement(f_region, "ForceY").text = "0"
            etree.SubElement(f_region, "ForceZ").text = "0"
            etree.SubElement(f_region, "TorqueX").text = "0"
            etree.SubElement(f_region, "TorqueY").text = "0"
            etree.SubElement(f_region, "TorqueZ").text = "0"
            etree.SubElement(f_region, "DisplaceX").text = "0"
            etree.SubElement(f_region, "DisplaceY").text = "0"
            etree.SubElement(f_region, "DisplaceZ").text = "0"
            etree.SubElement(f_region, "AngDisplaceX").text = "0"
            etree.SubElement(f_region, "AngDisplaceY").text = "0"
            etree.SubElement(f_region, "AngDisplaceZ").text = "0"

            f_region = etree.SubElement(boundary_conditions, "FRegion")
            etree.SubElement(f_region, "PrimType").text = "0"
            etree.SubElement(f_region, "X").text = str(fixed_regions_dict[1]["X"])
            etree.SubElement(f_region, "Y").text = "0"
            etree.SubElement(f_region, "Z").text = "0"
            etree.SubElement(f_region, "dX").text = str(fixed_regions_dict[1]["dX"])
            etree.SubElement(f_region, "dY").text = "1"
            etree.SubElement(f_region, "dZ").text = "1"
            etree.SubElement(f_region, "Radius").text = "0"
            etree.SubElement(f_region, "R").text = "0.4"
            etree.SubElement(f_region, "G").text = "0.6"
            etree.SubElement(f_region, "B").text = "0.4"
            etree.SubElement(f_region, "alpha").text = "1"
            etree.SubElement(f_region, "DofFixed").text = "63"
            etree.SubElement(f_region, "ForceX").text = "0"
            etree.SubElement(f_region, "ForceY").text = "0"
            etree.SubElement(f_region, "ForceZ").text = "0"
            etree.SubElement(f_region, "TorqueX").text = "0"
            etree.SubElement(f_region, "TorqueY").text = "0"
            etree.SubElement(f_region, "TorqueZ").text = "0"
            etree.SubElement(f_region, "DisplaceX").text = "0"
            etree.SubElement(f_region, "DisplaceY").text = "0"
            etree.SubElement(f_region, "DisplaceZ").text = "0"
            etree.SubElement(f_region, "AngDisplaceX").text = "0"
            etree.SubElement(f_region, "AngDisplaceY").text = "0"
            etree.SubElement(f_region, "AngDisplaceZ").text = "0"

            f_region = etree.SubElement(boundary_conditions, "FRegion")
            etree.SubElement(f_region, "PrimType").text = "0"
            etree.SubElement(f_region, "X").text = "0"
            etree.SubElement(f_region, "Y").text = str(fixed_regions_dict[2]["Y"])
            etree.SubElement(f_region, "Z").text = "0"
            etree.SubElement(f_region, "dX").text = "1"
            etree.SubElement(f_region, "dY").text = str(fixed_regions_dict[2]["dY"])
            etree.SubElement(f_region, "dZ").text = "1"
            etree.SubElement(f_region, "Radius").text = "0"
            etree.SubElement(f_region, "R").text = "0.4"
            etree.SubElement(f_region, "G").text = "0.6"
            etree.SubElement(f_region, "B").text = "0.4"
            etree.SubElement(f_region, "alpha").text = "1"
            etree.SubElement(f_region, "DofFixed").text = "63"
            etree.SubElement(f_region, "ForceX").text = "0"
            etree.SubElement(f_region, "ForceY").text = "0"
            etree.SubElement(f_region, "ForceZ").text = "0"
            etree.SubElement(f_region, "TorqueX").text = "0"
            etree.SubElement(f_region, "TorqueY").text = "0"
            etree.SubElement(f_region, "TorqueZ").text = "0"
            etree.SubElement(f_region, "DisplaceX").text = "0"
            etree.SubElement(f_region, "DisplaceY").text = "0"
            etree.SubElement(f_region, "DisplaceZ").text = "0"
            etree.SubElement(f_region, "AngDisplaceX").text = "0"
            etree.SubElement(f_region, "AngDisplaceY").text = "0"
            etree.SubElement(f_region, "AngDisplaceZ").text = "0"

            f_region = etree.SubElement(boundary_conditions, "FRegion")
            etree.SubElement(f_region, "PrimType").text = "0"
            etree.SubElement(f_region, "X").text = "0"
            etree.SubElement(f_region, "Y").text = str(fixed_regions_dict[3]["Y"])
            etree.SubElement(f_region, "Z").text = "0"
            etree.SubElement(f_region, "dX").text = "1"
            etree.SubElement(f_region, "dY").text = str(fixed_regions_dict[3]["dY"])
            etree.SubElement(f_region, "dZ").text = "1"
            etree.SubElement(f_region, "Radius").text = "0"
            etree.SubElement(f_region, "R").text = "0.4"
            etree.SubElement(f_region, "G").text = "0.6"
            etree.SubElement(f_region, "B").text = "0.4"
            etree.SubElement(f_region, "alpha").text = "1"
            etree.SubElement(f_region, "DofFixed").text = "63"
            etree.SubElement(f_region, "ForceX").text = "0"
            etree.SubElement(f_region, "ForceY").text = "0"
            etree.SubElement(f_region, "ForceZ").text = "0"
            etree.SubElement(f_region, "TorqueX").text = "0"
            etree.SubElement(f_region, "TorqueY").text = "0"
            etree.SubElement(f_region, "TorqueZ").text = "0"
            etree.SubElement(f_region, "DisplaceX").text = "0"
            etree.SubElement(f_region, "DisplaceY").text = "0"
            etree.SubElement(f_region, "DisplaceZ").text = "0"
            etree.SubElement(f_region, "AngDisplaceX").text = "0"
            etree.SubElement(f_region, "AngDisplaceY").text = "0"
            etree.SubElement(f_region, "AngDisplaceZ").text = "0"

            f_region = etree.SubElement(boundary_conditions, "FRegion")
            etree.SubElement(f_region, "PrimType").text = "0"
            etree.SubElement(f_region, "X").text = "0"
            etree.SubElement(f_region, "Y").text = "0"
            etree.SubElement(f_region, "Z").text = "0"
            etree.SubElement(f_region, "dX").text = "1"
            etree.SubElement(f_region, "dY").text = "1"
            etree.SubElement(f_region, "dZ").text = str(self.hurdle_height / length_workspace_xyz[2])
            etree.SubElement(f_region, "Radius").text = "0"
            etree.SubElement(f_region, "R").text = "0.4"
            etree.SubElement(f_region, "G").text = "0.6"
            etree.SubElement(f_region, "B").text = "0.4"
            etree.SubElement(f_region, "alpha").text = "1"
            etree.SubElement(f_region, "DofFixed").text = "63"
            etree.SubElement(f_region, "ForceX").text = "0"
            etree.SubElement(f_region, "ForceY").text = "0"
            etree.SubElement(f_region, "ForceZ").text = "0"
            etree.SubElement(f_region, "TorqueX").text = "0"
            etree.SubElement(f_region, "TorqueY").text = "0"
            etree.SubElement(f_region, "TorqueZ").text = "0"
            etree.SubElement(f_region, "DisplaceX").text = "0"
            etree.SubElement(f_region, "DisplaceY").text = "0"
            etree.SubElement(f_region, "DisplaceZ").text = "0"
            etree.SubElement(f_region, "AngDisplaceX").text = "0"
            etree.SubElement(f_region, "AngDisplaceY").text = "0"
            etree.SubElement(f_region, "AngDisplaceZ").text = "0"
        else:
            f_regions = etree.SubElement(env_root, "Fixed_Regions")
            etree.SubElement(f_regions, "NumFixed").text = "0"
            f_regions = etree.SubElement(env_root, "Forced_Regions")
            etree.SubElement(f_regions, "NumForced").text = "0"

        gravity = etree.SubElement(env_root, "Gravity")
        etree.SubElement(gravity, "GravEnabled").text = str(int(self.gravity_enabled))
        etree.SubElement(gravity, "GravAcc").text = str(self.grav_acc)
        etree.SubElement(gravity, "FloorEnabled").text = str(int(self.floor_enabled))
        etree.SubElement(gravity, "FloorSlope").text = str(self.floor_slope)

        thermal = etree.SubElement(env_root, "Thermal")
        etree.SubElement(thermal, "TempEnabled").text = str(int(self.temp_enabled))
        etree.SubElement(thermal, "TempAmp").text = str(self.temp_amp)
        etree.SubElement(thermal, "TempBase").text = str(self.temp_base)
        etree.SubElement(thermal, "VaryTempEnabled").text = str(int(self.vary_temp_enabled))
        etree.SubElement(thermal, "TempPeriod").text = str(1.0 / self.frequency)

        light_source = etree.SubElement(env_root, "LightSource")
        etree.SubElement(light_source, "X").text = str(self.lightsource_xyz[0])
        etree.SubElement(light_source, "Y").text = str(self.lightsource_xyz[1])
        etree.SubElement(light_source, "Z").text = str(self.lightsource_xyz[2])

        regen_model = etree.SubElement(env_root, "RegenerationModel")
        etree.SubElement(regen_model, "TiltVectorsUpdatesPerTempCycle").text = str(self.tilt_vectors_updates_per_cycle)
        etree.SubElement(regen_model, "RegenerationModelUpdatesPerTempCycle").text = str(self.regeneration_model_updates_per_cycle)
        etree.SubElement(regen_model, "NumHiddenRegenerationNeurons").text = str(self.num_hidden_regeneration_neurons)
        etree.SubElement(regen_model, "RegenerationModelInputBias").text = str(int(self.regeneration_model_input_bias))

        forward_model = etree.SubElement(env_root, "ForwardModel")
        etree.SubElement(forward_model, "ForwardModelUpdatesPerTempCycle").text = str(self.forward_model_updates_per_cycle)

        controller = etree.SubElement(env_root, "Controller")
        etree.SubElement(controller, "ControllerUpdatesPerTempCycle").text = str(self.controller_updates_per_cycle)

        electric_signaling = etree.SubElement(env_root, "Signaling")
        etree.SubElement(electric_signaling, "SignalingUpdatesPerTempCycle").text = str(self.signaling_updates_per_cycle)
        etree.SubElement(electric_signaling, "DepolarizationsPerTempCycle").text = str(self.depolarizations_per_cycle)
        etree.SubElement(electric_signaling, "RepolarizationsPerTempCycle").text = str(self.repolarizations_per_cycle)

        etree.SubElement(env_root, "GrowthAmplitude").text = str(self.growth_amp)
        etree.SubElement(env_root, "GrowthSpeedLimit").text = str(self.growth_speed_limit)
        etree.SubElement(env_root, "GreedyGrowth").text = str(int(self.greedy_growth))
        etree.SubElement(env_root, "GreedyThreshold").text = str(self.greedy_threshold)
        etree.SubElement(env_root, "TimeBetweenTraces").text = str(self.time_between_traces)
        etree.SubElement(env_root, "SavePassiveData").text = str(int(self.save_passive_data))
        etree.SubElement(env_root, "StickyFloor").text = str(self.sticky_floor)
        etree.SubElement(env_root, "NeedleInHaystack").text = str(int(self.needle_position > 0))
        etree.SubElement(env_root, "ContractionOnly").text = str(int(self.contraction_only))
        etree.SubElement(env_root, "ExpansionOnly").text = str(int(self.expansion_only))
        etree.SubElement(env_root, "FluidEnvironment").text = str(int(self.fluid_environment))
        etree.SubElement(env_root, "AggregateDragCoefficient").text = str(int(self.aggregate_drag_coefficient))
        etree.SubElement(env_root, "IceRinkEnvironment").text = str(int(self.ice_rink_environment))
        etree.SubElement(env_root, "BioElectricsEnabled").text = str(int(self.BioElectricsEnabled))

        return env_root


class VXC_Wrapper(object):
    def __init__(self, lattice_dimension=0.01):
        self.lattice_dimension = lattice_dimension

    def write_to_xml(self, root, material_pallette, robot, **kwargs):
        VXC_root = etree.SubElement(root, "VXC", Version="0.93")

        lattice = etree.SubElement(VXC_root, "Lattice")
        etree.SubElement(lattice, "Lattice_Dim").text = str(self.lattice_dimension)
        etree.SubElement(lattice, "X_Dim_Adj").text = "1"
        etree.SubElement(lattice, "Y_Dim_Adj").text = "1"
        etree.SubElement(lattice, "Z_Dim_Adj").text = "1"
        etree.SubElement(lattice, "X_Line_Offset").text = "0"
        etree.SubElement(lattice, "Y_Line_Offset").text = "0"
        etree.SubElement(lattice, "X_Layer_Offset").text = "0"
        etree.SubElement(lattice, "Y_Layer_Offset").text = "0"

        voxel = etree.SubElement(VXC_root, "Voxel")
        etree.SubElement(voxel, "Vox_Name").text = "BOX"
        etree.SubElement(voxel, "X_Squeeze").text = "1"
        etree.SubElement(voxel, "Y_Squeeze").text = "1"
        etree.SubElement(voxel, "Z_Squeeze").text = "1"

        palette = etree.SubElement(VXC_root, "Palette")
        for material in material_pallette:
            material.write_to_xml(palette)

        robot.write_to_xml(VXC_root)

        return VXC_root
