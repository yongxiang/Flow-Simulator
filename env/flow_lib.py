### Script for gym import ###
import json

from flow.envs import TrafficLightGridPOEnv
from flow.networks import TrafficLightGridNetwork
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter

# time horizon of a single rollout
# too short horizon will cause error
HORIZON = 2000

def gen_edges(col_num, row_num):
    """
    Generate the names of the outer edges in the traffic light grid network.

    Parameters
    ----------
    col_num : int
        number of columns in the traffic light grid
    row_num : int
        number of rows in the traffic light grid

    Returns
    -------
    list of str
        names of all the outer edges
    """
    edges = []
    for i in range(col_num):
        edges += ['left' + str(row_num) + '_' + str(i)]
        edges += ['right' + '0' + '_' + str(i)]

    # build the left and then the right edges
    for i in range(row_num):
        edges += ['bot' + str(i) + '_' + '0']
        edges += ['top' + str(i) + '_' + str(col_num)]

    return edges


def get_flow_params(col_num, row_num, additional_net_params):
    """
    Define the network and initial params in the presence of inflows.

    Parameters
    ----------
    col_num : int
        number of columns in the traffic light grid
    row_num : int
        number of rows in the traffic light grid
    additional_net_params : dict
        network-specific parameters that are unique to the traffic light grid

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the network
    """
    prob = {8: 0.6, 9: 0.3, 6:0.2, 7: 0.3, 10: 0.5, 11:0.15}
    
    initial = InitialConfig(
        spacing='custom', lanes_distribution=float('inf'), shuffle=True)

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        inflow.add(
            veh_type='idm',
            edge=outer_edges[i],
            probability=prob[i] if i in prob else 0.1,
            depart_lane='free',
            depart_speed=10)

    net = NetParams(
        inflows=inflow,
        additional_params=additional_net_params)

    return initial, net


def get_non_flow_params(enter_speed, add_net_params):
    """
    Define the network and initial params in the absence of inflows.

    Note that when a vehicle leaves a network in this case, it is immediately
    returns to the start of the row/column it was traversing, and in the same
    direction as it was before.

    Parameters
    ----------
    enter_speed : float
        initial speed of vehicles as they enter the network.
    add_net_params: dict
        additional network-specific parameters (unique to the traffic light grid)

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the network
    """
    additional_init_params = {'enter_speed': enter_speed}
    initial = InitialConfig(
        spacing='custom', additional_params=additional_init_params)
    net = NetParams(additional_params=add_net_params)

    return initial, net


V_ENTER = 15
INNER_LENGTH = 100
LONG_LENGTH = 50
SHORT_LENGTH = 100
N_ROWS = 3
N_COLUMNS = 3
NUM_CARS_LEFT = 1
NUM_CARS_RIGHT = 1
NUM_CARS_TOP = 1
NUM_CARS_BOT = 1
tot_cars = (NUM_CARS_LEFT + NUM_CARS_RIGHT) * N_COLUMNS \
           + (NUM_CARS_BOT + NUM_CARS_TOP) * N_ROWS

grid_array = {
    "short_length": SHORT_LENGTH,
    "inner_length": INNER_LENGTH,
    "long_length": LONG_LENGTH,
    "row_num": N_ROWS,
    "col_num": N_COLUMNS,
    "cars_left": NUM_CARS_LEFT,
    "cars_right": NUM_CARS_RIGHT,
    "cars_top": NUM_CARS_TOP,
    "cars_bot": NUM_CARS_BOT
}

additional_env_params = {
        'target_velocity': 50,
        'switch_time': 3.0,
        'num_observed': 3,
        'discrete': False,
        'tl_type': 'controlled'
    }

additional_net_params = {
    'speed_limit': 35,
    'grid_array': grid_array,
    'horizontal_lanes': 1,
    'vertical_lanes': 1
}

vehicles = VehicleParams()
vehicles.add(
    veh_id='idm',
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams(
        minGap=2.5,
        decel=7.5,  # avoid collisions at emergency stops
        max_speed=V_ENTER,
        speed_mode="all_checks",
    ),
    routing_controller=(GridRouter, {}),
    num_vehicles=tot_cars)

flow_params = dict(
    # name of the experiment
    exp_tag='traffic_light_grid',

    # name of the flow environment the experiment is running on
    env_name=TrafficLightGridPOEnv,

    # name of the network class the experiment is running on
    network=TrafficLightGridNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=1,
        render=False,
        restart_instance=False
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params=additional_env_params,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component). This is
    # filled in by the flow_env method below.
    net=None,

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig). This is filled in by the
    # flow_env method below.
    initial=None,
)


def flow_env(use_inflows=False, render=True):
    global flow_params
    flow_params['sim'] = SumoParams(
                            sim_step=0.1,
                            render=render,
                            restart_instance=True # for long horizon and visualization
                            )
    print(flow_params)
    """
    Parameters
    ----------
    use_inflows : bool, optional
        set to True if you would like to run the experiment with inflows of
        vehicles from the edges, and False otherwise

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    # collect the initialization and network-specific parameters based on the
    # choice to use inflows or not
    if use_inflows:
        initial_config, net_params = get_flow_params(
            col_num=N_COLUMNS,
            row_num=N_ROWS,
            additional_net_params=additional_net_params)
    else:
        initial_config, net_params = get_non_flow_params(
            enter_speed=V_ENTER,
            add_net_params=additional_net_params)

    # add the new parameters to flow_params
    flow_params['initial'] = initial_config
    flow_params['net'] = net_params
    
    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)

    create_env, gym_name = make_create_env(params=flow_params, version=0)
    env = create_env()
    return env, gym_name


if __name__ == '__main__':
    _, _ = flow_env()
