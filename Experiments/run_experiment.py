from ray.rllib.models import ModelCatalog

from LOKI.LOKI import LOKI
from PPO.PPO_CfC import initiate_PPO_CfC, ConvCfCModel
from Expert_policy.get_expert_policy import Expert_policy

# Import libraries for communication with the ABB IRB 120 robot arm
from ABB_robot_arm.ABB_robot_arm import initiate_ABB_robot_arm

def main():
    # Get the ABB robot arm
    ABB_robot_arm = initiate_ABB_robot_arm()

    # Get the model
    ModelCatalog.register_custom_model("ConvCfCModel", ConvCfCModel)

    # Inititate PP0_CfC
    algo = initiate_PPO_CfC()

    # Get the expert policy
    expert_policy = Expert_policy(model=algo)

    
    # Get the LOKI algorithm
    LOKI_algorithm = LOKI(model=algo, Nmax=50, d=3)

    # Run the LOKI algorithm
    LOKI_algorithm.run(expert_policy=expert_policy)
