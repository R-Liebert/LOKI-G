from PPO.PPO_CfC import initiate_PPO_CfC

"""Get the expert policy using imitation learning on the CfC neural network. """

# Classes

class Expert_policy:
    """
    Expert policy using imitation learning on the CfC neural network.
    """

    def __init__(self, model):
        """
        Initialize the expert policy.

        Input:
            model: the model to be used for the algorithm
        """
        self.model = model
        self.policy = None
        
    def train(self, expert_data):
        """
        Train the expert policy on the expert data.

        Input:
            expert_data: the expert data
        """
        self.policy = self.model.train(expert_data)

    def sample(self):
        """
        Sample from the expert policy.

        Output:
            obs: observations
            actions: actions
            rewards: rewards
            dones: dones
            infos: infos
        """
        return self.policy.sample()

    def evaluate(self):
        """
        Evaluate the expert policy.

        Output:
            reward: the reward of the expert policy
        """
        return self.policy.evaluate()

    def save(self):
        """
        Save the expert policy.
        """
        self.policy.save()

    def load(self):
        """
        Load the expert policy.
        """
        self.policy.load()
        
