############################################################


# Under construction!!!!!!!!!!


############################################################


from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf
import numpy as np
import time
from baselines.common import colorize
from mpi4py import MPI
from collections import deque
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from contextlib import contextmanager
import os.path as osp
import math

########################################



# Our Imports
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ncps.tf import CfC

import argparse
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPO




# Our CfC rnn model

class ConvCfCModel(RecurrentNetwork):
    """Example of using the Keras functional API to define a RNN model."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        cell_size=64,
    ):
        super(ConvCfCModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.cell_size = cell_size

        # Define input layers
        input_layer = tf.keras.layers.Input(
            # rllib flattens the input
            shape=(None, obs_space.shape[0] * obs_space.shape[1] * obs_space.shape[2]),
            name="inputs",
        )
        state_in_h = tf.keras.layers.Input(shape=(cell_size,), name="h")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        # Preprocess observation with a hidden layer and send to CfC
        self.conv_block = tf.keras.models.Sequential([
            tf.keras.Input(
                (obs_space.shape[0] * obs_spac.shapee[1] * obs_space.shape[2])
            ),  # batch dimension is implicit
            tf.keras.layers.Lambda(
                lambda x: tf.cast(x, tf.float32) / 255.0
            ),  # normalize input
            # unflatten the input image that has been done by rllib
            tf.keras.layers.Reshape((obs_space.shape[0], obs_space.shape[1], obs_space.shape[2])),
            tf.keras.layers.Conv2D(
                64, 5, padding="same", activation="relu", strides=2
            ),
            tf.keras.layers.Conv2D(
                128, 5, padding="same", activation="relu", strides=2
            ),
            tf.keras.layers.Conv2D(
                128, 5, padding="same", activation="relu", strides=2
            ),
            tf.keras.layers.Conv2D(
                256, 5, padding="same", activation="relu", strides=2
            ),
            tf.keras.layers.GlobalAveragePooling2D(),
        ])
        self.td_conv = tf.keras.layers.TimeDistributed(self.conv_block)

        dense1 = self.td_conv(input_layer)
        cfc_out, state_h = CfC(
            cell_size, return_sequences=True, return_state=True, name="cfc"
        )(
            inputs=dense1,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h],
        )

        # Postprocess CfC output with another hidden layer and compute values
        logits = tf.keras.layers.Dense(
            self.num_outputs, activation=tf.keras.activations.linear, name="logits"
        )(cfc_out)
        values = tf.keras.layers.Dense(1, activation=None, name="values")(cfc_out)

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h],
            outputs=[logits, values, state_h],
        )
        self.rnn_model.summary()

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h = self.rnn_model([inputs, seq_lens] + state)
        return model_out, [h]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self):
        return tf.reshape(self._value_out, [-1])

# Initiate RLLIB PPO with ConvCfCModel
def initiate_PPO_CfC(self, args):
    config = {
        "env": "atari_env",
        "preprocessor_pref": None,
        "gamma": 0.99,
        "num_gpus": 1,
        "num_workers": 16,
        "num_envs_per_worker": 4,
        "create_env_on_driver": True,
        "lambda": 0.95,
        "kl_coeff": 0.5, # Robin: Has to be changed to var as it changes between IL and RL
        "clip_rewards": True,
        "clip_param": 0.1,
        "vf_clip_param": 10.0,
        "entropy_coeff": 0.01,
        "rollout_fragment_length": 100,
        "sgd_minibatch_size": 500,
        "num_sgd_iter": 10,
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "model": {
            "vf_share_layers": True,
            "custom_model": "cfc",
            "max_seq_len": 20,
            "custom_model_config": {
                "cell_size": 64,
            },
        },
        "framework": "tf2",
    }

    algo = PPO(config=config)

    return algo

    
# Make predictions on samples
def make_predictions(self, samples, algo, args):
    # Get the policy
    policy = algo.get_policy()

    # Get the model
    model = policy.model()

    # Get the observation
    obs = samples["obs"]

    # Get the sequence length
    seq_len = samples["seq_len"]

    # Get the initial state
    state = model.get_initial_state()

    # Get the logits
    logits, _, _ = model.forward_rnn(obs, state, seq_len)

    # Get the action
    action = tf.argmax(logits, axis=-1)

    # Get the value
    value = model.value_function()



    return value




##########################################################

# Not our stuff and not from openai/baselines

###########################################################

def batch_value_prediction(pi, obz):
    ob_ph = U.get_placeholder_cached(name="ob")
    n_samples = len(obz)
    vpreds = np.zeros(n_samples, 'float32')
    batch_size = 2048
    n_batches = math.ceil(n_samples / batch_size)
    for i_batch in range(n_batches):
        idx = range(i_batch * batch_size, min((i_batch+1) * batch_size, n_samples))
        vpreds[idx] = pi.vpred.eval(feed_dict={ob_ph: obz[idx]})
    return vpreds


def create_saver(pi, for_expert=False):
    # Create saver, need to include some non-trainable variables, such as running mean std stuff.
    pi_mlp_var_list = pi.get_variables()
    if for_expert:
        # Since saved variables have names started with pi/, we work around it by creating a dictionary something like:
        # {"pi/pol/kernel": expert policy var, ...}
        pi_mlp_var_dict = {}
        for var in pi_mlp_var_list:
            splits = var.name.split("/")
            splits[0] = "pi"
            saved_var_name = "/".join(splits)
            # It's weird that the variables saved in the checkpoint do not have ":0", but var.name has,
            # although this ":0" works for pi, but not expert. We need to manually remove it.
            assert saved_var_name.split(":")[1] == "0"  # make sure it's :0
            saved_var_name = saved_var_name.split(":")[0]
            pi_mlp_var_dict[saved_var_name] = var
        pi_mlp_var_list = pi_mlp_var_dict  # a little abuse of list
    saver = tf.train.Saver(pi_mlp_var_list, max_to_keep=50)
    print(f'Create saver for expert: {for_expert}')
    # print('Saver variables:')
    # for var in pi_mlp_var_list:
    #     print(var if type(var) is str else var.name)
    return saver


#############################################################

# From openai/baselines

#############################################################

def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    # Indicates whether this sample is the first of a new episode / rollout.
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)    # In baseline this line is: pi.step(ob, stochastic=stochastic)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        # If this is the first step of a new episode, set vpred to zero. But why? So that the previous sample
        # would be considered as the last one of the episode?
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "nextob": ob, "nextnew": new} # "nextob": ob, "nextnew": new is not in baselines
            # Note that env is not reset after the policy and value function is updated.
            # Not sure why action is based on prevoius policy, but the value function is based on the new one.
            _, vpred = pi.act(stochastic, ob)  # In baseline this line is: pi.step(ob, stochastic=stochastic)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

##################################################################

# This is not from openai/baselines

##################################################################
def build_ilgain(mode, pi, expert, ilrew, ac_scale, ratio):
    """Dagger gain. ilrew can be used here. reduced_mean does not make sense here."""
    coeff_var = 0.0  # the coeff_var in dagger cost
    if expert is not None:
        print('Build dagger ilgain')
        ac_means = pi.pd.mode()
        expert_acs = expert.pd.mode()  # gradient should only be taken w.r.t. pi
        diff = (ac_means - expert_acs) / ac_scale
        gain = tf.reduce_mean(tf.reduce_sum(-(diff ** 2), axis=1))
        gain -= coeff_var * tf.reduce_mean(tf.reduce_sum(tf.exp(2 * pi.pd.logstd) / ac_scale**2, axis=1))
    else:
        gain = tf.constant(0.0)
    return gain

##################################################################

# This is from openai/baselines, but modified. Mode and truncated_horizon are added.
# is added.

##################################################################
def add_vtarg_and_adv(mode, seg, gamma, lam, truncated_horizon):
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    new = np.append(seg["new"], 0)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.zeros(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0

    # Mode selction is not from baselines.
    if mode == "pretrain_il":
        print("pretrain with TD(0)")
        seg["tdlamret"] = rew + gamma * vpred[1:]
    else:
        for t in reversed(range(T)):
            # whether this sample is terminal is based on the new of next sample.
            nonterminal = 1-new[t+1]
            delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        seg["tdlamret"] = seg["adv"] + seg["vpred"]
        print("GAE v estimation")

##################################################################

# This is not openai/baselines.

##################################################################
def add_expert_v(expert, seg):
    expert_v = batch_value_prediction(expert, seg["ob"])
    expert_v = np.append(expert_v, expert.act(True, seg["nextob"])[1])
    seg["expert_v"] = expert_v


def add_ilrew(mode, seg, expert, gamma, lam, truncated_horizon=None, il_gae=False):
    T = len(seg["rew"])
    vpred = np.append(seg["vpred"], seg["nextvpred"])

    ilrew = np.zeros(T, 'float32')

    seg["ilrew"] = ilrew


##################################################################

# This is from openai/baselines, but modified several ways for args.

##################################################################
def learn(env, policy_fn, *,
          timesteps_per_batch,  # what to train on
          cg_iters,
          gamma, lam,  # advantage estimation
          entcoeff=0.0,
          cg_damping=1e-2,
          vf_stepsize=3e-4,
          vf_iters=3,
          max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
          mode="train_expert", policy_save_freq=50, expert_dir=None, expert_file_prefix='policy',
          hard_switch_iter=None, truncated_horizon=None,
          save_no_policy=False, pretrain_dir=None, pretrain_file_prefix='policy', il_gae=False,
          callback=None,
          deterministic_expert=False,
          ):

    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    grad_batch_size = 64 # Different 
    pretrain_rollouts_file = 'pretrain_rollouts.npz' # Different

    # In baselines there is a policy = build_policy because no policy_fn is passed in. And set seeds

# different
    if hard_switch_iter is not None:
        # Since it can be randomly generated, in order to have consensus among workers
        hard_switch_iter = MPI.COMM_WORLD.bcast(hard_switch_iter, root=0)

    # Print with color for te rank 0 process. Not from baselines.
    def r0print(msg, with_color=False):
        if rank == 0:
            if with_color:
                print(colorize(msg, color='magenta'))
            else:
                print(msg)

    np.set_printoptions(precision=3)
    policy_files_prefix = 'policy' # Different
    policy_dir = logger.get_dir() # Different
    r0print('The policies will be saved to: {}'.format(policy_dir), True) # Different
    # Setup losses and stuff
    # ----------------------------------------
    # Building blocks
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)  # mlp_policy. So this is different but the same ish as baselines
    oldpi = policy_fn("oldpi", ob_space, ac_space)
    # Create expert policy model if necessary. Different from baselines until next whitespace
    if expert_dir:
        assert osp.isdir(expert_dir)
    expert = None if not expert_dir else policy_fn("expert", ob_space, ac_space)
    if expert_dir:
        expert_saver = create_saver(expert, for_expert=True)
        expert_path = osp.join(expert_dir, expert_file_prefix + '.ckpt')
    if pretrain_dir:
        pretrain_path = osp.join(pretrain_dir, pretrain_file_prefix + '.ckpt')
    policy_saver = create_saver(pi)

    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return
    # created in MlpPolicy init, 2d tensor
    ob = U.get_placeholder_cached(name="ob") # different

    # arg [None] is prepend_shape, so ac is a 2d tensor
    ac = pi.pdtype.sample_placeholder([None])
    # different until next whitespace
    ac_scale = tf.placeholder(dtype=tf.float32, shape=[ac.shape[1]])
    ilrew_ph = tf.placeholder(dtype=tf.float32, shape=[None])  # imitation learning reward, maybe needed for ilgain
    ilrate = tf.placeholder(dtype=tf.float32, shape=[])  # a float scalar
    phs = [ob, ac, atarg, ilrew_ph, ilrate, ac_scale]  # placeholders
    assign_updates = [tf.assign(oldv, newv) for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())]
    assign_old_eq_new = U.function([], [], updates=assign_updates)
    # Make sure trainable policy parameters are only from pi.
    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    # value function is using adam, and policy is not.
    vfadam = MpiAdam(vf_var_list)
    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)

    # Build up.
    # Value function objective. Modified from baselines.
    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

    # Policy gradient objective.
    ent = pi.pd.entropy() 
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent
    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
    rlgain = tf.reduce_mean(ratio * atarg) # This is the surrgain in baseline
    ilgain = build_ilgain(mode, pi, expert, ilrew_ph, ac_scale, ratio) # Different
    surrgain = ilgain * ilrate + (1 - ilrate) * rlgain # Different
    optimgain = surrgain + entbonus
    kloldnew = oldpi.pd.kl(pi.pd)
    meankl = tf.reduce_mean(kloldnew)
    losses = [optimgain, meankl, entbonus, ilgain, rlgain, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "ilgain", "rlgain", "surrgain", "entropy"]
    compute_losses = U.function(phs, losses) # Different
    compute_lossandgrad = U.function(phs, losses + [U.flatgrad(optimgain, var_list)]) # Different

    # KL constraint. Some stuff is removed
    dist = meankl
    klgrads = tf.gradients(dist, var_list) 
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)
    # Different
    compute_fvp = U.function([flat_tangent] + [ob, ac], fvp)  # not sure why atarg and ilrew are needed
    # in spite of the name, only grad is computed
    # ret will be fed by tdlamret, which is adv + vpred, and
    # vferr: tf.reduce_mean(tf.square(pi.vpred - ret))
    # It seems that he is not estimate V^{\pi, \gamma} (Eq.4 in trpo-gae paper), instead some weird stuff:
    # A^{GAE} in Eq.25 + \hat V_t
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()
    # Restore expert and/or initialize learner. This is different from baselines.
    if expert_dir:
        r0print('Retoring expert: {}'.format(expert_path), True)
        expert_saver.restore(tf.get_default_session(), expert_path)
        with tf.variable_scope("expert", reuse=True):
            with tf.variable_scope("pol", reuse=True):
                logstd = tf.get_variable(name="logstd")
                expert_logstd = logstd.eval()
        with tf.variable_scope("pi", reuse=True):
            with tf.variable_scope("pol", reuse=True):
                logstd = tf.get_variable(name="logstd")
        r0print('Expert logstd: {}'.format(expert_logstd), True)


    if pretrain_dir: # Different
        r0print('Initialize learner with pretrained model: {}'.format(pretrain_path), True)
        policy_saver.restore(tf.get_default_session(), pretrain_path)
        assign_old_eq_new()
        with tf.variable_scope("pi", reuse=True):
            with tf.variable_scope("pol", reuse=True):
                logstd = tf.get_variable(name="logstd")
                r0print('Pretrain policy logstd: {}'.format(logstd.eval()))

    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts 
    # ----------------------------------------
    # All mode stuff are different
    if mode == "pretrain_il":
        assert expert
        seg_gen = traj_segment_generator(expert, env, timesteps_per_batch,
                                         stochastic=not deterministic_expert)
    else:
        seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    # EpLenMean and EpRewMean are not just for the current batch, but a sliding window.
    lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards

    # Missing some stuff
    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) >= 1 

    while True:
        if callback:
            callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************" % iters_so_far)

        with timed("sampling"):
            # Still mode not in baselines and there is new stuff until next whitespace. I guess this is the main difference.
            if mode == "pretrain_il":
                # assert rank == 0  # only one worker! Not sure why now....
                if iters_so_far == 0:
                    seg = seg_gen.__next__()
                    ac_scale_val = np.std(seg["ac"], axis=0) if rank == 0 else np.ones(seg["ac"][0].shape)
                    ac_scale_val = MPI.COMM_WORLD.bcast(ac_scale_val, root=0)
                else:
                    # need to update vpred, nextvpred in seg!
                    seg['vpred'] = batch_value_prediction(pi, seg["ob"])
                    _, seg['nextvpred'] = pi.act(True, seg['nextob'])
            else:
                seg = seg_gen.__next__()
                # Action scale for dagger cost.
                # Use expert to generate rollouts
                if iters_so_far == 0:
                    ac_scale_val = np.ones(seg["ac"][0].shape)
                    if expert_dir:
                        r0print('Generating expert rollouts for action scaling', True)
                        if rank == 0:
                            expert_seg_gen = traj_segment_generator(expert, env, 4 * timesteps_per_batch, stochastic=True)
                            expert_seg = next(expert_seg_gen)
                            ac_scale_val = np.std(expert_seg["ac"], axis=0)
                        ac_scale_val = MPI.COMM_WORLD.bcast(ac_scale_val, root=0)
        print(f'ac scale: {ac_scale_val}')
        add_vtarg_and_adv(mode, seg, gamma, lam, truncated_horizon)
        add_ilrew(mode, seg, expert, gamma, lam, truncated_horizon, il_gae)

        ob_val, ac, atarg, tdlamret, ilrew = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg["ilrew"] # ilrew is new
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        # differences until hasattr
        if not np.allclose(atarg.std(), 0.0):
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
        else:
            atarg = (atarg - atarg.mean())
        if not np.allclose(ilrew.std(), 0.0):
            ilrew = (ilrew - ilrew.mean()) / ilrew.std()
        else:
            ilrew = (ilrew - ilrew.mean())
        # mlp_policy does not have this attr.
        if hasattr(pi, "ret_rms"):
            pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"):
            pi.ob_rms.update(ob_val)  # update running mean/std for policy, which is a tf layer before the nn.

        lossargs = [ob_val, ac]
        fvpargs = [arr[::5] for arr in lossargs]
        lossargs += [atarg, ilrew, ilrate_val, ac_scale_val]

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new()  # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*lossargs)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                # g: gradient over processes,
                # fisher_vector_product: after compute kl stuff using samples from local process, uses mpi
                # aggregate them. Although some computation can be saved in cg.
                stepdir = cg(fisher_vector_product, g,
                             cg_iters=cg_iters, verbose=rank == 0)
            assert np.isfinite(stepdir).all()  # interesting assert
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl) # Robin: max_kl is moved to LOKI.py, must find new way of importing it. TODO: Only used here, rewrite lm. 
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*lossargs)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum()))  # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with timed("vf"):
            for _ in range(vf_iters):
                # Note that we go over the data collected this iterations vf_iters times.
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                         include_final_partial_batch=False, batch_size=grad_batch_size):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.record_tabular("ilrate", ilrate_val)
        if rank == 0: # different 
            logger.dump_tabular()
            if iters_so_far % policy_save_freq == 0 and not save_no_policy:
                prefix = '{}_{}'.format(policy_files_prefix, iters_so_far)
                save_path = policy_saver.save(tf.get_default_session(), osp.join(policy_dir, prefix + '.ckpt'))
                r0print('Intemediate policy has been saved to {}'.format(save_path), True)
            r0print('Print logstd:', True)
            with tf.variable_scope("pi", reuse=True):
                with tf.variable_scope("pol", reuse=True):
                    logstd = tf.get_variable(name="logstd")
                    print(logstd.eval())

    if rank == 0 and not save_no_policy: # different
        r0print('Saving the final policy', True)
        save_path = policy_saver.save(tf.get_default_session(), osp.join(policy_dir, policy_files_prefix + '.ckpt'))
        r0print('Final policy has been saved to: {}'.format(save_path))


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
