from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from tensorboardX import SummaryWriter
from baselines.ppo1.gcn_policy import discriminator,discriminator_net
import os
import copy


def traj_segment_generator(args, pi, env, horizon, stochastic, d_step_func, d_final_func):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()
    ob_adj = ob['adj']
    ob_node = ob['node']

    cur_ep_ret = 0 # return in current episode
    cur_ep_ret_env = 0
    cur_ep_ret_d_step = 0
    cur_ep_ret_d_final = 0
    cur_ep_len = 0 # len of current episode
    cur_ep_len_valid = 0
    ep_rets = [] # returns of completed episodes in this segment
    ep_rets_d_step = []
    ep_rets_d_final = []
    ep_rets_env = []
    ep_lens = [] # lengths of ...
    ep_lens_valid = [] # lengths of ...
    ep_rew_final = []
    ep_rew_final_stat = []



    # Initialize history arrays
    # obs = np.array([ob for _ in range(horizon)])
    ob_adjs = np.array([ob_adj for _ in range(horizon)])
    ob_nodes = np.array([ob_node for _ in range(horizon)])
    ob_adjs_final = []
    ob_nodes_final = []
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        # print('-------ac-call-----------')
        ac, vpred, debug = pi.act(stochastic, ob)
        # print('ob',ob)
        # print('debug ob_len',debug['ob_len'])
        # print('debug logits_stop_yes', debug['logits_stop_yes'])
        # print('debug logits_second_mask',debug['logits_second_mask'])
        # print('debug logits_first_mask', debug['logits_first_mask'])
        # print('debug logits_second_mask', debug['logits_second_mask'])
        # print('debug',debug)
        # print('ac',ac)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob_adj" : ob_adjs, "ob_node" : ob_nodes,"ob_adj_final" : np.array(ob_adjs_final), "ob_node_final" : np.array(ob_nodes_final), "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "ep_lens_valid" : ep_lens_valid, "ep_final_rew":ep_rew_final, "ep_final_rew_stat":ep_rew_final_stat,"ep_rets_env" : ep_rets_env,"ep_rets_d_step" : ep_rets_d_step,"ep_rets_d_final" : ep_rets_d_final}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            ep_lens_valid = []
            ep_rew_final = []
            ep_rew_final_stat = []
            ep_rets_d_step = []
            ep_rets_d_final = []
            ep_rets_env = []
            ob_adjs_final = []
            ob_nodes_final = []

        i = t % horizon
        # obs[i] = ob
        ob_adjs[i] = ob['adj']
        ob_nodes[i] = ob['node']
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew_env, new, info = env.step(ac)
        rew_d_step = 0 # default
        if rew_env>0: # if action valid
            cur_ep_len_valid += 1
            # add stepwise discriminator reward
            if args.has_d_step==1:
                if args.gan_type=='normal' or args.gan_type=='wgan':
                    rew_d_step = args.gan_step_ratio * (
                        d_step_func(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :])) / env.max_atom
                elif args.gan_type == 'recommend':
                    rew_d_step = args.gan_step_ratio * (
                        max(1-d_step_func(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :]),-2)) / env.max_atom
        rew_d_final = 0 # default
        if new:
            if args.has_d_final==1:
                if args.gan_type == 'normal' or args.gan_type=='wgan':
                    rew_d_final = args.gan_final_ratio * (
                        d_final_func(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :]))
                elif args.gan_type == 'recommend':
                    rew_d_final = args.gan_final_ratio * (
                        max(1 - d_final_func(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :]),
                            -2))

        rews[i] = rew_d_step + rew_env +rew_d_final

        cur_ep_ret += rews[i]
        cur_ep_ret_d_step += rew_d_step
        cur_ep_ret_d_final += rew_d_final
        cur_ep_ret_env += rew_env
        cur_ep_len += 1

        if new:
            if args.env=='molecule':
                with open('molecule_gen/'+args.name_full+'.csv', 'a') as f:
                    str = ''.join(['{},']*(len(info)+3))[:-1]+'\n'
                    f.write(str.format(info['smile'], info['reward_valid'], info['reward_qed'], info['reward_sa'], info['final_stat'], rew_env, rew_d_step, rew_d_final, cur_ep_ret, info['flag_steric_strain_filter'], info['flag_zinc_molecule_filter'], info['stop']))
            ob_adjs_final.append(ob['adj'])
            ob_nodes_final.append(ob['node'])
            ep_rets.append(cur_ep_ret)
            ep_rets_env.append(cur_ep_ret_env)
            ep_rets_d_step.append(cur_ep_ret_d_step)
            ep_rets_d_final.append(cur_ep_ret_d_final)
            ep_lens.append(cur_ep_len)
            ep_lens_valid.append(cur_ep_len_valid)
            ep_rew_final.append(rew_env)
            ep_rew_final_stat.append(info['final_stat'])
            cur_ep_ret = 0
            cur_ep_len = 0
            cur_ep_len_valid = 0
            cur_ep_ret_d_step = 0
            cur_ep_ret_d_final = 0
            cur_ep_ret_env = 0
            ob = env.reset()

        t += 1

def traj_final_generator(pi, env, batch_size, stochastic):
    ob = env.reset()
    ob_adj = ob['adj']
    ob_node = ob['node']
    ob_adjs = np.array([ob_adj for _ in range(batch_size)])
    ob_nodes = np.array([ob_node for _ in range(batch_size)])
    for i in range(batch_size):
        ob = env.reset()
        while True:
            ac, vpred, debug = pi.act(stochastic, ob)
            ob, rew_env, new, info = env.step(ac)
            np.set_printoptions(precision=2, linewidth=200)
            # print('ac',ac)
            # print('ob',ob['adj'],ob['node'])
            if new:
                ob_adjs[i]=ob['adj']
                ob_nodes[i]=ob['node']
                break
    return ob_adjs,ob_nodes

# def traj_segment_generator_scaffold(args, pi, env, horizon, stochastic, d_step_func, d_final_func):
#     t = 0
#     ac = env.action_space.sample() # not used, just so we have the datatype
#     new = True # marks if we're on first timestep of an episode
#     ob = env.reset()
#     ob_adj = ob['adj']
#     ob_node = ob['node']
#     ob_adj_scaffold = ob['adj_scaffold']
#     ob_node_scaffold = ob['node_scaffold']
#
#     cur_ep_ret = 0 # return in current episode
#     cur_ep_ret_env = 0
#     cur_ep_ret_d_step = 0
#     cur_ep_ret_d_final = 0
#     cur_ep_len = 0 # len of current episode
#     cur_ep_len_valid = 0
#     ep_rets = [] # returns of completed episodes in this segment
#     ep_rets_d_step = []
#     ep_rets_d_final = []
#     ep_rets_env = []
#     ep_lens = [] # lengths of ...
#     ep_lens_valid = [] # lengths of ...
#     ep_rew_final = []
#     ep_rew_final_stat = []
#
#
#
#     # Initialize history arrays
#     # obs = np.array([ob for _ in range(horizon)])
#     ob_adjs = np.array([ob_adj for _ in range(horizon)])
#     ob_nodes = np.array([ob_node for _ in range(horizon)])
#     ob_adjs_scaffold = np.array([ob_adj_scaffold for _ in range(horizon)])
#     ob_nodes_scaffold = np.array([ob_node_scaffold for _ in range(horizon)])
#     ob_adjs_final = []
#     ob_nodes_final = []
#     ob_adjs_scaffold_final = []
#     ob_nodes_scaffold_final = []
#     rews = np.zeros(horizon, 'float32')
#     vpreds = np.zeros(horizon, 'float32')
#     news = np.zeros(horizon, 'int32')
#     acs = np.array([ac for _ in range(horizon)])
#     prevacs = acs.copy()
#
#     while True:
#         prevac = ac
#         # print('-------ac-call-----------')
#         ac, vpred, debug = pi.act(stochastic, ob)
#         # print('ob',ob)
#         # print('debug ob_len',debug['ob_len'])
#         # print('debug logits_stop_yes', debug['logits_stop_yes'])
#         # print('debug logits_second_mask',debug['logits_second_mask'])
#         # print('debug logits_first_mask', debug['logits_first_mask'])
#         # print('debug logits_second_mask', debug['logits_second_mask'])
#         # print('debug',debug)
#         # print('ac',ac)
#
#         # Slight weirdness here because we need value function at time T
#         # before returning segment [0, T-1] so we get the correct
#         # terminal value
#         if t > 0 and t % horizon == 0:
#             yield {"ob_adj" : ob_adjs, "ob_node" : ob_nodes,"ob_adj_scaffold":ob_adjs_scaffold,
#                    "ob_node_scaffold" : ob_nodes_scaffold, "ob_adj_final" : np.array(ob_adjs_final),
#                    "ob_node_final" : np.array(ob_nodes_final), "rew" : rews, "vpred" : vpreds, "new" : news,
#                     "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
#                     "ep_rets" : ep_rets, "ep_lens" : ep_lens, "ep_lens_valid" : ep_lens_valid,
#                    "ep_final_rew":ep_rew_final, "ep_final_rew_stat":ep_rew_final_stat,"ep_rets_env" : ep_rets_env,
#                    "ep_rets_d_step" : ep_rets_d_step,"ep_rets_d_final" : ep_rets_d_final}
#             # Be careful!!! if you change the downstream algorithm to aggregate
#             # several of these batches, then be sure to do a deepcopy
#             ep_rets = []
#             ep_lens = []
#             ep_lens_valid = []
#             ep_rew_final = []
#             ep_rew_final_stat = []
#             ep_rets_d_step = []
#             ep_rets_d_final = []
#             ep_rets_env = []
#             ob_adjs_final = []
#             ob_nodes_final = []
#             ob_adjs_scaffold_final = []
#             ob_nodes_scaffold_final = []
#
#         i = t % horizon
#         # obs[i] = ob
#         ob_adjs[i] = ob['adj']
#         ob_nodes[i] = ob['node']
#         vpreds[i] = vpred
#         news[i] = new
#         acs[i] = ac
#         prevacs[i] = prevac
#
#         ob, rew_env, new, info = env.step(ac)
#         rew_d_step = 0 # default
#         if rew_env>0: # if action valid
#             cur_ep_len_valid += 1
#             # add stepwise discriminator reward
#             if args.has_d_step==1:
#                 rew_d_step = args.gan_step_ratio * (
#                     1 - d_step_func(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :])[0]) / env.max_atom
#         rew_d_final = 0 # default
#         if new:
#             if args.has_d_final==1:
#                 rew_d_final = args.gan_final_ratio * (
#                     1 - d_final_func(ob['adj'][np.newaxis, :, :, :], ob['node'][np.newaxis, :, :, :])[0])
#
#         rews[i] = rew_d_step + rew_env +rew_d_final
#
#         cur_ep_ret += rews[i]
#         cur_ep_ret_d_step += rew_d_step
#         cur_ep_ret_d_final += rew_d_final
#         cur_ep_ret_env += rew_env
#         cur_ep_len += 1
#
#         if new:
#             if args.env=='molecule':
#                 with open('molecule_gen/'+args.name_full+'.csv', 'a') as f:
#                     str = ''.join(['{},']*(len(info)+3))[:-1]+'\n'
#                     f.write(str.format(info['smile'], info['reward_valid'], info['reward_qed'], info['reward_sa'], info['final_stat'], rew_env, rew_d_step, rew_d_final, cur_ep_ret, info['flag_steric_strain_filter'], info['flag_zinc_molecule_filter'], info['stop']))
#             ob_adjs_final.append(ob['adj'])
#             ob_nodes_final.append(ob['node'])
#             ep_rets.append(cur_ep_ret)
#             ep_rets_env.append(cur_ep_ret_env)
#             ep_rets_d_step.append(cur_ep_ret_d_step)
#             ep_rets_d_final.append(cur_ep_ret_d_final)
#             ep_lens.append(cur_ep_len)
#             ep_lens_valid.append(cur_ep_len_valid)
#             ep_rew_final.append(rew_env)
#             ep_rew_final_stat.append(info['final_stat'])
#             cur_ep_ret = 0
#             cur_ep_len = 0
#             cur_ep_len_valid = 0
#             cur_ep_ret_d_step = 0
#             cur_ep_ret_d_final = 0
#             cur_ep_ret_env = 0
#             ob = env.reset()
#
#         t += 1
#
# def traj_final_generator_scaffold(pi, env, batch_size, stochastic):
#     ob = env.reset()
#     ob_adj = ob['adj']
#     ob_node = ob['node']
#     ob_adj_scaffold = ob['adj_scaffold']
#     ob_node_scaffold = ob['node_scaffold']
#     ob_adjs = np.array([ob_adj for _ in range(batch_size)])
#     ob_nodes = np.array([ob_node for _ in range(batch_size)])
#     ob_adjs_scaffold = np.array([ob_adj_scaffold for _ in range(batch_size)])
#     ob_nodes_scaffold = np.array([ob_node_scaffold for _ in range(batch_size)])
#     for i in range(batch_size):
#         ob = env.reset()
#         while True:
#             ac, vpred, debug = pi.act(stochastic, ob)
#             ob, rew_env, new, info = env.step(ac)
#             if new:
#                 ob_adjs[i]=ob['adj']
#                 ob_nodes[i]=ob['node']
#                 break
#     return ob_adjs,ob_nodes,ob_adjs_scaffold,ob_nodes_scaffold

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]





def learn(args,env, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        writer=None
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    # ob = U.get_placeholder_cached(name="ob")
    ob = {}
    ob['adj'] = U.get_placeholder_cached(name="adj")
    ob['node'] = U.get_placeholder_cached(name="node")

    ob_gen = {}
    ob_gen['adj'] = U.get_placeholder(shape=[None, ob_space['adj'].shape[0], None, None], dtype=tf.float32,name='adj_gen')
    ob_gen['node'] = U.get_placeholder(shape=[None, 1, None, ob_space['node'].shape[2]], dtype=tf.float32,name='node_gen')

    ob_real = {}
    ob_real['adj'] = U.get_placeholder(shape=[None,ob_space['adj'].shape[0],None,None],dtype=tf.float32,name='adj_real')
    ob_real['node'] = U.get_placeholder(shape=[None,1,None,ob_space['node'].shape[2]],dtype=tf.float32,name='node_real')

    # ac = pi.pdtype.sample_placeholder([None])
    # ac = tf.placeholder(dtype=tf.int64,shape=env.action_space.nvec.shape)
    ac = tf.placeholder(dtype=tf.int64, shape=[None,4],name='ac_real')

    ## PPO loss
    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    pi_logp = pi.pd.logp(ac)
    oldpi_logp = oldpi.pd.logp(ac)
    ratio_log = pi.pd.logp(ac) - oldpi.pd.logp(ac)

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    ## Expert loss
    loss_expert = -tf.reduce_mean(pi_logp)

    ## Discriminator loss
    # loss_d_step, _, _ = discriminator(ob_real, ob_gen,args, name='d_step')
    # loss_d_gen_step,_ = discriminator_net(ob_gen,args, name='d_step')
    # loss_d_final, _, _ = discriminator(ob_real, ob_gen,args, name='d_final')
    # loss_d_gen_final,_ = discriminator_net(ob_gen,args, name='d_final')


    step_pred_real, step_logit_real = discriminator_net(ob_real, args, name='d_step')
    step_pred_gen, step_logit_gen = discriminator_net(ob_gen, args, name='d_step')
    loss_d_step_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_real, labels=tf.ones_like(step_logit_real)*0.9))
    loss_d_step_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen, labels=tf.zeros_like(step_logit_gen)))
    loss_d_step = loss_d_step_real+loss_d_step_gen
    if args.gan_type=='normal':
        loss_g_step_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen, labels=tf.zeros_like(step_logit_gen)))
    elif args.gan_type=='recommend':
        loss_g_step_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen, labels=tf.ones_like(step_logit_gen)*0.9))
    elif args.gan_type=='wgan':
        loss_d_step, _, _ = discriminator(ob_real, ob_gen,args, name='d_step')
        loss_d_step = loss_d_step*-1
        loss_g_step_gen,_ = discriminator_net(ob_gen,args, name='d_step')


    final_pred_real, final_logit_real = discriminator_net(ob_real, args, name='d_final')
    final_pred_gen, final_logit_gen = discriminator_net(ob_gen, args, name='d_final')
    loss_d_final_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_real, labels=tf.ones_like(final_logit_real)*0.9))
    loss_d_final_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_gen, labels=tf.zeros_like(final_logit_gen)))
    loss_d_final = loss_d_final_real+loss_d_final_gen
    if args.gan_type == 'normal':
        loss_g_final_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_gen, labels=tf.zeros_like(final_logit_gen)))
    elif args.gan_type == 'recommend':
        loss_g_final_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_gen, labels=tf.ones_like(final_logit_gen)*0.9))
    elif args.gan_type=='wgan':
        loss_d_final, _, _ = discriminator(ob_real, ob_gen,args, name='d_final')
        loss_d_final = loss_d_final*-1
        loss_g_final_gen,_ = discriminator_net(ob_gen,args, name='d_final')


    var_list_pi = pi.get_trainable_variables()
    var_list_pi_stop = [var for var in var_list_pi if ('emb' in var.name) or ('gcn' in var.name) or ('stop' in var.name)]
    var_list_d_step = [var for var in tf.global_variables() if 'd_step' in var.name]
    var_list_d_final = [var for var in tf.global_variables() if 'd_final' in var.name]

    ## debug
    debug={}
    # debug['ac'] = ac
    # debug['ob_adj'] = ob['adj']
    # debug['ob_node'] = ob['node']
    # debug['pi_logp'] = pi_logp
    # debug['oldpi_logp'] = oldpi_logp
    # debug['kloldnew'] = kloldnew
    # debug['ent'] = ent
    # debug['ratio'] = ratio
    # debug['ratio_log'] = ratio_log
    # debug['emb_node2'] = pi.emb_node2
    # debug['pi_logitfirst'] = pi.logits_first
    # debug['pi_logitsecond'] = pi.logits_second
    # debug['pi_logitedge'] = pi.logits_edge
    #
    # debug['pi_ac'] = pi.ac
    # debug['oldpi_logitfirst'] = oldpi.logits_first
    # debug['oldpi_logitsecond'] = oldpi.logits_second
    # debug['oldpi_logitedge'] = oldpi.logits_edge
    #
    # debug['oldpi_ac'] = oldpi.ac
    #
    # with tf.variable_scope('pi/gcn1', reuse=tf.AUTO_REUSE):
    #     w = tf.get_variable('W')
    #     debug['w'] = w


    ## loss update function
    lossandgrad_ppo = U.function([ob['adj'], ob['node'], ac, pi.ac_real, oldpi.ac_real, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list_pi)])
    lossandgrad_expert = U.function([ob['adj'], ob['node'], ac, pi.ac_real], [loss_expert, U.flatgrad(loss_expert, var_list_pi)])
    lossandgrad_expert_stop = U.function([ob['adj'], ob['node'], ac, pi.ac_real], [loss_expert, U.flatgrad(loss_expert, var_list_pi_stop)])
    lossandgrad_d_step = U.function([ob_real['adj'], ob_real['node'], ob_gen['adj'], ob_gen['node']], [loss_d_step, U.flatgrad(loss_d_step, var_list_d_step)])
    lossandgrad_d_final = U.function([ob_real['adj'], ob_real['node'], ob_gen['adj'], ob_gen['node']], [loss_d_final, U.flatgrad(loss_d_final, var_list_d_final)])
    loss_g_gen_step_func = U.function([ob_gen['adj'], ob_gen['node']], loss_g_step_gen)
    loss_g_gen_final_func = U.function([ob_gen['adj'], ob_gen['node']], loss_g_final_gen)



    adam_pi = MpiAdam(var_list_pi, epsilon=adam_epsilon)
    adam_pi_stop = MpiAdam(var_list_pi_stop, epsilon=adam_epsilon)
    adam_d_step = MpiAdam(var_list_d_step, epsilon=adam_epsilon)
    adam_d_final = MpiAdam(var_list_d_final, epsilon=adam_epsilon)


    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    #
    # compute_losses_expert = U.function([ob['adj'], ob['node'], ac, pi.ac_real],
    #                                 loss_expert)
    compute_losses = U.function([ob['adj'], ob['node'], ac, pi.ac_real, oldpi.ac_real, atarg, ret, lrmult], losses)



    # Prepare for rollouts
    # ----------------------------------------
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    lenbuffer_valid = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_env = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_d_step = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_d_final = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_final = deque(maxlen=100) # rolling buffer for episode rewards
    rewbuffer_final_stat = deque(maxlen=100) # rolling buffer for episode rewardsn


    seg_gen = traj_segment_generator(args, pi, env, timesteps_per_actorbatch, True, loss_g_gen_step_func,loss_g_gen_final_func)


    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"
    if args.load==1:
        try:
            fname = './ckpt/' + args.name_full_load
            sess = tf.get_default_session()
            # sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(var_list_pi)
            saver.restore(sess, fname)
            iters_so_far = int(fname.split('_')[-1])+1
            print('model restored!', fname, 'iters_so_far:', iters_so_far)
        except:
            print(fname,'ckpt not found, start with iters 0')

    U.initialize()
    adam_pi.sync()
    adam_pi_stop.sync()
    adam_d_step.sync()
    adam_d_final.sync()

    counter = 0
    level = 0
    ## start training
    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        # logger.log("********** Iteration %i ************"%iters_so_far)



        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)
        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob_adj, ob_node, ac, atarg, tdlamret = seg["ob_adj"], seg["ob_node"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
        d = Dataset(dict(ob_adj=ob_adj, ob_node=ob_node, ac=ac, atarg=atarg, vtarg=tdlamret),
                    shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob_adj.shape[0]


        # inner training loop, train policy
        for i_optim in range(optim_epochs):

            loss_expert=0
            loss_expert_stop=0
            g_expert=0
            g_expert_stop=0


            loss_d_step = 0
            loss_d_final = 0
            g_ppo = 0
            g_d_step = 0
            g_d_final = 0


            pretrain_shift = 5
            ## Expert
            if iters_so_far>=args.expert_start and iters_so_far<=args.expert_end+pretrain_shift:
                ## Expert train
                # # # learn how to stop
                # ob_expert, ac_expert = env.get_expert(optim_batchsize, is_final=True)
                # loss_expert_stop, g_expert_stop = lossandgrad_expert_stop(ob_expert['adj'], ob_expert['node'], ac_expert,ac_expert)
                # loss_expert_stop = np.mean(loss_expert_stop)

                ob_expert, ac_expert = env.get_expert(optim_batchsize)
                loss_expert, g_expert = lossandgrad_expert(ob_expert['adj'], ob_expert['node'], ac_expert, ac_expert)
                loss_expert = np.mean(loss_expert)


            ## PPO
            if iters_so_far>=args.rl_start and iters_so_far<=args.rl_end:
                assign_old_eq_new() # set old parameter values to new parameter values
                batch = d.next_batch(optim_batchsize)
                # ppo
                # if args.has_ppo==1:
                if iters_so_far >= args.rl_start+pretrain_shift: # start generator after discriminator trained a well..
                    *newlosses, g_ppo = lossandgrad_ppo(batch["ob_adj"], batch["ob_node"], batch["ac"], batch["ac"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    losses_ppo=newlosses

                if args.has_d_step==1 and i_optim>=optim_epochs//2:
                    # update step discriminator
                    ob_expert, _ = env.get_expert(optim_batchsize,curriculum=args.curriculum,level_total=args.curriculum_num,level=level)
                    loss_d_step, g_d_step = lossandgrad_d_step(ob_expert["adj"], ob_expert["node"], batch["ob_adj"], batch["ob_node"])
                    adam_d_step.update(g_d_step, optim_stepsize * cur_lrmult)
                    loss_d_step = np.mean(loss_d_step)

                if args.has_d_final==1 and i_optim>=optim_epochs//4*3:
                    # update final discriminator
                    ob_expert, _ = env.get_expert(optim_batchsize, is_final=True, curriculum=args.curriculum,level_total=args.curriculum_num, level=level)
                    seg_final_adj, seg_final_node = traj_final_generator(pi, copy.deepcopy(env), optim_batchsize,True)
                    # update final discriminator
                    loss_d_final, g_d_final = lossandgrad_d_final(ob_expert["adj"], ob_expert["node"], seg_final_adj, seg_final_node)
                    # loss_d_final, g_d_final = lossandgrad_d_final(ob_expert["adj"], ob_expert["node"], ob_adjs, ob_nodes)
                    adam_d_final.update(g_d_final, optim_stepsize * cur_lrmult)
                    # print(seg["ob_adj_final"].shape)
                    # logger.log(fmt_row(13, np.mean(losses, axis=0)))

            # update generator
            # adam_pi_stop.update(0.1*g_expert_stop, optim_stepsize * cur_lrmult)

            # if g_expert==0:
            #     adam_pi.update(g_ppo, optim_stepsize * cur_lrmult)
            # else:
            adam_pi.update(0.2*g_ppo+0.05*g_expert, optim_stepsize * cur_lrmult)

        # WGAN
        # if args.has_d_step == 1:
        #     clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in var_list_d_step]
        # if args.has_d_final == 1:
        #     clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in var_list_d_final]
        #


        ## PPO val
        # if iters_so_far >= args.rl_start and iters_so_far <= args.rl_end:
        # logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob_adj"],batch["ob_node"], batch["ac"], batch["ac"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        # logger.log(fmt_row(13, meanlosses))

        if writer is not None:
            writer.add_scalar("loss_expert", loss_expert, iters_so_far)
            writer.add_scalar("loss_expert_stop", loss_expert_stop, iters_so_far)
            writer.add_scalar("loss_d_step", loss_d_step, iters_so_far)
            writer.add_scalar("loss_d_final", loss_d_final, iters_so_far)
            writer.add_scalar('grad_expert_min', np.amin(g_expert), iters_so_far)
            writer.add_scalar('grad_expert_max', np.amax(g_expert), iters_so_far)
            writer.add_scalar('grad_expert_norm', np.linalg.norm(g_expert), iters_so_far)
            writer.add_scalar('grad_expert_stop_min', np.amin(g_expert_stop), iters_so_far)
            writer.add_scalar('grad_expert_stop_max', np.amax(g_expert_stop), iters_so_far)
            writer.add_scalar('grad_expert_stop_norm', np.linalg.norm(g_expert_stop), iters_so_far)
            writer.add_scalar('grad_rl_min', np.amin(g_ppo), iters_so_far)
            writer.add_scalar('grad_rl_max', np.amax(g_ppo), iters_so_far)
            writer.add_scalar('grad_rl_norm', np.linalg.norm(g_ppo), iters_so_far)
            writer.add_scalar('g_d_step_min', np.amin(g_d_step), iters_so_far)
            writer.add_scalar('g_d_step_max', np.amax(g_d_step), iters_so_far)
            writer.add_scalar('g_d_step_norm', np.linalg.norm(g_d_step), iters_so_far)
            writer.add_scalar('g_d_final_min', np.amin(g_d_final), iters_so_far)
            writer.add_scalar('g_d_final_max', np.amax(g_d_final), iters_so_far)
            writer.add_scalar('g_d_final_norm', np.linalg.norm(g_d_final), iters_so_far)
            writer.add_scalar('learning_rate', optim_stepsize * cur_lrmult, iters_so_far)

        for (lossval, name) in zipsame(meanlosses, loss_names):
            # logger.record_tabular("loss_"+name, lossval)
            if writer is not None:
                writer.add_scalar("loss_"+name, lossval, iters_so_far)
        # logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        if writer is not None:
            writer.add_scalar("ev_tdlam_before", explained_variance(vpredbefore, tdlamret), iters_so_far)
        lrlocal = (seg["ep_lens"],seg["ep_lens_valid"], seg["ep_rets"],seg["ep_rets_env"],seg["ep_rets_d_step"],seg["ep_rets_d_final"],seg["ep_final_rew"],seg["ep_final_rew_stat"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, lens_valid, rews, rews_env, rews_d_step,rews_d_final, rews_final,rews_final_stat = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        lenbuffer_valid.extend(lens_valid)
        rewbuffer.extend(rews)
        rewbuffer_d_step.extend(rews_d_step)
        rewbuffer_d_final.extend(rews_d_final)
        rewbuffer_env.extend(rews_env)
        rewbuffer_final.extend(rews_final)
        rewbuffer_final_stat.extend(rews_final_stat)
        # logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        # logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        # logger.record_tabular("EpThisIter", len(lens))
        if writer is not None:
            writer.add_scalar("EpLenMean", np.mean(lenbuffer),iters_so_far)
            writer.add_scalar("EpLenValidMean", np.mean(lenbuffer_valid),iters_so_far)
            writer.add_scalar("EpRewMean", np.mean(rewbuffer),iters_so_far)
            writer.add_scalar("EpRewDStepMean", np.mean(rewbuffer_d_step), iters_so_far)
            writer.add_scalar("EpRewDFinalMean", np.mean(rewbuffer_d_final), iters_so_far)
            writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env),iters_so_far)
            writer.add_scalar("EpRewFinalMean", np.mean(rewbuffer_final),iters_so_far)
            writer.add_scalar("EpRewFinalStatMean", np.mean(rewbuffer_final_stat),iters_so_far)
            writer.add_scalar("EpThisIter", len(lens), iters_so_far)
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        # logger.record_tabular("EpisodesSoFar", episodes_so_far)
        # logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        # logger.record_tabular("TimeElapsed", time.time() - tstart)
        if writer is not None:
            writer.add_scalar("EpisodesSoFar", episodes_so_far, iters_so_far)
            writer.add_scalar("TimestepsSoFar", timesteps_so_far, iters_so_far)
            writer.add_scalar("TimeElapsed", time.time() - tstart, iters_so_far)


        if MPI.COMM_WORLD.Get_rank() == 0:
            with open('molecule_gen/' + args.name_full + '.csv', 'a') as f:
                f.write('***** Iteration {} *****\n'.format(iters_so_far))
            # save
            if iters_so_far % args.save_every == 0:
                fname = './ckpt/' + args.name_full + '_' + str(iters_so_far)
                saver = tf.train.Saver(var_list_pi)
                saver.save(tf.get_default_session(), fname)
                print('model saved!',fname)
                # fname = os.path.join(ckpt_dir, task_name)
                # os.makedirs(os.path.dirname(fname), exist_ok=True)
                # saver = tf.train.Saver()
                # saver.save(tf.get_default_session(), fname)
            # if iters_so_far==args.load_step:
        iters_so_far += 1
        counter += 1
        if counter%args.curriculum_step and counter//args.curriculum_step<args.curriculum_num:
            level += 1

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


# # scaffold
# def learn_scaffold(args,env, policy_fn, *,
#         timesteps_per_actorbatch, # timesteps per actor per update
#         clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
#         optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
#         gamma, lam, # advantage estimation
#         max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
#         callback=None, # you can do anything in the callback, since it takes locals(), globals()
#         adam_epsilon=1e-5,
#         schedule='constant', # annealing for stepsize parameters (epsilon and adam)
#         writer=None
#         ):
#     # Setup losses and stuff
#     # ----------------------------------------
#     ob_space = env.observation_space
#     ac_space = env.action_space
#     pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
#     oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
#     atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
#     ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
#
#     lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
#     clip_param = clip_param * lrmult # Annealed cliping parameter epislon
#
#     # ob = U.get_placeholder_cached(name="ob")
#     ob = {}
#     ob['adj'] = U.get_placeholder_cached(name="adj")
#     ob['node'] = U.get_placeholder_cached(name="node")
#     ob['adj_scaffold'] = U.get_placeholder_cached(name="adj_scaffold")
#     ob['node_scaffold'] = U.get_placeholder_cached(name="node_scaffold")
#
#     ob_gen = {}
#     ob_gen['adj'] = U.get_placeholder(shape=[None, ob_space['adj'].shape[0], None, None], dtype=tf.float32,name='adj_gen')
#     ob_gen['node'] = U.get_placeholder(shape=[None, 1, None, ob_space['node'].shape[2]], dtype=tf.float32,name='node_gen')
#     ob_gen['adj_scaffold'] = U.get_placeholder(shape=[None, ob_space['adj'].shape[0], None, None], dtype=tf.float32,
#                                       name='adj_scaffold_gen')
#     ob_gen['node_scaffold'] = U.get_placeholder(shape=[None, 1, None, ob_space['node'].shape[2]], dtype=tf.float32,
#                                        name='node_scaffold_gen')
#
#     ob_real = {}
#     ob_real['adj'] = U.get_placeholder(shape=[None,ob_space['adj'].shape[0],None,None],dtype=tf.float32,name='adj_real')
#     ob_real['node'] = U.get_placeholder(shape=[None,1,None,ob_space['node'].shape[2]],dtype=tf.float32,name='node_real')
#     ob_real['adj_scaffold'] = U.get_placeholder(shape=[None, ob_space['adj'].shape[0], None, None], dtype=tf.float32,
#                                       name='adj_scaffold_real')
#     ob_real['node_scaffold'] = U.get_placeholder(shape=[None, 1, None, ob_space['node'].shape[2]], dtype=tf.float32,
#                                        name='node_scaffold_real')
#
#     # ac = pi.pdtype.sample_placeholder([None])
#     # ac = tf.placeholder(dtype=tf.int64,shape=env.action_space.nvec.shape)
#     ac = tf.placeholder(dtype=tf.int64, shape=[None,4],name='ac_real')
#
#     ## PPO loss
#     kloldnew = oldpi.pd.kl(pi.pd)
#     ent = pi.pd.entropy()
#     meankl = tf.reduce_mean(kloldnew)
#     meanent = tf.reduce_mean(ent)
#     pol_entpen = (-entcoeff) * meanent
#
#     pi_logp = pi.pd.logp(ac)
#     oldpi_logp = oldpi.pd.logp(ac)
#     ratio_log = pi.pd.logp(ac) - oldpi.pd.logp(ac)
#
#     ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
#     surr1 = ratio * atarg # surrogate from conservative policy iteration
#     surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
#     pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
#     vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
#     total_loss = pol_surr + pol_entpen + vf_loss
#     losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
#     loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]
#
#     ## Expert loss
#     loss_expert = -tf.reduce_mean(pi_logp)
#
#     ## Discriminator loss
#     # loss_d_step, _, _ = discriminator(ob_real, ob_gen,args, name='d_step')
#     # loss_d_gen_step,_ = discriminator_net(ob_gen,args, name='d_step')
#     # loss_d_final, _, _ = discriminator(ob_real, ob_gen,args, name='d_final')
#     # loss_d_gen_final,_ = discriminator_net(ob_gen,args, name='d_final')
#
#
#     step_pred_real, step_logit_real = discriminator_net(ob_real, args, name='d_step')
#     step_pred_gen, step_logit_gen = discriminator_net(ob_gen, args, name='d_step')
#     loss_d_step_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_real, labels=tf.ones_like(step_logit_real)*0.9))
#     loss_d_step_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen, labels=tf.zeros_like(step_logit_gen)))
#     loss_d_step = loss_d_step_real+loss_d_step_gen
#     loss_g_step_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=step_logit_gen, labels=tf.zeros_like(step_logit_gen)))
#
#     final_pred_real, final_logit_real = discriminator_net(ob_real, args, name='d_final')
#     final_pred_gen, final_logit_gen = discriminator_net(ob_gen, args, name='d_final')
#     loss_d_final_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_real, labels=tf.ones_like(final_logit_real)*0.9))
#     loss_d_final_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_gen, labels=tf.zeros_like(final_logit_gen)))
#     loss_d_final = loss_d_final_real+loss_d_final_gen
#     loss_g_final_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_logit_gen, labels=tf.zeros_like(final_logit_gen)))
#
#
#     var_list_pi = pi.get_trainable_variables()
#     var_list_pi_stop = [var for var in var_list_pi if ('emb' in var.name) or ('gcn' in var.name) or ('stop' in var.name)]
#     var_list_d_step = [var for var in tf.global_variables() if 'd_step' in var.name]
#     var_list_d_final = [var for var in tf.global_variables() if 'd_final' in var.name]
#     # for var in var_list_pi:
#     #     print('var_list_pi',var)
#     # for var in var_list_pi_stop:
#     #     print('var_list_pi_stop', var)
#     # for var in var_list_d:
#     #     print('var_list_d', var)
#     ## debug
#     debug={}
#     debug['ac'] = ac
#     debug['ob_adj'] = ob['adj']
#     debug['ob_node'] = ob['node']
#     debug['pi_logp'] = pi_logp
#     debug['oldpi_logp'] = oldpi_logp
#     debug['kloldnew'] = kloldnew
#     debug['ent'] = ent
#     debug['ratio'] = ratio
#     debug['ratio_log'] = ratio_log
#     debug['emb_node2'] = pi.emb_node2
#     debug['pi_logitfirst'] = pi.logits_first
#     debug['pi_logitsecond'] = pi.logits_second
#     debug['pi_logitedge'] = pi.logits_edge
#
#     debug['pi_ac'] = pi.ac
#     debug['oldpi_logitfirst'] = oldpi.logits_first
#     debug['oldpi_logitsecond'] = oldpi.logits_second
#     debug['oldpi_logitedge'] = oldpi.logits_edge
#
#     debug['oldpi_ac'] = oldpi.ac
#
#     with tf.variable_scope('pi/gcn1', reuse=tf.AUTO_REUSE):
#         w = tf.get_variable('W')
#         debug['w'] = w
#
#
#     ## loss update function
#     lossandgrad_ppo = U.function([ob['adj'], ob['node'],ob['adj_scaffold'],ob['node_scaffold'], ac, pi.ac_real, oldpi.ac_real, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list_pi)])
#     lossandgrad_expert = U.function([ob['adj'], ob['node'],ob['adj_scaffold'],ob['node_scaffold'], ac, pi.ac_real], [loss_expert, U.flatgrad(loss_expert, var_list_pi)])
#     lossandgrad_expert_stop = U.function([ob['adj'], ob['node'],ob['adj_scaffold'],ob['node_scaffold'], ac, pi.ac_real], [loss_expert, U.flatgrad(loss_expert, var_list_pi_stop)])
#     lossandgrad_d_step = U.function([ob_real['adj'], ob_real['node'], ob_gen['adj'], ob_gen['node']], [loss_d_step, U.flatgrad(loss_d_step, var_list_d_step)])
#     lossandgrad_d_final = U.function([ob_real['adj'], ob_real['node'], ob_gen['adj'], ob_gen['node']], [loss_d_final, U.flatgrad(loss_d_final, var_list_d_final)])
#     loss_g_gen_step_func = U.function([ob_gen['adj'], ob_gen['node']], loss_g_step_gen)
#     loss_g_gen_final_func = U.function([ob_gen['adj'], ob_gen['node']], loss_g_final_gen)
#
#
#
#     adam_pi = MpiAdam(var_list_pi, epsilon=adam_epsilon)
#     adam_pi_stop = MpiAdam(var_list_pi_stop, epsilon=adam_epsilon)
#     adam_d_step = MpiAdam(var_list_d_step, epsilon=adam_epsilon)
#     adam_d_final = MpiAdam(var_list_d_final, epsilon=adam_epsilon)
#
#
#     assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
#         for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
#     #
#     # compute_losses_expert = U.function([ob['adj'], ob['node'], ac, pi.ac_real],
#     #                                 loss_expert)
#     compute_losses = U.function([ob['adj'], ob['node'], ac, pi.ac_real, oldpi.ac_real, atarg, ret, lrmult], losses)
#
#
#
#     # Prepare for rollouts
#     # ----------------------------------------
#     episodes_so_far = 0
#     timesteps_so_far = 0
#     iters_so_far = 0
#     tstart = time.time()
#     lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
#     lenbuffer_valid = deque(maxlen=100) # rolling buffer for episode lengths
#     rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
#     rewbuffer_env = deque(maxlen=100) # rolling buffer for episode rewards
#     rewbuffer_d_step = deque(maxlen=100) # rolling buffer for episode rewards
#     rewbuffer_d_final = deque(maxlen=100) # rolling buffer for episode rewards
#     rewbuffer_final = deque(maxlen=100) # rolling buffer for episode rewards
#     rewbuffer_final_stat = deque(maxlen=100) # rolling buffer for episode rewardsn
#
#
#     seg_gen = traj_segment_generator_scaffold(args, pi, env, timesteps_per_actorbatch, True, loss_g_gen_step_func,loss_g_gen_final_func)
#
#
#     assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"
#     if args.load==1:
#         try:
#             fname = './ckpt/' + args.name_full_load
#             sess = tf.get_default_session()
#             # sess.run(tf.global_variables_initializer())
#             saver = tf.train.Saver(var_list_pi)
#             saver.restore(sess, fname)
#             iters_so_far = int(fname.split('_')[-1])+1
#             print('model restored!', fname, 'iters_so_far:', iters_so_far)
#         except:
#             print(fname,'ckpt not found, start with iters 0')
#
#     U.initialize()
#     adam_pi.sync()
#     adam_pi_stop.sync()
#     adam_d_step.sync()
#     adam_d_final.sync()
#
#     counter = 0
#     level = 0
#     ## start training
#     while True:
#         if callback: callback(locals(), globals())
#         if max_timesteps and timesteps_so_far >= max_timesteps:
#             break
#         elif max_episodes and episodes_so_far >= max_episodes:
#             break
#         elif max_iters and iters_so_far >= max_iters:
#             break
#         elif max_seconds and time.time() - tstart >= max_seconds:
#             break
#
#         if schedule == 'constant':
#             cur_lrmult = 1.0
#         elif schedule == 'linear':
#             cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
#         else:
#             raise NotImplementedError
#
#         # logger.log("********** Iteration %i ************"%iters_so_far)
#
#
#         #
#         ## Expert
#         loss_expert=0
#         loss_expert_stop=0
#         g_expert=0
#         # # if args.has_expert==1:
#         # if iters_so_far>=args.expert_start and iters_so_far<=args.expert_end:
#         #     ## Expert train
#         #     losses = []  # list of tuples, each of which gives the loss for a minibatch
#         #     losses_stop = []  # list of tuples, each of which gives the loss for a minibatch
#         #     for _ in range(optim_epochs*args.supervise_time):
#         #         # learn how to stop
#         #         ob_expert, ac_expert = env.get_expert(optim_batchsize, is_final=True)
#         #         losses_expert_stop, g_expert = lossandgrad_expert_stop(ob_expert['adj'], ob_expert['node'], ac_expert,ac_expert)
#         #         adam_pi_stop.update(g_expert, optim_stepsize * cur_lrmult/10)
#         #         losses_stop.append(losses_expert_stop)
#         #         ob_expert, ac_expert = env.get_expert(optim_batchsize)
#         #         losses_expert, g_expert = lossandgrad_expert(ob_expert['adj'], ob_expert['node'], ac_expert, ac_expert)
#         #         adam_pi.update(g_expert, optim_stepsize * cur_lrmult/10)
#         #         losses.append(losses_expert)
#         #
#         #     loss_expert = np.mean(losses, axis=0, keepdims=True)
#         #     loss_expert_stop = np.mean(losses_stop, axis=0, keepdims=True)
#         #     # logger.log(fmt_row(13, loss_expert))
#
#         ## PPO
#         seg = seg_gen.__next__()
#         add_vtarg_and_adv(seg, gamma, lam)
#
#         # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
#         ob_adj, ob_node,ob_adj_scaffold,ob_node_scaffold, ac, atarg, tdlamret = seg["ob_adj"], seg["ob_node"],seg["ob_adj_scaffold"], seg["ob_node_scaffold"], seg["ac"], seg["adv"], seg["tdlamret"]
#         vpredbefore = seg["vpred"] # predicted value function before udpate
#         atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
#         d = Dataset(dict(ob_adj=ob_adj, ob_node=ob_node,ob_adj_scaffold=ob_adj_scaffold,ob_node_scaffold=ob_node_scaffold, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
#         optim_batchsize = optim_batchsize or ob_adj.shape[0]
#
#
#         loss_d_step=0
#         loss_d_final=0
#         g_ppo=0
#         g_d_step=0
#         g_d_final=0
#         # if args.has_rl==1:
#         if iters_so_far>=args.rl_start and iters_so_far<=args.rl_end:
#             ## PPO train
#             assign_old_eq_new() # set old parameter values to new parameter values
#             # logger.log("Optimizing...")
#             # logger.log(fmt_row(13, loss_names))
#             # Here we do a bunch of optimization epochs over the data
#             if args.has_d_final == 1:
#                 ob_expert, _ = env.get_expert(optim_batchsize, is_final=True, curriculum=args.curriculum,
#                                               level_total=args.curriculum_num, level=level)
#                 seg_final_adj, seg_final_node = traj_final_generator_scaffold(pi, copy.deepcopy(env), optim_batchsize, True)
#             for _ in range(optim_epochs):
#                 losses_ppo = [] # list of tuples, each of which gives the loss for a minibatch
#                 losses_d_step = []
#                 for batch in d.iterate_once(optim_batchsize):
#                     # ppo
#                     if args.has_ppo==1:
#                         *newlosses, g_ppo = lossandgrad_ppo(batch["ob_adj"], batch["ob_node"], batch["ac"], batch["ac"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
#                         adam_pi.update(g_ppo, optim_stepsize * cur_lrmult)
#                         losses_ppo.append(newlosses)
#                     if args.has_d_step==1:
#                         # update step discriminator
#                         ob_expert, _ = env.get_expert(optim_batchsize,curriculum=args.curriculum,level_total=args.curriculum_num,level=level)
#                         loss_d_step, g_d_step = lossandgrad_d_step(ob_expert["adj"], ob_expert["node"], batch["ob_adj"], batch["ob_node"])
#                         # print('loss_d_step',loss_d_step,g_d_step)
#                         adam_d_step.update(g_d_step, optim_stepsize * cur_lrmult)
#                         losses_d_step.append(loss_d_step)
#                 loss_d_step = np.mean(losses_d_step, axis=0, keepdims=True)
#                 if args.has_d_final==1:
#                     # update final discriminator
#                     loss_d_final, g_d_final = lossandgrad_d_final(ob_expert["adj"], ob_expert["node"], seg_final_adj, seg_final_node)
#                     # loss_d_final, g_d_final = lossandgrad_d_final(ob_expert["adj"], ob_expert["node"], ob_adjs, ob_nodes)
#                     adam_d_final.update(g_d_final, optim_stepsize * cur_lrmult)
#                     # print(seg["ob_adj_final"].shape)
#                     # logger.log(fmt_row(13, np.mean(losses, axis=0)))
#         #
#         # if args.has_d_step == 1:
#         #     clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in var_list_d_step]
#         # if args.has_d_final == 1:
#         #     clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in var_list_d_final]
#         #
#
#         ## PPO val
#         # logger.log("Evaluating losses...")
#         losses = []
#         for batch in d.iterate_once(optim_batchsize):
#             newlosses = compute_losses(batch["ob_adj"],batch["ob_node"], batch["ac"], batch["ac"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
#             losses.append(newlosses)
#         meanlosses,_,_ = mpi_moments(losses, axis=0)
#         # logger.log(fmt_row(13, meanlosses))
#
#         # logger.record_tabular("loss_expert", loss_expert)
#         # logger.record_tabular('grad_expert_min',np.amin(g_expert))
#         # logger.record_tabular('grad_expert_max',np.amax(g_expert))
#         # logger.record_tabular('grad_expert_norm', np.linalg.norm(g_expert))
#         # logger.record_tabular('grad_rl_min', np.amin(g))
#         # logger.record_tabular('grad_rl_max', np.amax(g))
#         # logger.record_tabular('grad_rl_norm', np.linalg.norm(g))
#         # logger.record_tabular('learning_rate', optim_stepsize * cur_lrmult)
#
#         if writer is not None:
#             writer.add_scalar("loss_expert", loss_expert, iters_so_far)
#             writer.add_scalar("loss_expert_stop", loss_expert_stop, iters_so_far)
#             writer.add_scalar("loss_d_step", loss_d_step, iters_so_far)
#             writer.add_scalar("loss_d_final", loss_d_final, iters_so_far)
#             writer.add_scalar('grad_expert_min', np.amin(g_expert), iters_so_far)
#             writer.add_scalar('grad_expert_max', np.amax(g_expert), iters_so_far)
#             writer.add_scalar('grad_expert_norm', np.linalg.norm(g_expert), iters_so_far)
#             writer.add_scalar('grad_rl_min', np.amin(g_ppo), iters_so_far)
#             writer.add_scalar('grad_rl_max', np.amax(g_ppo), iters_so_far)
#             writer.add_scalar('grad_rl_norm', np.linalg.norm(g_ppo), iters_so_far)
#             writer.add_scalar('g_d_step_min', np.amin(g_d_step), iters_so_far)
#             writer.add_scalar('g_d_step_max', np.amax(g_d_step), iters_so_far)
#             writer.add_scalar('g_d_step_norm', np.linalg.norm(g_d_step), iters_so_far)
#             writer.add_scalar('g_d_final_min', np.amin(g_d_final), iters_so_far)
#             writer.add_scalar('g_d_final_max', np.amax(g_d_final), iters_so_far)
#             writer.add_scalar('g_d_final_norm', np.linalg.norm(g_d_final), iters_so_far)
#             writer.add_scalar('learning_rate', optim_stepsize * cur_lrmult, iters_so_far)
#
#         for (lossval, name) in zipsame(meanlosses, loss_names):
#             # logger.record_tabular("loss_"+name, lossval)
#             if writer is not None:
#                 writer.add_scalar("loss_"+name, lossval, iters_so_far)
#         # logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
#         if writer is not None:
#             writer.add_scalar("ev_tdlam_before", explained_variance(vpredbefore, tdlamret), iters_so_far)
#         lrlocal = (seg["ep_lens"],seg["ep_lens_valid"], seg["ep_rets"],seg["ep_rets_env"],seg["ep_rets_d_step"],seg["ep_rets_d_final"],seg["ep_final_rew"],seg["ep_final_rew_stat"]) # local values
#         listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
#         lens, lens_valid, rews, rews_env, rews_d_step,rews_d_final, rews_final,rews_final_stat = map(flatten_lists, zip(*listoflrpairs))
#         lenbuffer.extend(lens)
#         lenbuffer_valid.extend(lens_valid)
#         rewbuffer.extend(rews)
#         rewbuffer_d_step.extend(rews_d_step)
#         rewbuffer_d_final.extend(rews_d_final)
#         rewbuffer_env.extend(rews_env)
#         rewbuffer_final.extend(rews_final)
#         rewbuffer_final_stat.extend(rews_final_stat)
#         # logger.record_tabular("EpLenMean", np.mean(lenbuffer))
#         # logger.record_tabular("EpRewMean", np.mean(rewbuffer))
#         # logger.record_tabular("EpThisIter", len(lens))
#         if writer is not None:
#             writer.add_scalar("EpLenMean", np.mean(lenbuffer),iters_so_far)
#             writer.add_scalar("EpLenValidMean", np.mean(lenbuffer_valid),iters_so_far)
#             writer.add_scalar("EpRewMean", np.mean(rewbuffer),iters_so_far)
#             writer.add_scalar("EpRewDStepMean", np.mean(rewbuffer_d_step), iters_so_far)
#             writer.add_scalar("EpRewDFinalMean", np.mean(rewbuffer_d_final), iters_so_far)
#             writer.add_scalar("EpRewEnvMean", np.mean(rewbuffer_env),iters_so_far)
#             writer.add_scalar("EpRewFinalMean", np.mean(rewbuffer_final),iters_so_far)
#             writer.add_scalar("EpRewFinalStatMean", np.mean(rewbuffer_final_stat),iters_so_far)
#             writer.add_scalar("EpThisIter", len(lens), iters_so_far)
#         episodes_so_far += len(lens)
#         timesteps_so_far += sum(lens)
#         # logger.record_tabular("EpisodesSoFar", episodes_so_far)
#         # logger.record_tabular("TimestepsSoFar", timesteps_so_far)
#         # logger.record_tabular("TimeElapsed", time.time() - tstart)
#         if writer is not None:
#             writer.add_scalar("EpisodesSoFar", episodes_so_far, iters_so_far)
#             writer.add_scalar("TimestepsSoFar", timesteps_so_far, iters_so_far)
#             writer.add_scalar("TimeElapsed", time.time() - tstart, iters_so_far)
#
#
#         if MPI.COMM_WORLD.Get_rank() == 0:
#             with open('molecule_gen/' + args.name_full + '.csv', 'a') as f:
#                 f.write('***** Iteration {} *****\n'.format(iters_so_far))
#             # save
#             if iters_so_far % args.save_every == 0:
#                 fname = './ckpt/' + args.name_full + '_' + str(iters_so_far)
#                 saver = tf.train.Saver(var_list_pi)
#                 saver.save(tf.get_default_session(), fname)
#                 print('model saved!',fname)
#                 # fname = os.path.join(ckpt_dir, task_name)
#                 # os.makedirs(os.path.dirname(fname), exist_ok=True)
#                 # saver = tf.train.Saver()
#                 # saver.save(tf.get_default_session(), fname)
#             # if iters_so_far==args.load_step:
#         iters_so_far += 1
#         counter += 1
#         if counter%args.curriculum_step and counter//args.curriculum_step<args.curriculum_num:
#             level += 1
#
#
#
#







