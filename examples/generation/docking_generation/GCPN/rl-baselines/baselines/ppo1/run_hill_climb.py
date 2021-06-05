import gym
from gym_molecule.envs.molecule import GraphEnv
import numpy as np

env = gym.make('molecule-v0') # in gym format

env.init(data_type='zinc',reward_type='qed',force_final=True)
# env.reset()
# act = np.array([[0,1,0,0]])
# ob,reward,new,info = env.step(act)
# print(reward,new,info['smile'])
# print(env.get_observation())


# best_smile = 'C'
# best_reward = 0
# for i in range(40):
#     env.reset(best_smile)
#     atom_num = env.mol.GetNumAtoms()
#     list_smile = []
#     list_reward = []
#     for first in range(atom_num):
#         env.reset(best_smile)
#         second = atom_num
#         edge = 0
#         stop = 0
#         act = np.array([[first,second,edge,stop]])
#         ob,reward,new,info = env.step(act)
#         list_smile.append(info['smile'])
#         list_reward.append(reward/2.0)
#     list_reward = np.array(list_reward)
#     best_reward = np.amax(list_reward)
#     best_reward_id = np.argmax(list_reward)
#     best_smile = list_smile[best_reward_id]
#     print('Step {}, best reward {}, best smile {}'.format(i,best_reward,best_smile))


# ## harder
# best_smile = 'C'
# best_reward = 0
# for i in range(40):
#     env.reset(best_smile)
#     atom_num = env.mol.GetNumAtoms()
#     list_smile = []
#     list_reward = []
#     for first in range(atom_num):
#         for second in range(atom_num,atom_num+3):
#             for edge in range(0,2):
#                 env.reset(best_smile)
#                 stop = 0
#                 act = np.array([[first, second, edge, stop]])
#                 ob, reward, new, info = env.step(act)
#                 list_smile.append(info['smile'])
#                 list_reward.append(reward / 2.0)
#     list_reward = np.array(list_reward)
#     best_reward = np.amax(list_reward)
#     best_reward_id = np.argmax(list_reward)
#     best_smile = list_smile[best_reward_id]
#     print('Step {}, best reward {}, best smile {}'.format(i, best_reward, best_smile))


# ## traverse
# best_smile = 'C'
# best_reward = 0
# for i in range(40):
#     env.reset(best_smile)
#     atom_num = env.mol.GetNumAtoms()
#     list_smile = []
#     list_reward = []
#     for first in range(atom_num):
#         for second in range(0, atom_num + 3):
#             if second==first:
#                 continue
#             for edge in range(0, 2):
#                 env.reset(best_smile)
#                 stop = 0
#                 act = np.array([[first, second, edge, stop]])
#                 ob, reward, new, info = env.step(act)
#                 list_smile.append(info['smile'])
#                 list_reward.append(reward / 2.0)
#     list_reward = np.array(list_reward)
#     best_reward = np.amax(list_reward)
#     best_reward_id = np.argmax(list_reward)
#     best_smile = list_smile[best_reward_id]
#     print('Step {}, best reward {}, best smile {}'.format(i, best_reward, best_smile))

# ## random
# best_smile = 'C'
# best_reward = 0
# topk = 5
# for i in range(10):
#     env.reset(best_smile)
#     atom_num = env.mol.GetNumAtoms()
#     list_smile = []
#     list_reward = []
#     for first in range(atom_num):
#         for second in range(0, atom_num + 3):
#             if second == first:
#                 continue
#             for edge in range(0, 2):
#                 env.reset(best_smile)
#                 stop = 0
#                 act = np.array([[first, second, edge, stop]])
#                 ob, reward, new, info = env.step(act)
#                 list_smile.append(info['smile'])
#                 list_reward.append(reward / 2.0)
#     list_reward = np.array(list_reward)
#     if i<39:
#         select = min(len(list_reward)-1,topk)
#         list_select = list_reward.argsort()[-select:][::-1]
#         best_reward_id = np.random.choice(list_select)
#     else:
#         best_reward_id = np.argmax(list_reward)
#     best_reward = list_reward[best_reward_id]
#     best_smile = list_smile[best_reward_id]
#     print('Step {}, best reward {}, best smile {}'.format(i, best_reward, best_smile))


import multiprocessing
import os


class Worker(multiprocessing.Process):

    def run(self):
        """worker function"""
        # fname = 'hill_climb_results/' + self.name + '.txt'
        fname = 'hill_climb_results/all.txt'
        # if os.path.isfile(fname):
        #     os.remove(fname)
        ## random
        np.random.seed()
        env = gym.make('molecule-v0')  # in gym format
        env.init(data_type='zinc', reward_type='qed', force_final=True)
        best_smile_best = 'C'
        best_reward_best = 0

        atom_num = 38
        topk = 5
        repeat_time = 20

        for repeat in range(repeat_time):
            best_smile = 'C'
            best_reward = 0
            for i in range(atom_num):
                env.reset(best_smile)
                atom_num = env.mol.GetNumAtoms()
                list_smile = []
                list_reward = []
                for first in range(atom_num):
                    for second in range(0, atom_num + 9):
                        if second == first:
                            continue
                        for edge in range(0, 3):
                            env.reset(best_smile)
                            stop = 0
                            act = np.array([[first, second, edge, stop]])
                            ob, reward, new, info = env.step(act)
                            list_smile.append(info['smile'])
                            list_reward.append(reward / 2.0)
                list_reward = np.array(list_reward)
                if i < atom_num:
                    select = min(len(list_reward) - 1, topk)
                    list_select = list_reward.argsort()[-select:][::-1]
                    best_reward_id = np.random.choice(list_select)
                else:
                    best_reward_id = np.argmax(list_reward)
                best_reward = list_reward[best_reward_id]
                best_smile = list_smile[best_reward_id]
                if best_reward>best_reward_best:
                    best_reward_best = best_reward
                    best_smile_best = best_smile
                # print('Process {} Step {}, best reward {}, best smile {}, best reward best {}, best smile best {}'.format(self.name, i, best_reward, best_smile, best_reward_best, best_smile_best))

            with open(fname,'a') as f:
                f.write('Process {}, best reward best {}, best smile best {}\n'.format(self.name, best_reward_best, best_smile_best))
        return

if __name__ == '__main__':
    if not os.path.exists('hill_climb_results'):
        os.makedirs('hill_climb_results')
    jobs = []
    for i in range(288):
        # p = multiprocessing.Process(target=worker)
        p = Worker()
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()