from gym.envs.registration import register

register(
    id="molecule-v0",
    entry_point="gym_molecule.envs:MoleculeEnv",
)
