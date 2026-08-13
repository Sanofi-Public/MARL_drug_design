from gymnasium.envs.registration import register

# New modular environment classes
from environment.base_mol_env import BaseMoleculeEnv, Agent
from environment.fragment_env import FragmentMoleculeEnv

# Legacy environment registration
def register_fmpo_collect_env(env_length):
     register(
          id= "molMARL/madfmpo_2025" ,
          entry_point="environment.fmpo_collect_v2:madfmpo_drugenv",
          kwargs={"env_length": env_length}
     )

# # Example usage:
register_fmpo_collect_env(env_length=100)

__all__ = [
    'BaseMoleculeEnv',
    'Agent', 
    'FragmentMoleculeEnv',
    'register_fmpo_collect_env',
]