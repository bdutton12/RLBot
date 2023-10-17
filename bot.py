import rlgym
from model import PPOModel

"""
Main driver of the program 
"""
def main():
    # Create model here
    model = PPOModel()

    # Create env and set model env
    env = rlgym.make()
    model.SetEnv(env)

    while True:
        obs = env.reset()
        done = False
        print("Hello World")

        while not done:
            # Here we sample a random action. If you have an agent, you would get an action from it here.
            action = env.action_space.sample()

            next_obs, reward, done, gameinfo = env.step(action)

            obs = next_obs

main()