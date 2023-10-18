import rlgym
from model.agent import Agent

"""
Main driver of the program 
"""
def main():


    # Create env and model
    env = rlgym.make()
    N = 20
    batch_size = 5
    n_epochs = 4
    eta = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                eta=eta, n_epochs=n_epochs, 
                input_dims=env.observation_space.shape)

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