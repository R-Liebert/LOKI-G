import gym
import datetime
import csv

env = gym.make('Breakout-v5')

# Create a unique file name based on the current time
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f"breakout-{timestamp}.csv"

observations = []
rewards = []
actions = []
dones = []

while True:
    env.reset()
    done = False
    while not done:
        env.render()
        action = input("Enter action (0-3): ")
        action = int(action)
        obs, reward, done, info = env.step(action)
        observations.append(obs)
        rewards.append(reward)
        actions.append(action)
        dones.append(done)
    print("Game over, reward:", reward)

# Save the observations, rewards, actions, and dones to a CSV file
with open(filename, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["observation", "reward", "action", "done"])
    writer.writeheader()
    for i in range(len(observations)):
        writer.writerow({
            "observation": observations[i],
            "reward": rewards[i],
            "action": actions[i],
            "done": dones[i]
        })
