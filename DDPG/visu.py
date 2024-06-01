import matplotlib.pyplot as plt

# Read data from file
with open('data.txt', 'r') as file:
    lines = file.readlines()

# Extract timesteps, episodes, and values
timesteps = []
episodes = []
values = []
for line in lines:
    parts = line.split(',')
    timestep = int(parts[0].split()[1])
    episode = int(parts[1].split()[1])
    value = float(parts[2].split()[1])
    timesteps.append(timestep)
    episodes.append(episode)
    values.append(value)

# Plot data
plt.plot(episodes, values)
plt.xlabel('Episode')
plt.ylabel('Episode reward mean')
# plt.title('Episode reward mean of DDPG during training')
plt.grid()
plt.show()
