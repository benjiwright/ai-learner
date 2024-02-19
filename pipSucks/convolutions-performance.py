import matplotlib.pyplot as plt

# Data
epoch = list(range(1, 51))
seconds = [
    6.26, 5.67, 5.55, 5.65, 5.67, 5.59, 5.66, 5.92, 5.91, 6.02,
    5.92, 6.18, 5.89, 5.86, 5.74, 5.73, 5.64, 5.57, 5.57, 5.79,
    5.56, 5.5, 5.53, 5.52, 5.55, 5.72, 5.5, 5.66, 5.91, 6.42,
    6.55, 6.48, 6.51, 6.24, 6.08, 6.2, 6.08, 6.01, 5.59, 5.61,
    5.75, 5.54, 5.65, 5.65, 5.56, 5.67, 5.62, 5.5, 5.53, 5.49
]
accuracy = [
    0.8397, 0.8931, 0.9086, 0.9199, 0.9291, 0.936, 0.9451, 0.9519, 0.9561, 0.9622,
    0.9668, 0.9706, 0.9733, 0.9764, 0.9794, 0.9798, 0.9824, 0.9846, 0.9866, 0.9862,
    0.9868, 0.9884, 0.9891, 0.9887, 0.9889, 0.991, 0.9894, 0.9909, 0.9921, 0.9906,
    0.992, 0.9921, 0.9927, 0.9917, 0.9925, 0.9932, 0.993, 0.9934, 0.993, 0.9935,
    0.9936, 0.9939, 0.9941, 0.9937, 0.9933, 0.9946, 0.9934, 0.9942, 0.9945, 0.9945
]

# Calculate accumulated seconds
accumulated_seconds = [sum(seconds[:i+1]) for i in range(len(seconds))]

# Creating a single plot with subplots
fig, ax1 = plt.subplots(figsize=(12, 6))

# Set the window title (for Matplotlib version 3.4 and later)
fig.canvas.manager.set_window_title('AMD CPU Performance')

# First axis (accumulated seconds)
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accumulated Seconds', color=color)
ax1.plot(epoch, accumulated_seconds, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Second axis (accuracy) sharing the same x-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(epoch, accuracy, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Title and layout
plt.suptitle('CPU Workload Performance')
plt.title('Ryzen 9 7900X 12 Cores & 64GB RAM')
plt.tight_layout()

# Show the single plot
plt.show()
