import matplotlib.pyplot as plt

# Data
settings = ["Setting 1", "Setting 2", "Setting 3"]
values = [0.1938, 0.2897, 0.2965]
# Create a bar chart
bars = plt.bar(settings, values, color='blue', width = 0.4)

# Add labels and title
plt.ylabel('stAP', rotation=0, labelpad=15)
plt.title('Result')


# Add x-value lines (horizontal gridlines)
for value in plt.yticks()[0]:
    plt.axhline(y=value, color='gray', linestyle='dashed', xmin=0, xmax=1, alpha=0.1)

plt.gca().set_facecolor('lightgray')

# Display the chart
plt.savefig('result.png')