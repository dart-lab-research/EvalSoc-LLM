import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV data
file_path = 'D:\HaolingX\Role-Playing LLM+Survey\code\Full_Simulation\data_prepro\Data\gptfew4merged.csv'
data = pd.read_csv(file_path)

# Extract the 3rd and 4th columns
data_subset = data.iloc[:, [2, 5]]

# Set a new color palette
sns.set_palette("pastel")

# Create a violin plot
plt.figure(figsize=(12, 8))  # Increase figure size for better visibility
ax = sns.violinplot(data=data_subset, scale='width', alpha=0.7)

# Remove x-axis labels
plt.xticks([])  # Remove x-axis text

# Adjust tick label font size and weight
plt.tick_params(axis='both', which='major', labelsize=16, width=2)  # Larger font size for ticks

# Remove axis and ticks
plt.gca().spines['top'].set_visible(True)  # Keep the top border
plt.gca().spines['right'].set_visible(True)  # Keep the right border
plt.gca().spines['left'].set_visible(True)  # Keep the left border
plt.gca().spines['bottom'].set_visible(True)  # Keep the bottom border

# Remove x and y axis labels
plt.xlabel('')  # Remove x-axis label
plt.ylabel('')  # Remove y-axis label

# Set the figure and axes background color to transparent
plt.gcf().patch.set_facecolor('none')  # Ensure the figure's background is transparent
ax.set_facecolor('none')  # Ensure the axes background is transparent

# Save the plot with a transparent background
plt.savefig('violin_plot_no_x_labels.png', transparent=True, bbox_inches='tight', pad_inches=0)

# Show the plot
plt.show()
