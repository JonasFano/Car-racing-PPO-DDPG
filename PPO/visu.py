import pandas as pd
import matplotlib.pyplot as plt
from config import Config


def plot_training_data(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Exclude the first entry in the Episode variable
    df = df.iloc[1:]

    plt.figure()
    # Plot the episode scores
    plt.plot(df['Episode'], df['Interval reward'], label='Episode reward')

    # Plot the moving average scores
    # plt.plot(df['Episode'], df['EMA reward'], label='EMA reward')
    # plt.legend()

    # Add labels and legend
    plt.xlabel('Episode')
    plt.ylabel('Episode reward mean')
    plt.grid()
    
    # Show the plot
    # plt.show()
    


if __name__ == "__main__":
    # csv_file = "Training/Logs/" + Config.csv_log_file
    # plot_training_data(csv_file)

    plot_training_data("Training/Logs/training_data_v4.csv")
    # plot_training_data("Training/Logs/training_data_v3.csv")
    # plot_training_data("Training/Logs/training_data_v2.csv")
    # plot_training_data("Training/Logs/training_data_v1.csv")
    # Show the plot
    plt.show()