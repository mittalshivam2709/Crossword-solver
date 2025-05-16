import os
import pandas as pd
from Solver import solve
import matplotlib.pyplot as plt
N_CROSSWORDS = 100

df = pd.read_csv('../data/crossword_info.csv')
data_folder_path = "../nyt_crossword-master"

sample = df.sample(n=N_CROSSWORDS,random_state=42)
# print(sample)
solution_df = sample.copy()
acc_df = sample.copy()

letter_acc_col = []
word_acc_col = []
solutions_col = []
word_pred_col = []
if __name__ == "__main__":
    count = 0
    for row in sample.iterrows():
        count += 1
        print(f"{count} files complete!")
        filepath = row[-1][-1]
        full_path = os.path.join(data_folder_path, filepath)
        try:
            letter_acc, word_acc, solution, word_pred = solve(full_path)
        except:
            letter_acc, word_acc, solution, word_pred = -1, -1, "WRONG", "WRONG"
            print("Invalid attempt")
        letter_acc_col.append(letter_acc)
        word_acc_col.append(word_acc)
        solutions_col.append(solution)
        word_pred_col.append(word_pred)
    
    # Save and print results
    output_file_path = "output.txt"
    with open(output_file_path, "w") as f:
        for i in range(N_CROSSWORDS):
            print(f"\nCrossword {i + 1}:", file=f)
            print(f"Letter Accuracy: {letter_acc_col[i]}", file=f)
            print(f"Word Accuracy: {word_acc_col[i]}", file=f)
            print(f"Prediction Pairs: {word_pred_col[i]}", file=f)
            
            # Also print to terminal
            print(f"\nCrossword {i + 1}:")
            print(f"Letter Accuracy: {letter_acc_col[i]}")
            print(f"Word Accuracy: {word_acc_col[i]}")
            print(f"Prediction Pairs: {word_pred_col[i]}")

    print(f"\nAll results have been saved to {output_file_path}")

    
    acc_df['PredictionPairs'] = word_pred_col

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, N_CROSSWORDS + 1), letter_acc_col, marker='o', color='blue', label='Letter Accuracy')
    plt.plot(range(1, N_CROSSWORDS + 1), word_acc_col, marker='x', color='green', label='Word Accuracy')
    plt.title("Crossword Solver Accuracy Over Multiple Crosswords")
    plt.xlabel("Crossword Index")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.xticks(range(1, N_CROSSWORDS + 1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_plot.png")  # Saves the figure
    plt.show()




# ===== PLOTTING CODE ENDS HERE =====

