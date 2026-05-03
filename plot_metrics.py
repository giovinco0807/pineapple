import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_metrics():
    csv_path = "ai/training_metrics.csv"
    if not os.path.exists(csv_path):
        print("No metrics data yet.")
        return
        
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("No data in CSV.")
        return
        
    has_breakdown = 'P1_QQ' in df.columns
    n_plots = 4 if has_breakdown else 3
    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 5 * n_plots))
    
    # Plot 1: Score
    axs[0].plot(df['Iteration'], df['P1_Score'], marker='o', color='b', label='P1 (BB) Score')
    axs[0].set_title('Average Score per Game over Iterations')
    axs[0].set_ylabel('Points')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot 2: Bust Rate
    axs[1].plot(df['Iteration'], df['P1_Bust'], marker='o', color='r', linestyle='--', label='P1 Bust %')
    axs[1].plot(df['Iteration'], df['P2_Bust'], marker='x', color='r', linestyle=':', label='P2 Bust %')
    axs[1].set_title('Bust Rate (Foul %)')
    axs[1].set_ylabel('Percentage (%)')
    axs[1].grid(True)
    axs[1].legend()
    
    # Plot 3: FL Rate
    axs[2].plot(df['Iteration'], df['P1_FL'], marker='o', color='g', label='P1 FL %')
    axs[2].plot(df['Iteration'], df['P2_FL'], marker='x', color='g', linestyle='--', label='P2 FL %')
    axs[2].set_title('Fantasyland Entry Rate')
    axs[2].set_ylabel('Percentage (%)')
    axs[2].grid(True)
    axs[2].legend()
    
    if has_breakdown:
        # Plot 4: FL Breakdown for P1 (Stacked Bar)
        # Convert raw counts to percentage of games
        GAMES = 200 # Since each iteration is 200 games
        p1_qq = df['P1_QQ'] / GAMES * 100
        p1_kk = df['P1_KK'] / GAMES * 100
        p1_aa = df['P1_AA'] / GAMES * 100
        p1_trips = df['P1_Trips'] / GAMES * 100
        
        axs[3].bar(df['Iteration'], p1_qq, label='QQ', color='#1f77b4')
        axs[3].bar(df['Iteration'], p1_kk, bottom=p1_qq, label='KK', color='#ff7f0e')
        axs[3].bar(df['Iteration'], p1_aa, bottom=p1_qq+p1_kk, label='AA', color='#2ca02c')
        axs[3].bar(df['Iteration'], p1_trips, bottom=p1_qq+p1_kk+p1_aa, label='Trips', color='#d62728')
        axs[3].set_title('P1 Fantasyland Breakdown (QQ / KK / AA / Trips)')
        axs[3].set_ylabel('Percentage (%)')
        axs[3].set_xlabel('Iteration')
        axs[3].legend()
    else:
        axs[2].set_xlabel('Iteration')
    
    plt.tight_layout()
    out_path = "ai/training_progress.png"
    plt.savefig(out_path)
    
    artifact_path = "C:/Users/Owner/.gemini/antigravity/brain/620aa64f-e5e1-4744-a36b-51bbefa78485/training_progress.png"
    plt.savefig(artifact_path)
    print(f"Plot saved to {out_path} and {artifact_path}")

if __name__ == '__main__':
    plot_metrics()
