import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def generate_kde(count_data, width=8, height=6):
    """
    Generates a KDE heatmap as a matplotlib figure for live Streamlit display.
    count_data: list of (x, y, count) tuples
    Returns: matplotlib Figure
    """
    df = pd.DataFrame(count_data, columns=['x', 'y', 'count'])
    fig, ax = plt.subplots(figsize=(width, height))
    if len(df) > 1:
        # Remove duplicate points to avoid singular covariance
        df_unique = df.drop_duplicates(subset=['x', 'y'])
        if len(df_unique) > 1:
            sns.kdeplot(data=df_unique, x='x', y='y', weights='count', fill=True, thresh=0.05, cmap='Reds', ax=ax, warn_singular=False)
    ax.set_title('Heatmap', fontsize=16)
    ax.axis('off')
    return fig