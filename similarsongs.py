# Similar song search: design an algorithm to search similar songs of an input song

# import necessary libraries
import pandas as pd
from pandastable import Table
import tkinter as tk

# deleted first row from dataset as multiindex data makes no sense in this context

def display_df(df_name):
    root = tk.Tk()

    frame = tk.Frame(root)
    frame.pack(fill='both', expand=True)

    pt = Table(frame, dataframe = df_name)
    pt.show()

    root.mainloop()

class SimilarSongs():
    # Load the dataset
    tracks = pd.read_csv("./Data/tracks.csv", skipinitialspace=True)
    tracks = tracks.loc[:, ~tracks.columns.str.contains('^Unnamed')]

    # Get input
    song_name = input("Please enter a song name (case sensitive): ")
    
    # Find the title(s) in dataset
    found_rows = tracks.loc[tracks['title.1'] == song_name]

    # Get all genres associated with the title(s)
    top_genres = found_rows['genre_top']

    
