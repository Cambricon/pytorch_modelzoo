import os
from pathlib import Path


class Paths:
    """Manages and configures the paths used by WaveRNN, Tacotron, and the data."""
    def __init__(self, data_path):
        self.base = Path(__file__).parent.parent.expanduser().resolve()

        # Data Paths
        self.data = Path(data_path).expanduser().resolve()
        self.coarse = self.data/'coarse'
        self.fine = self.data/'fine'


        self.create_paths()

    def create_paths(self):
        os.makedirs(self.data, exist_ok=True)
        os.makedirs(self.coarse, exist_ok=True)
        os.makedirs(self.fine, exist_ok=True)



