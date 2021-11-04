from torch.utils.data import Dataset
import torchaudio

import glob

class NSynthDataset(Dataset):
    """Dataset to load NSynth data."""
    
    def __init__(self, audio_dir):
        super().__init__()
        
        self.filenames = glob.glob(audio_dir+"/*.wav")
        _, self.sr = torchaudio.load(self.filenames[0])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        return torchaudio.load(self.filenames[index])[0]
