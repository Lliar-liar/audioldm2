import os
import glob
from typing import Union, Optional, Dict, Any
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import trange
import json
import torch.nn.functional as F
from audioldm2.utilities.data.util import print0
import random
import torchaudio 
import numpy as np
import sys

class AudioWaveformDataset(Dataset):
    """
    A PyTorch Dataset class to load audio waveforms directly from media files (e.g., .mp4).
    This is a simplified version of HyCatVidAudTokDataset, focused solely on providing
    audio data for models like an Audio VAE Tokenizer.
    """
    def __init__(
        self,
        meta_path: str,
        source_dir: str, # Directory containing the original .mp4 files
        audio_waveform_params: dict,
        data_frac: float = 1.0,
        skip_missing_files: bool = True,
        shuffle: bool = False,
    ):
        super().__init__()

        self.source_dir = source_dir
        print0(f"[bold yellow]\[vidtok.data.audio][HyCatAudioWaveformDataset][/bold yellow] Use source media dir: {self.source_dir}")

        self.meta_path = meta_path
        print0(f"[bold yellow]\[vidtok.data.audio][HyCatAudioWaveformDataset][/bold yellow] Use meta path: {self.meta_path}")

        self.audio_waveform_params = audio_waveform_params
        self.data_frac = data_frac
        self.skip_missing_files = skip_missing_files
        self.shuffle = shuffle
        
        self.missing_files = []
        self._load_metadata()

    def _load_metadata(self):
        """Loads the file list from a CSV or by globbing."""
        if self.meta_path is not None and self.meta_path.endswith('.csv'):
            metadata = pd.read_csv(
                self.meta_path,
                on_bad_lines="skip",
                encoding="ISO-8859-1",
                engine="python",
                sep=",",
            )
            self.metadata = metadata
            self.metadata.dropna(inplace=True)
        else:
            # If no CSV, glob all mp4 files in the source directory
            search_path = self.meta_path if self.meta_path and not self.meta_path.endswith('.csv') else self.source_dir
            print0(f"[bold yellow]\[vidtok.data.audio][HyCatAudioWaveformDataset][/bold yellow] Globbing for .mp4 files in {search_path}")
            # Create a DataFrame from the glob result for consistent handling
            files = glob.glob(os.path.join(search_path, '**', '*.mp4'), recursive=True)
            self.metadata = pd.DataFrame({'videos': [os.path.basename(f) for f in files]})

        if self.shuffle:
            print0(f"[bold yellow]\[vidtok.data.audio][HyCatAudioWaveformDataset][/bold yellow] Shuffling metadata ({len(self.metadata)} entries).")
            self.metadata = self.metadata.sample(frac=1).reset_index(drop=True)

    def _get_path(self, sample):
        """
        Get the absolute path for the source media file.
        """
        # Assumes the 'videos' column might have an incorrect extension like .npy
        # We construct the correct path to the .mp4 file.
        rel_path = str(sample["videos"])
        
        # Ensure the path ends with .mp4
        mp4_rel_path = os.path.splitext(rel_path)[0] + ".mp4"

        abs_mp4_fp = os.path.join(self.source_dir, mp4_rel_path)
        return abs_mp4_fp

    def __len__(self):
        return len(self.metadata)

    def normalize_wav(self, waveform):
        waveform = waveform - torch.mean(waveform)
        
        # 2. Peak normalization and scaling
        # Add a small epsilon to avoid division by zero
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        return waveform * 0.5

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        
        abs_mp4_fp = self._get_path(sample)

        # 1. Load Audio Waveform
        if os.path.isfile(abs_mp4_fp):
            audio_waveform = self._load_audio_waveform_from_mp4(abs_mp4_fp)
        else:
            if abs_mp4_fp not in self.missing_files:
                self.missing_files.append(abs_mp4_fp)
            
            if self.skip_missing_files:
                print0(f"[bold yellow]\[vidtok.data.audio][HyCatAudioWaveformDataset][/bold yellow] Warning: missing media file {abs_mp4_fp}. Resampling.")
                return self.__getitem__(random.randint(0, len(self) - 1))
            else:
                raise FileNotFoundError(f"Source media file not found: {abs_mp4_fp}")

        # 2. Prepare the final data dictionary
        final_data = {
            'audio': audio_waveform, 
            "path": abs_mp4_fp,
        }
        

        # 3. Final check for data integrity
        if not torch.all(torch.isfinite(audio_waveform)):
            print0(f"[bold red]FATAL DATA ERROR: NaN or inf detected in waveform from file: {abs_mp4_fp}. Resampling.[/bold red]")
            return self.__getitem__(random.randint(0, len(self) - 1))
        if audio_waveform.dim() != 1:
            print(f"DATA SHAPE ERROR: Waveform dimension is {audio_waveform.dim()} (expected 1) from file: {abs_mp4_fp}. Resampling.")
            return self.__getitem__(random.randint(0, len(self) - 1))

        return final_data

    def _load_audio_waveform_from_mp4(self, mp4_path: str) -> torch.Tensor:
        """
        Loads, resamples, and pads/truncates an audio waveform from a media file.
        """
        target_len = self.audio_waveform_params.get("target_length")
        target_sr = self.audio_waveform_params.get("sampling_rate", 16000)

        try:
            # torchaudio.load can handle various formats including mp4
            waveform, sr = torchaudio.load(mp4_path)
        except Exception as e:
            print0(f"[bold red]Warning: Failed to load audio from '{mp4_path}': {e}. Returning zeros.[/bold red]")
            return torch.zeros((1, target_len))

        # Resample if the sample rate is different
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        # Convert to mono by averaging channels if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad or truncate to the target length
        current_len = waveform.shape[1]
        if current_len > target_len:
            waveform = waveform[:, :target_len]
        elif current_len < target_len:
            padding = torch.zeros((1, target_len - current_len))
            waveform = torch.cat([waveform, padding], dim=1)
        waveform= self.normalize_wav(waveform)
        waveform=waveform.squeeze()
        return waveform