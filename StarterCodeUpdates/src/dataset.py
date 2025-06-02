from torch.utils.data import Dataset
import torch

class ClimateSequenceDataset(Dataset):
    def __init__(self, inputs_norm_dask, outputs_dask, seq_len=12, output_is_normalized=True):
        self.seq_len = seq_len

        # Load data into memory
        inputs_np = inputs_norm_dask.compute()
        outputs_np = outputs_dask.compute()

        # Convert to tensors
        self.input_tensors = torch.from_numpy(inputs_np).float()
        self.output_tensors = torch.from_numpy(outputs_np).float()

        # Ensure enough data
        self.size = self.input_tensors.shape[0] - seq_len

        if torch.isnan(self.input_tensors).any() or torch.isnan(self.output_tensors).any():
            raise ValueError("NaN values detected in dataset tensors")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Inputs: sliding window of `seq_len` months
        input_seq = self.input_tensors[idx : idx + self.seq_len]      # [12, C, H, W]
        # Target: the "next" month after the sequence
        target = self.output_tensors[idx + self.seq_len]              # [C_out, H, W]
        return input_seq, target