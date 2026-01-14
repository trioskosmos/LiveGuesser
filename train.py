import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import shutil
from model import LoveLiveTransformer

# Configuration
BATCH_SIZE = 32
EPOCHS = 100
CHECKPOINT_EPOCH = 50
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 20
SAMPLES_PER_EPOCH = 1000

class LoveLiveDataset(Dataset):
    def __init__(self, game_data_path='game_data.json', mappings_path='mappings.json', samples=1000):
        with open(game_data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        with open(mappings_path, 'r') as f:
            self.mappings = json.load(f)

        self.lives = self.data['lives']
        self.songs = self.data['songs']
        self.artists = self.data['artists']

        self.live_ids = list(self.lives.keys())
        self.song_ids = list(self.songs.keys())
        self.artist_ids = list(self.artists.keys())

        self.song_to_idx = self.mappings['song_to_idx']
        self.artist_to_idx = self.mappings['artist_to_idx']
        self.live_to_idx = self.mappings['live_to_idx']

        # Precompute sets for faster feedback calculation
        self.live_data_fast = {}
        for lid, ldata in self.lives.items():
            self.live_data_fast[lid] = {
                'song_set': set(ldata['song_ids']),
                'artist_set': set(ldata['artist_ids'])
            }

        self.samples = samples

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        # 1. Pick a random target live
        target_live_id = random.choice(self.live_ids)
        target_live_idx = self.live_to_idx[target_live_id]
        target_data = self.live_data_fast[target_live_id]

        # 2. Simulate a game history
        # Length of history
        seq_len = random.randint(1, MAX_SEQ_LEN)

        song_seq = []
        artist_seq = []
        feedback_seq = []

        # We want a mix of random guesses and "correct" guesses to help it learn positive signals
        # If we only do random guesses, most feedbacks will be 0 (Song incorrect).

        for _ in range(seq_len):
            # 20% chance to pick a correct song from the live
            if random.random() < 0.2 and target_data['song_set']:
                guess_song_id = random.choice(list(target_data['song_set']))
                # 50% chance to pick correct artist given song
                if random.random() < 0.5:
                     # Pick an artist from the live (approximation of correct artist for song)
                     # Ideally we should pick artist associated with song AND in live.
                     # But simple heuristic: pick any artist in live
                     if target_data['artist_set']:
                         guess_artist_id = random.choice(list(target_data['artist_set']))
                     else:
                         guess_artist_id = random.choice(self.artist_ids)
                else:
                     guess_artist_id = random.choice(self.artist_ids)
            else:
                guess_song_id = random.choice(self.song_ids)
                guess_artist_id = random.choice(self.artist_ids)

            # Compute Feedback
            feedback = 0
            if guess_song_id in target_data['song_set']:
                if guess_artist_id in target_data['artist_set']:
                    feedback = 2
                else:
                    feedback = 1
            else:
                feedback = 0

            # Map to indices and shift by 1 (0 is padding)
            if guess_song_id in self.song_to_idx:
                s_idx = self.song_to_idx[guess_song_id] + 1
            else:
                s_idx = 0 # Should not happen if mappings are complete

            if guess_artist_id in self.artist_to_idx:
                a_idx = self.artist_to_idx[guess_artist_id] + 1
            else:
                a_idx = 0

            f_idx = feedback + 1 # 0->1, 1->2, 2->3. Padding is 0.

            song_seq.append(s_idx)
            artist_seq.append(a_idx)
            feedback_seq.append(f_idx)

        # Pad sequences
        pad_len = MAX_SEQ_LEN - seq_len
        song_seq += [0] * pad_len
        artist_seq += [0] * pad_len
        feedback_seq += [0] * pad_len

        return (
            torch.tensor(song_seq, dtype=torch.long),
            torch.tensor(artist_seq, dtype=torch.long),
            torch.tensor(feedback_seq, dtype=torch.long),
            torch.tensor(target_live_idx, dtype=torch.long)
        )

def train():
    print("Loading Data...")
    dataset = LoveLiveDataset(samples=SAMPLES_PER_EPOCH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    with open('mappings.json', 'r') as f:
        mappings = json.load(f)

    num_songs = len(mappings['song_to_idx']) + 1
    num_artists = len(mappings['artist_to_idx']) + 1
    num_feedback = 4 # 0 (pad), 1 (wrong), 2 (song ok), 3 (perfect)
    num_lives = len(mappings['live_to_idx'])

    print(f"Vocab Sizes: Songs={num_songs}, Artists={num_artists}, Lives={num_lives}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = LoveLiveTransformer(num_songs, num_artists, num_feedback, num_lives).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()

    print("Starting Training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for song_seq, artist_seq, feedback_seq, target in dataloader:
            # Transpose to (seq_len, batch_size) for Transformer
            song_seq = song_seq.transpose(0, 1).to(device)
            artist_seq = artist_seq.transpose(0, 1).to(device)
            feedback_seq = feedback_seq.transpose(0, 1).to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(song_seq, artist_seq, feedback_seq)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Acc: {accuracy:.4f}")

        if (epoch + 1) == CHECKPOINT_EPOCH:
            print(f"Saving Low Skill Model (Epoch {CHECKPOINT_EPOCH})...")
            torch.save(model.state_dict(), 'transformer_model_low.pth')

    print("Saving High Skill Model (Epoch {EPOCHS})...")
    torch.save(model.state_dict(), 'transformer_model_high.pth')

    print("Copying High Skill Model to transformer_model.pth...")
    shutil.copy('transformer_model_high.pth', 'transformer_model.pth')

    print("Done!")

if __name__ == "__main__":
    train()
