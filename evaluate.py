import json
import torch
import random
import numpy as np
from model import LoveLiveTransformer
from game import LoveLiveGame

def evaluate():
    print("Loading resources...")
    with open('mappings.json', 'r') as f:
        mappings = json.load(f)

    song_to_idx = mappings['song_to_idx']
    artist_to_idx = mappings['artist_to_idx']
    live_to_idx = mappings['live_to_idx']

    idx_to_live = {v: k for k, v in live_to_idx.items()}
    idx_to_song = {v: k for k, v in song_to_idx.items()}
    idx_to_artist = {v: k for k, v in artist_to_idx.items()}

    game = LoveLiveGame()

    # Model parameters must match train.py (using mappings to match trained model)
    num_songs = len(mappings['song_to_idx']) + 1
    num_artists = len(mappings['artist_to_idx']) + 1
    num_feedback = 4
    num_lives = len(mappings['live_to_idx'])

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = LoveLiveTransformer(num_songs, num_artists, num_feedback, num_lives).to(device)
    model.load_state_dict(torch.load('transformer_model.pth', map_location=device))
    model.eval()

    # Start a simulation
    target_id = game.start_game()
    print(f"Target Live: {game.lives[target_id]['name']}")

    songs_seq = []
    artists_seq = []
    feedbacks_seq = []

    guessed_lives = set()

    max_turns = 20
    solved = False

    all_song_ids = list(game.songs.keys())

    for turn in range(max_turns):
        # Prepare input
        # Pad to max_len (20) used in training, or just use current seq?
        # Model expects (seq_len, batch_size)
        # We can pass current length seq.

        if len(songs_seq) == 0:
            # First turn: random guess or empty input?
            # Model trained on seq_len >= 1.
            # So first guess random.
            # Ideally "optimal" would mean picking a song that splits the space well initially.
            # Let's pick a very common song or just random.
            # Random for diversity.
            guess_song_id = random.choice(all_song_ids)
            # Pick an artist for this song
            artist_candidates = game.songs[guess_song_id]['artist_ids']
            guess_artist_id = random.choice(artist_candidates) if artist_candidates else random.choice(list(game.artists.keys()))

            print(f"Turn {turn+1}: First guess random -> {game.songs[guess_song_id]['name']}")
        else:
            # Use model to predict live
            # Map indices + 1
            s_in = torch.tensor([x + 1 for x in songs_seq], device=device).unsqueeze(1) # (seq_len, 1)
            a_in = torch.tensor([x + 1 for x in artists_seq], device=device).unsqueeze(1)
            f_in = torch.tensor([x + 1 for x in feedbacks_seq], device=device).unsqueeze(1)

            with torch.no_grad():
                logits = model(s_in, a_in, f_in)
                probs = torch.softmax(logits, dim=1).squeeze(0) # (num_lives)

            # Check if model's top choice is invalid (pruned)
            raw_top_idx = torch.argmax(probs).item()
            raw_top_live_id = idx_to_live[raw_top_idx]
            if raw_top_live_id not in game.possible_live_ids:
                 print(f"  [Model Warning] Model wanted to pick {game.lives[raw_top_live_id]['name']} but it is pruned.")

            # Apply hard constraints (pruning)
            # Mask out impossible lives based on game.possible_live_ids
            mask = torch.zeros_like(probs)
            possible_indices = [live_to_idx[lid] for lid in game.possible_live_ids]

            if not possible_indices:
                print("Error: No possible lives remaining according to hard constraints!")
                break

            mask[possible_indices] = 1.0
            probs = probs * mask
            if probs.sum() == 0:
                 # Fallback (shouldn't happen if logic correct)
                 probs[possible_indices] = 1.0
            probs = probs / (probs.sum() + 1e-9)

            # Sort predictions
            sorted_indices = torch.argsort(probs, descending=True)

            top_idx = sorted_indices[0]
            top_live_id = idx_to_live[top_idx.item()]
            top_prob = probs[top_idx]

            print(f"Turn {turn+1}: Top Prediction: {game.lives[top_live_id]['name']} ({top_prob.item():.4f}) [Candidates: {len(possible_indices)}]")

            if top_prob.item() > 0.7 and top_live_id not in guessed_lives:
                # Try guessing the live
                print(">> Guessing LIVE!")
                if game.guess_live(top_live_id):
                    print("CORRECT! Solved.")
                    solved = True
                    break
                else:
                    print("WRONG Live guess. Continuing...")
                    guessed_lives.add(top_live_id)
                    if top_live_id in game.possible_live_ids:
                        game.possible_live_ids.remove(top_live_id)

            # Choose next song: Use Game Engine's Best Move (Entropy)
            # The game engine uses uniform probability over remaining candidates.
            # We can upgrade this to use the model's probabilities?

            # Option A: Use game.get_best_moves() (Pure Entropy on uniform priors)
            # Option B: Use Model Weighted Entropy (similar to what I had, but maybe cleaner?)

            # Let's use the game engine's pure entropy for robustness, as the model
            # can be overconfident or biased. Pure entropy ensures we cut the search space.

            best_moves = game.get_best_moves(top_k=1)

            if best_moves:
                guess_song_id = best_moves[0][0]
                print(f"Guessing Song: {game.songs[guess_song_id]['name']} (Score: {best_moves[0][1]:.4f})")
            else:
                # Fallback if no moves (shouldn't happen if candidates > 1)
                guess_song_id = random.choice(all_song_ids)
                print(f"Guessing Song: {game.songs[guess_song_id]['name']} (Random Fallback)")
            # Pick likely artist for this song
            a_ids = game.songs[guess_song_id]['artist_ids']
            guess_artist_id = a_ids[0] if a_ids else list(game.artists.keys())[0]

        # Execute guess
        feedback = game.guess_song(guess_song_id, guess_artist_id)
        print(f"Feedback: {feedback}")

        # Prune candidates based on feedback
        game.prune_candidates(guess_song_id, guess_artist_id, feedback)

        songs_seq.append(song_to_idx[guess_song_id])
        artists_seq.append(artist_to_idx[guess_artist_id])
        feedbacks_seq.append(feedback)

    if not solved:
        print("Failed to solve in max turns.")

if __name__ == "__main__":
    evaluate()
