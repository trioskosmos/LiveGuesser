import json
import torch
import random
import numpy as np
from model import LoveLiveTransformer
from game import LoveLiveGame
from tqdm import tqdm

def load_model(path, num_songs, num_artists, num_feedback, num_lives, device):
    try:
        model = LoveLiveTransformer(num_songs, num_artists, num_feedback, num_lives).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load model from {path}: {e}")
        return None

def agent_random(game):
    # Pick a random song from all songs
    # To be slightly smarter, pick from songs that are in at least one candidate live?
    # Or purely random from all songs? "Random" implies purely random usually.
    # But let's restrict to "valid" songs (in remaining candidates) to make it a fair baseline for "guessing"?
    # If purely random from ALL songs, it will take forever.
    # Let's pick from relevant songs (union of candidates).

    candidate_songs = set()
    for lid in game.possible_live_ids:
        candidate_songs.update(game.lives[lid]['song_ids_set'])

    if not candidate_songs:
        return random.choice(list(game.songs.keys())), random.choice(list(game.artists.keys()))

    sid = random.choice(list(candidate_songs))
    # Pick random artist
    aid = random.choice(list(game.artists.keys()))
    return sid, aid

def agent_entropy(game):
    best_moves = game.get_best_moves(top_k=1)
    if best_moves:
        sid = best_moves[0][0]
    else:
        sid = random.choice(list(game.songs.keys()))

    # Pick likely artist
    a_ids = game.songs[sid]['artist_ids']
    aid = a_ids[0] if a_ids else list(game.artists.keys())[0]
    return sid, aid

def agent_ai(game, model, song_to_idx, artist_to_idx, idx_to_live, live_to_idx, device):
    # If no history, first guess random (or entropy)
    if not game.history:
        return agent_random(game)

    songs_seq = [song_to_idx[h[0]] + 1 for h in game.history]
    artists_seq = [artist_to_idx[h[1]] + 1 for h in game.history]
    feedbacks_seq = [h[2] + 1 for h in game.history]

    s_in = torch.tensor(songs_seq, device=device).unsqueeze(1)
    a_in = torch.tensor(artists_seq, device=device).unsqueeze(1)
    f_in = torch.tensor(feedbacks_seq, device=device).unsqueeze(1)

    with torch.no_grad():
        logits = model(s_in, a_in, f_in)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    # Mask impossible
    possible_indices = [live_to_idx[lid] for lid in game.possible_live_ids]
    if not possible_indices:
        return agent_random(game) # Should not happen

    mask = torch.zeros_like(probs)
    mask[possible_indices] = 1.0
    probs = probs * mask

    if probs.sum() == 0:
        probs[possible_indices] = 1.0

    # Get top prediction
    top_idx = torch.argmax(probs).item()
    top_live_id = idx_to_live[top_idx]
    top_prob = probs[top_idx].item() / probs.sum().item()

    # Strategy:
    # If confidence > threshold, guess live.
    # Else, guess song that discriminates best (Entropy) or Random?
    # The AI model predicts the Live. It doesn't directly predict the next best song.
    # So the "AI Agent" usually uses the Model to check if it knows the answer,
    # and if not, falls back to Entropy to gather more info.
    # To differentiate "Skill Levels", maybe "Low Skill" doesn't use Entropy for song selection?
    # Or maybe "Low Skill" has a worse model so it guesses Live wrong more often?

    # Let's assume standard strategy:
    # 1. Check Model Confidence.
    # 2. If > 0.5, guess Live.
    # 3. Else, use Entropy to pick song.

    if top_prob > 0.5:
        return "GUESS_LIVE", top_live_id

    return agent_entropy(game)

def run_simulation(agent_type, model, num_games, mappings, device):
    wins = 0
    total_turns = 0

    song_to_idx = mappings['song_to_idx']
    artist_to_idx = mappings['artist_to_idx']
    live_to_idx = mappings['live_to_idx']
    idx_to_live = {v: k for k, v in live_to_idx.items()}

    game = LoveLiveGame()

    for _ in range(num_games):
        game.start_game()
        turns = 0
        solved = False

        while turns < 20:
            turns += 1

            move = None
            if agent_type == 'Random':
                sid, aid = agent_random(game)
                move = ('SONG', sid, aid)
            elif agent_type == 'Entropy':
                # Entropy agent guesses live if 1 candidate remaining
                if len(game.possible_live_ids) == 1:
                    lid = list(game.possible_live_ids)[0]
                    move = ('LIVE', lid)
                else:
                    sid, aid = agent_entropy(game)
                    move = ('SONG', sid, aid)
            elif agent_type.startswith('AI'):
                # AI Agent
                res = agent_ai(game, model, song_to_idx, artist_to_idx, idx_to_live, live_to_idx, device)
                if res[0] == "GUESS_LIVE":
                    move = ('LIVE', res[1])
                else:
                    move = ('SONG', res[0], res[1])

            # Execute Move
            if move[0] == 'LIVE':
                if game.guess_live(move[1]):
                    solved = True
                    break
                else:
                    # Wrong live guess
                    if move[1] in game.possible_live_ids:
                        game.possible_live_ids.remove(move[1])
            else:
                sid, aid = move[1], move[2]
                feedback = game.guess_song(sid, aid)
                game.prune_candidates(sid, aid, feedback)

                # Check if pruned to 1
                if len(game.possible_live_ids) == 1:
                    # Next turn will guess it (or immediate?)
                    # Let's count it as solved next turn to be fair to step count
                    pass

        if solved:
            wins += 1
            total_turns += turns
        else:
            total_turns += 20 # Penalty

    avg_turns = total_turns / num_games
    win_rate = wins / num_games
    return win_rate, avg_turns

def compare():
    with open('mappings.json', 'r') as f:
        mappings = json.load(f)

    num_songs = len(mappings['song_to_idx']) + 1
    num_artists = len(mappings['artist_to_idx']) + 1
    num_feedback = 4
    num_lives = len(mappings['live_to_idx'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Models
    model_low = load_model('transformer_model_low.pth', num_songs, num_artists, num_feedback, num_lives, device)
    model_high = load_model('transformer_model_high.pth', num_songs, num_artists, num_feedback, num_lives, device)

    simulations = [
        ('Random', None),
        ('Entropy', None),
        ('AI Low (50 ep)', model_low),
        ('AI High (100 ep)', model_high)
    ]

    print(f"{'Agent':<20} | {'Win Rate':<10} | {'Avg Turns':<10}")
    print("-" * 46)

    for name, model in simulations:
        win_rate, avg_turns = run_simulation(name, model, 50, mappings, device)
        print(f"{name:<20} | {win_rate:.2%}   | {avg_turns:.2f}")

if __name__ == "__main__":
    compare()
