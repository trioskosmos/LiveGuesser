import gradio as gr
import json
import torch
import random
import numpy as np
from game import LoveLiveGame
try:
    from model import LoveLiveTransformer
except ImportError:
    LoveLiveTransformer = None

# --- Game State Management ---

def init_game():
    game = LoveLiveGame()
    target_id = game.start_game()
    return serialize_game(game), f"Game Started! Guess the live concert.", ""

def serialize_game(game):
    return {
        'target_live_id': game.target_live_id,
        'possible_live_ids': list(game.possible_live_ids),
        'guessed_song_ids': list(game.guessed_song_ids),
        'guessed_live_ids': list(game.guessed_live_ids),
        'history': game.history
    }

def deserialize_game(state):
    game = LoveLiveGame()
    if not state:
        game.start_game()
        return game

    game.target_live_id = state['target_live_id']
    game.target_live = game.lives[game.target_live_id]
    game.possible_live_ids = set(state['possible_live_ids'])
    game.guessed_song_ids = set(state['guessed_song_ids'])
    game.guessed_live_ids = set(state['guessed_live_ids'])
    game.history = state['history']
    return game

# --- AI Model Loading ---

models = {}
ai_mappings = None
device = torch.device('cpu')

try:
    with open('mappings.json', 'r') as f:
        ai_mappings = json.load(f)

    # Init sizing from mappings
    num_songs = len(ai_mappings['song_to_idx']) + 1
    num_artists = len(ai_mappings['artist_to_idx']) + 1
    num_feedback = 4
    num_lives = len(ai_mappings['live_to_idx'])

    if torch.cuda.is_available():
        map_loc = torch.device('cuda')
    else:
        map_loc = torch.device('cpu')

    if LoveLiveTransformer:
        def load_model(path, name):
            try:
                m = LoveLiveTransformer(num_songs, num_artists, num_feedback, num_lives).to(device)
                m.load_state_dict(torch.load(path, map_location=map_loc))
                m.eval()
                models[name] = m
                print(f"Loaded {name} from {path}")
            except Exception as e:
                print(f"Failed to load {name} from {path}: {e}")

        load_model('transformer_model.pth', 'Default (High)')
        load_model('transformer_model_low.pth', 'Low Skill')
        load_model('transformer_model_high.pth', 'High Skill')
    else:
        raise Exception("LoveLiveTransformer class not available")

except Exception as e:
    print(f"AI Model Initialization Failed: {e}")
    ai_mappings = None

# --- Logic Functions ---

def guess_song(state, song_name, artist_name):
    game = deserialize_game(state)

    sid = game.find_song_id(song_name)
    aid = game.find_artist_id(artist_name)

    if not sid:
        return state, "Song not found.", format_history(game), ""
    if not aid:
        return state, "Artist not found.", format_history(game), ""

    if sid in game.guessed_song_ids:
        return state, "Already guessed this song.", format_history(game), ""

    feedback = game.guess_song(sid, aid)
    game.prune_candidates(sid, aid, feedback)

    msg = ""
    if feedback == 2: msg = "PERFECT MATCH! (Song & Artist correct)"
    elif feedback == 1: msg = "SONG CORRECT! (Artist incorrect)"
    else: msg = "WRONG. (Song not in live)"

    msg += f"\nCandidates remaining: {len(game.possible_live_ids)}"

    return serialize_game(game), msg, format_history(game), ""

def guess_live(state, live_name):
    game = deserialize_game(state)
    lid = game.find_live_id(live_name)

    if not lid:
        return state, "Live not found.", format_history(game), ""

    is_correct = game.guess_live(lid)

    review_txt = ""
    if is_correct:
        msg = f"CONGRATULATIONS! You found the live: {game.lives[lid]['name']}"
        review_txt = generate_game_review(game)
    else:
        msg = "Incorrect Live."
        if lid in game.possible_live_ids:
            game.possible_live_ids.remove(lid)
        msg += f"\nCandidates remaining: {len(game.possible_live_ids)}"

    return serialize_game(game), msg, format_history(game), review_txt

def get_entropy_hint(state):
    game = deserialize_game(state)
    moves = game.get_best_moves(top_k=5)

    if not moves:
        return "No moves available."

    txt = "Top Entropy Suggestions:\n"
    for sid, score in moves:
        txt += f"- {game.songs[sid]['name']} (Score: {score:.4f})\n"
    return txt

def get_ai_prediction(state, skill_level):
    if skill_level == "Random":
        return get_random_prediction(state)

    if not models or not ai_mappings:
        return "AI Model not available."

    model_key = 'Default (High)'
    if skill_level == 'Low (50 epochs)':
        model_key = 'Low Skill'
    elif skill_level == 'High (100 epochs)':
        model_key = 'High Skill'

    ai_model = models.get(model_key)
    if not ai_model:
        return f"Model '{model_key}' not loaded."

    game = deserialize_game(state)
    if not game.history:
        return "Make at least one guess for AI prediction."

    song_to_idx = ai_mappings['song_to_idx']
    artist_to_idx = ai_mappings['artist_to_idx']
    idx_to_live = {v: k for k, v in ai_mappings['live_to_idx'].items()}

    try:
        songs_seq = [song_to_idx[h[0]] + 1 for h in game.history]
        artists_seq = [artist_to_idx[h[1]] + 1 for h in game.history]
        feedbacks_seq = [h[2] + 1 for h in game.history]

        s_in = torch.tensor(songs_seq, device=device).unsqueeze(1)
        a_in = torch.tensor(artists_seq, device=device).unsqueeze(1)
        f_in = torch.tensor(feedbacks_seq, device=device).unsqueeze(1)

        with torch.no_grad():
            logits = ai_model(s_in, a_in, f_in)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        mask = torch.zeros_like(probs)
        live_to_idx = ai_mappings['live_to_idx']
        possible_indices = [live_to_idx[lid] for lid in game.possible_live_ids if lid in live_to_idx]

        if possible_indices:
            mask[possible_indices] = 1.0
            probs = probs * mask
            if probs.sum() > 0:
                probs = probs / probs.sum()

        top_k = torch.topk(probs, k=5)

        txt = f"AI Live Predictions ({skill_level}):\n"
        for i in range(len(top_k.indices)):
            idx = top_k.indices[i].item()
            prob = top_k.values[i].item()
            if prob < 0.001: continue
            lid = idx_to_live[idx]
            txt += f"{i+1}. {game.lives[lid]['name']} ({prob:.1%})\n"

        return txt

    except Exception as e:
        return f"AI Error: {e}"

def get_random_prediction(state):
    game = deserialize_game(state)
    if not game.possible_live_ids:
        return "No possible lives remaining."
    candidates = list(game.possible_live_ids)
    picks = random.sample(candidates, k=min(5, len(candidates)))
    txt = "Random Predictions:\n"
    for i, lid in enumerate(picks):
        txt += f"{i+1}. {game.lives[lid]['name']} (Random)\n"
    return txt

def format_history(game):
    txt = "History:\n"
    for h in game.history:
        s_name = game.songs[h[0]]['name']
        a_name = game.artists[h[1]]['name']
        fb = h[2]
        if fb == 2: res = "PERFECT"
        elif fb == 1: res = "SONG OK"
        else: res = "MISS"
        txt += f"- {s_name} / {a_name}: {res}\n"
    return txt

# --- Review & Simulation ---

def run_agent_simulation(target_id, agent_type):
    # Minimal reproduction of compare_models.py logic for single run
    game = LoveLiveGame()
    game.start_game(target_id)

    song_to_idx = ai_mappings['song_to_idx']
    artist_to_idx = ai_mappings['artist_to_idx']
    live_to_idx = ai_mappings['live_to_idx']
    idx_to_live = {v: k for k, v in live_to_idx.items()}

    model = None
    if agent_type == 'Low': model = models.get('Low Skill')
    elif agent_type == 'High': model = models.get('High Skill')

    turns = 0
    while turns < 20:
        turns += 1

        move = None
        if agent_type == 'Random':
            # Simple Random
            candidate_songs = set()
            for lid in game.possible_live_ids:
                candidate_songs.update(game.lives[lid]['song_ids_set'])
            if not candidate_songs: break
            sid = random.choice(list(candidate_songs))
            aid = random.choice(list(game.artists.keys()))
            move = ('SONG', sid, aid)

        elif agent_type in ['Low', 'High'] and model:
            # AI Logic
            if not game.history:
                candidate_songs = set()
                for lid in game.possible_live_ids:
                    candidate_songs.update(game.lives[lid]['song_ids_set'])
                if not candidate_songs: break
                sid = random.choice(list(candidate_songs))
                aid = random.choice(list(game.artists.keys())) # Simplify artist
                move = ('SONG', sid, aid)
            else:
                songs_seq = [song_to_idx[h[0]] + 1 for h in game.history]
                artists_seq = [artist_to_idx[h[1]] + 1 for h in game.history]
                feedbacks_seq = [h[2] + 1 for h in game.history]

                s_in = torch.tensor(songs_seq, device=device).unsqueeze(1)
                a_in = torch.tensor(artists_seq, device=device).unsqueeze(1)
                f_in = torch.tensor(feedbacks_seq, device=device).unsqueeze(1)

                with torch.no_grad():
                    logits = model(s_in, a_in, f_in)
                    probs = torch.softmax(logits, dim=1).squeeze(0)

                possible_indices = [live_to_idx[lid] for lid in game.possible_live_ids if lid in live_to_idx]
                if not possible_indices: break

                mask = torch.zeros_like(probs)
                mask[possible_indices] = 1.0
                probs = probs * mask
                if probs.sum() > 0: probs = probs / probs.sum()

                top_idx = torch.argmax(probs).item()
                top_prob = probs[top_idx].item()

                if top_prob > 0.5:
                    move = ('LIVE', idx_to_live[top_idx])
                else:
                    # Entropy fallback
                    best = game.get_best_moves(top_k=1)
                    if best:
                        sid = best[0][0]
                        # Pick likely artist
                        a_ids = game.songs[sid]['artist_ids']
                        aid = a_ids[0] if a_ids else list(game.artists.keys())[0]
                        move = ('SONG', sid, aid)
                    else:
                        break

        if not move: break

        if move[0] == 'LIVE':
            if game.guess_live(move[1]):
                return turns
            else:
                if move[1] in game.possible_live_ids:
                    game.possible_live_ids.remove(move[1])
        else:
            sid, aid = move[1], move[2]
            fb = game.guess_song(sid, aid)
            game.prune_candidates(sid, aid, fb)
            if len(game.possible_live_ids) == 1:
                return turns + 1

    return 20

def generate_game_review(game):
    txt = "=== GAME REVIEW ===\n\n"

    # 1. Turn Comparison
    txt += "[Turn Comparison]\n"
    player_turns = len(game.history) + len(game.guessed_live_ids) # Approx
    # Actually just re-simulate or count? history has song guesses. guessed_live_ids has lives.
    # Player moves = song_guesses + live_guesses.
    player_moves_count = len(game.history) + len(game.guessed_live_ids)

    txt += f"Player: {player_moves_count} turns\n"

    # Sim Agents
    if models:
        turns_rand = run_agent_simulation(game.target_live_id, 'Random')
        turns_low = run_agent_simulation(game.target_live_id, 'Low')
        turns_high = run_agent_simulation(game.target_live_id, 'High')

        txt += f"Random Agent: {turns_rand} turns\n"
        txt += f"Low Skill AI: {turns_low} turns\n"
        txt += f"High Skill AI: {turns_high} turns\n"

    txt += "\n[Move Quality Analysis]\n"

    # Replay history to calculate entropy at each step
    replay_game = LoveLiveGame()
    replay_game.start_game(game.target_live_id) # Set target just in case, but we manage candidates manually if needed?
    # Actually we just need to replicate the pruning.

    for i, h in enumerate(game.history):
        sid, aid, fb = h
        song_name = game.songs[sid]['name']

        # Calculate Entropy of ALL songs at this state
        best_moves = replay_game.get_best_moves(top_k=9999) # Get all

        # Find player's rank
        rank = -1
        player_score = 0
        best_score = 0

        for r, (bsid, score) in enumerate(best_moves):
            if r == 0: best_score = score
            if bsid == sid:
                rank = r + 1
                player_score = score
                break

        quality = "Unknown"
        if rank == 1: quality = "Perfect!"
        elif rank <= 5: quality = "Excellent"
        elif rank <= 20: quality = "Good"
        elif rank <= 100: quality = "Okay"
        else: quality = "Suboptimal"

        txt += f"Turn {i+1} ({song_name}):\n"
        txt += f"  Rank: {rank}/{len(best_moves)} | Score: {player_score:.4f} (Best: {best_score:.4f}) -> {quality}\n"

        # Advance state
        replay_game.prune_candidates(sid, aid, fb)
        replay_game.guessed_song_ids.add(sid)

    return txt

# --- UI Construction ---

game_instance = LoveLiveGame()
all_songs = sorted([s['name'] for s in game_instance.songs.values()])
all_artists = sorted([a['name'] for a in game_instance.artists.values()])
all_lives = sorted([l['name'] for l in game_instance.lives.values()])

with gr.Blocks(title="Love Live! Wordle AI") as demo:
    gr.Markdown("# Love Live! Setlist Guessing Game (AI Assisted)")

    state = gr.State()

    with gr.Row():
        with gr.Column(scale=2):
            status_output = gr.Textbox(label="Game Status", value="Press 'New Game' to start!", interactive=False)
            history_output = gr.TextArea(label="Guess History", interactive=False, lines=10)
            review_output = gr.TextArea(label="Game Review (Appears after Win)", interactive=False, lines=10)

        with gr.Column(scale=1):
            btn_new = gr.Button("New Game", variant="primary")

            gr.Markdown("### Make a Guess")
            # OPTIMIZATION: Use filterable=True with full choices
            dd_song = gr.Dropdown(choices=all_songs, label="Song Name", interactive=True, filterable=True)
            dd_artist = gr.Dropdown(choices=all_artists, label="Artist Name", filterable=True)
            btn_guess_song = gr.Button("Guess Song")

            gr.Markdown("### Guess Live")
            dd_live = gr.Dropdown(choices=all_lives, label="Live Concert", interactive=True, filterable=True)
            btn_guess_live = gr.Button("Guess Live", variant="stop")

    with gr.Row():
        with gr.Column():
            btn_hint_entropy = gr.Button("Get Entropy Hints")
            hint_output = gr.TextArea(label="Entropy Suggestions", interactive=False)
        with gr.Column():
            gr.Markdown("### AI Assistant")
            dd_ai_skill = gr.Dropdown(
                choices=["Random", "Low (50 epochs)", "High (100 epochs)"],
                value="High (100 epochs)",
                label="AI Skill Level"
            )
            btn_hint_ai = gr.Button("Get AI Predictions")
            ai_output = gr.TextArea(label="AI Model Analysis", interactive=False)

    # Event Handlers
    btn_new.click(init_game, inputs=None, outputs=[state, status_output, review_output])

    btn_guess_song.click(guess_song,
                         inputs=[state, dd_song, dd_artist],
                         outputs=[state, status_output, history_output, review_output])

    btn_guess_live.click(guess_live,
                         inputs=[state, dd_live],
                         outputs=[state, status_output, history_output, review_output])

    btn_hint_entropy.click(get_entropy_hint, inputs=[state], outputs=[hint_output])

    btn_hint_ai.click(get_ai_prediction, inputs=[state, dd_ai_skill], outputs=[ai_output])

if __name__ == "__main__":
    demo.launch()
