import gradio as gr
import json
import torch
import random
from game import LoveLiveGame
from model import LoveLiveTransformer

# --- Game State Management ---

def init_game():
    game = LoveLiveGame()
    target_id = game.start_game()
    return serialize_game(game), f"Game Started! Guess the live concert."

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

try:
    with open('mappings.json', 'r') as f:
        ai_mappings = json.load(f)

    # Init sizing from mappings (decoupled from game_data.json)
    num_songs = len(ai_mappings['song_to_idx']) + 1
    num_artists = len(ai_mappings['artist_to_idx']) + 1
    num_feedback = 4
    num_lives = len(ai_mappings['live_to_idx'])

    device = torch.device('cpu') # Use CPU for HF Spaces inference usually
    ai_model = LoveLiveTransformer(num_songs, num_artists, num_feedback, num_lives).to(device)
    if torch.cuda.is_available():
        map_loc = torch.device('cuda')
    else:
        map_loc = torch.device('cpu')

    ai_model.load_state_dict(torch.load('transformer_model.pth', map_location=map_loc))
    ai_model.eval()
    print("AI Model Loaded")
except Exception as e:
    print(f"AI Model Load Failed: {e}")
    ai_model = None
    ai_mappings = None

# --- Logic Functions ---

def guess_song(state, song_name, artist_name):
    game = deserialize_game(state)

    sid = game.find_song_id(song_name)
    aid = game.find_artist_id(artist_name)

    if not sid:
        return state, "Song not found.", format_history(game)
    if not aid:
        return state, "Artist not found.", format_history(game)

    if sid in game.guessed_song_ids:
        return state, "Already guessed this song.", format_history(game)

    feedback = game.guess_song(sid, aid)
    game.prune_candidates(sid, aid, feedback)

    msg = ""
    if feedback == 2: msg = "PERFECT MATCH! (Song & Artist correct)"
    elif feedback == 1: msg = "SONG CORRECT! (Artist incorrect)"
    else: msg = "WRONG. (Song not in live)"

    msg += f"\nCandidates remaining: {len(game.possible_live_ids)}"

    return serialize_game(game), msg, format_history(game)

def guess_live(state, live_name):
    game = deserialize_game(state)
    lid = game.find_live_id(live_name)

    if not lid:
        return state, "Live not found.", format_history(game)

    is_correct = game.guess_live(lid)

    if is_correct:
        msg = f"CONGRATULATIONS! You found the live: {game.lives[lid]['name']}"
    else:
        msg = "Incorrect Live."
        if lid in game.possible_live_ids:
            game.possible_live_ids.remove(lid)
        msg += f"\nCandidates remaining: {len(game.possible_live_ids)}"

    return serialize_game(game), msg, format_history(game)

def get_entropy_hint(state):
    game = deserialize_game(state)
    moves = game.get_best_moves(top_k=5)

    if not moves:
        return "No moves available."

    txt = "Top Entropy Suggestions:\n"
    for sid, score in moves:
        txt += f"- {game.songs[sid]['name']} (Score: {score:.4f})\n"
    return txt

def get_ai_prediction(state):
    if not ai_model or not ai_mappings:
        return "AI Model not available."

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

        # Apply mask
        mask = torch.zeros_like(probs)
        live_to_idx = ai_mappings['live_to_idx']
        possible_indices = [live_to_idx[lid] for lid in game.possible_live_ids if lid in live_to_idx]

        if possible_indices:
            mask[possible_indices] = 1.0
            probs = probs * mask
            if probs.sum() > 0:
                probs = probs / probs.sum()

        top_k = torch.topk(probs, k=5)

        txt = "AI Live Predictions:\n"
        for i in range(len(top_k.indices)):
            idx = top_k.indices[i].item()
            prob = top_k.values[i].item()
            if prob < 0.001: continue
            lid = idx_to_live[idx]
            txt += f"{i+1}. {game.lives[lid]['name']} ({prob:.1%})\n"

        return txt

    except Exception as e:
        return f"AI Error: {e}"

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

        with gr.Column(scale=1):
            btn_new = gr.Button("New Game", variant="primary")

            gr.Markdown("### Make a Guess")
            dd_song = gr.Dropdown(choices=all_songs, label="Song Name", filterable=True)
            dd_artist = gr.Dropdown(choices=all_artists, label="Artist Name", filterable=True)
            btn_guess_song = gr.Button("Guess Song")

            gr.Markdown("### Guess Live")
            dd_live = gr.Dropdown(choices=all_lives, label="Live Concert", filterable=True)
            btn_guess_live = gr.Button("Guess Live", variant="stop")

    with gr.Row():
        with gr.Column():
            btn_hint_entropy = gr.Button("Get Entropy Hints")
            hint_output = gr.TextArea(label="Entropy Suggestions", interactive=False)
        with gr.Column():
            btn_hint_ai = gr.Button("Get AI Predictions")
            ai_output = gr.TextArea(label="AI Model Analysis", interactive=False)

    # Event Handlers
    btn_new.click(init_game, inputs=None, outputs=[state, status_output])

    btn_guess_song.click(guess_song,
                         inputs=[state, dd_song, dd_artist],
                         outputs=[state, status_output, history_output])

    btn_guess_live.click(guess_live,
                         inputs=[state, dd_live],
                         outputs=[state, status_output, history_output])

    btn_hint_entropy.click(get_entropy_hint, inputs=[state], outputs=[hint_output])

    btn_hint_ai.click(get_ai_prediction, inputs=[state], outputs=[ai_output])

if __name__ == "__main__":
    demo.launch()
