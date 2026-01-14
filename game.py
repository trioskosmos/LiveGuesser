import json
import random
import difflib
import math
import numpy as np
from collections import Counter

class LoveLiveGame:
    def __init__(self, data_path='game_data.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.lives = self.data['lives']
        self.songs = self.data['songs']
        self.artists = self.data['artists']

        # Pre-compute sets for O(1) lookups
        for live in self.lives.values():
            live['song_ids_set'] = set(live['song_ids'])
            live['artist_ids_set'] = set(live['artist_ids'])

        self.live_ids = list(self.lives.keys())
        self.song_ids = list(self.songs.keys())
        self.artist_ids = list(self.artists.keys())

        # Mappings for search
        self.song_name_map = {s['name']: sid for sid, s in self.songs.items()}
        self.artist_name_map = {a['name']: aid for aid, a in self.artists.items()}
        self.live_name_map = {l['name']: lid for lid, l in self.lives.items()}

        # --- Vectorization Setup ---
        self.live_idx_map = {lid: i for i, lid in enumerate(self.live_ids)}
        self.song_idx_map = {sid: i for i, sid in enumerate(self.song_ids)}

        self.num_lives = len(self.live_ids)
        self.num_songs = len(self.song_ids)

        # Matrix: [lives, songs] -> 1 if song in live, 0 otherwise
        self.live_song_matrix = np.zeros((self.num_lives, self.num_songs), dtype=np.float32)

        for i, lid in enumerate(self.live_ids):
            for sid in self.lives[lid]['song_ids_set']:
                if sid in self.song_idx_map:
                    j = self.song_idx_map[sid]
                    self.live_song_matrix[i, j] = 1.0

        self.target_live_id = None
        self.target_live = None

        # Candidates & History (Initialized here for safety, reset in start_game)
        self.possible_live_ids = set(self.live_ids)
        self.guessed_song_ids = set()
        self.guessed_live_ids = set()
        self.history = [] # List of (song_id, artist_id, feedback)

    def start_game(self, target_id=None):
        if target_id and target_id in self.lives:
            self.target_live_id = target_id
        else:
            self.target_live_id = random.choice(self.live_ids)
        self.target_live = self.lives[self.target_live_id]

        # Reset state
        self.possible_live_ids = set(self.live_ids)
        self.guessed_song_ids = set()
        self.guessed_live_ids = set()
        self.history = []

        return self.target_live_id

    def guess_song(self, song_id, artist_id):
        """
        Returns feedback code:
        0: Song NOT in live.
        1: Song in live, but Artist NOT in live.
        2: Song in live AND Artist in live.
        """
        if song_id not in self.songs:
            return -1 # Invalid song

        self.guessed_song_ids.add(song_id)

        # Check if song is in target live
        feedback = 0
        if song_id in self.target_live['song_ids_set']:
            # Song is correct. Check artist.
            if artist_id in self.target_live['artist_ids_set']:
                feedback = 2 # Song & Artist Correct
            else:
                feedback = 1 # Song Correct, Artist Incorrect
        else:
            feedback = 0 # Song Incorrect

        self.history.append((song_id, artist_id, feedback))
        return feedback

    def guess_song_only(self, song_id):
        """
        Used for Song-Only mode.
        Returns: (is_correct, matched_artist_ids)
        """
        if song_id not in self.songs:
            return False, []

        self.guessed_song_ids.add(song_id)

        if song_id in self.target_live['song_ids_set']:
            # Find artists in this live that are associated with this song
            live_artists = self.target_live['artist_ids_set']
            song_artists = set(self.songs[song_id]['artist_ids'])

            # Intersection: Artists in the live who are known to perform this song
            matched = list(live_artists.intersection(song_artists))

            if not matched:
                 matched = list(song_artists)

            # Record in history (Take first matched artist)
            aid = matched[0] if matched else self.artist_ids[0]
            self.history.append((song_id, aid, 2))

            return True, matched
        else:
            # Record failure
            aid = self.artist_ids[0]
            self.history.append((song_id, aid, 0))

            return False, []

    def prune_candidates(self, song_id, artist_id, feedback):
        """
        Updates self.possible_live_ids based on feedback.
        Returns remaining count.
        """
        to_remove = set()
        for lid in self.possible_live_ids:
            live = self.lives[lid]
            has_song = song_id in live['song_ids_set']

            # For artist check, we only check if artist is in live's artist list
            has_artist = artist_id in live['artist_ids_set']

            if feedback == 0:
                # Song NOT in live
                if has_song:
                    to_remove.add(lid)
            elif feedback == 1:
                # Song IN live, Artist NOT in live
                if not has_song:
                    to_remove.add(lid)
                if has_artist:
                    to_remove.add(lid)
            elif feedback == 2:
                # Song IN live, Artist IN live
                if not has_song:
                    to_remove.add(lid)
                if not has_artist:
                    to_remove.add(lid)

        self.possible_live_ids -= to_remove
        return len(self.possible_live_ids)

    def guess_live(self, live_id):
        self.guessed_live_ids.add(live_id)
        return live_id == self.target_live_id

    def find_song_id(self, name):
        if name in self.song_name_map:
            return self.song_name_map[name]
        # Fuzzy search
        matches = difflib.get_close_matches(name, self.song_name_map.keys(), n=1, cutoff=0.6)
        if matches:
            return self.song_name_map[matches[0]]
        return None

    def find_artist_id(self, name):
        if name in self.artist_name_map:
            return self.artist_name_map[name]
        matches = difflib.get_close_matches(name, self.artist_name_map.keys(), n=1, cutoff=0.6)
        if matches:
            return self.artist_name_map[matches[0]]
        return None

    def find_live_id(self, name):
        if name in self.live_name_map:
            return self.live_name_map[name]
        matches = difflib.get_close_matches(name, self.live_name_map.keys(), n=1, cutoff=0.6)
        if matches:
            return self.live_name_map[matches[0]]
        return None

    def get_best_moves(self, top_k=5, candidates_override=None):
        """
        Vectorized calculation of entropy for all songs based on current candidates.
        """
        if candidates_override is not None:
            candidate_ids = list(candidates_override)
        else:
            candidate_ids = list(self.possible_live_ids)

        if not candidate_ids:
            return []

        candidate_indices = [self.live_idx_map[lid] for lid in candidate_ids]

        # Sub-matrix for candidates: [num_candidates, num_songs]
        candidate_matrix = self.live_song_matrix[candidate_indices, :]

        num_candidates = len(candidate_ids)
        if num_candidates <= 1:
            return []

        # P(Song in Live) = Sum(Song in Candidate) / Num Candidates
        # Sum along axis 0 (lives) -> [num_songs]
        p_yes = np.sum(candidate_matrix, axis=0) / num_candidates
        p_no = 1.0 - p_yes

        # Avoid log(0)
        epsilon = 1e-9
        p_yes = np.clip(p_yes, epsilon, 1.0 - epsilon)
        p_no = np.clip(p_no, epsilon, 1.0 - epsilon)

        # Entropy = - p_yes * log2(p_yes) - p_no * log2(p_no)
        entropy = - (p_yes * np.log2(p_yes) + p_no * np.log2(p_no))

        # Zero out already guessed songs (optional, but good for gameplay)
        # Note: If we are evaluating a past move, we shouldn't zero it out!
        # But for 'hint', we should.
        # Let's keep it pure here and filter later if needed, OR only filter if not evaluating.
        # For simplicity, we calculate for ALL songs.

        # Find indices of top K
        # We want HIGHEST entropy
        top_indices = np.argsort(entropy)[::-1]

        results = []
        count = 0
        for idx in top_indices:
            sid = self.song_ids[idx]
            if sid in self.guessed_song_ids and candidates_override is None:
                continue

            results.append((sid, float(entropy[idx])))
            count += 1
            if count >= top_k:
                break

        return results

    def get_song_entropy(self, song_id, candidate_ids):
        """
        Calculate entropy for a specific song given a set of candidates.
        Efficiently uses the matrix.
        """
        if not candidate_ids:
            return 0.0

        candidate_indices = [self.live_idx_map[lid] for lid in candidate_ids]
        s_idx = self.song_idx_map[song_id]

        # Column for this song
        col = self.live_song_matrix[candidate_indices, s_idx]

        yes_count = np.sum(col)
        total = len(candidate_ids)

        if total == 0: return 0.0

        p_yes = yes_count / total
        p_no = 1.0 - p_yes

        if p_yes == 0 or p_no == 0:
            return 0.0

        return - (p_yes * math.log2(p_yes) + p_no * math.log2(p_no))

def play_cli():
    # ... (Keep existing CLI logic for debugging if needed, but app.py is primary)
    pass

if __name__ == "__main__":
    play_cli()
