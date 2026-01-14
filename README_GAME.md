# LoveLive! Wordle Game & AI

This project implements a Wordle-style game where you guess a specific LoveLive! live concert based on songs and artists. It also includes a Transformer model trained to play this game.

## Prerequisites

Install dependencies:
```bash
pip install torch numpy tqdm
```

## Setup

1.  **Preprocessing**: Combine the raw JSON data into a game-friendly format.
    ```bash
    python preprocess.py
    ```
    This creates `game_data.json`.

## Playing the Game

Run the game script to play interactively:
```bash
python game.py
```
- Select `[S]` to guess a Song (and Artist).
- Select `[L]` to guess the Live concert directly.
- Select `[Q]` to give up and see the answer.

## Training the AI

To train the Transformer model:
```bash
python train.py
```
This will:
- Generate mappings (`mappings.json`).
- Train the model on synthetic game episodes.
- Save the model to `transformer_model.pth`.

## Evaluating the AI

To watch the AI play a game:
```bash
python evaluate.py
```
The AI uses the trained model to predict the live and selects songs to narrow down the possibilities.
