
"""
Module: q_learning_exact.py
Paradigm: Canonical Exact State Q-Learning with Dual-Mode Bootstrapping

Description:
    This agent implements an exact tabular Q-learning algorithm tailored for tracking discrete 
    macro-structural transitions. Rather than aggressively compressing the state space, it maps 
    the precise configuration of controlled houses into a permutation-invariant canonical form, 
    effectively exploiting structural symmetries to bound the combinatoric expansion.

Technical Highlights:
    - Permutation-Invariant Encoding: State representation enforces an ordered layout of player 
      and opponent banners, eliminating redundant state allocations for identical sets.
    - Off-Policy Bellman Boostrapping: Implements standard TD(0) learning updates with strict 
      maximization over the next available action frontier to optimize long-term expected returns.
    - Dual-Mode Training Architecture: Features a seamless dual infrastructure combining real-time 
      online inference loop simulations with a standalone, decoupled offline self-play pipeline.
      
Research Significance:
    Serves as an empirical study on the 'Curse of Dimensionality'. By tracking an uncompressed 
    and exact state key space, it provides absolute tactical precision but highlights the necessity 
    of state abstraction models when state cardinality explodes.
"""

import random
import copy
from random_agent import get_valid_moves, get_valid_jon_sandor_jaqan, get_valid_ramsay
from main import make_move, make_companion_move, get_possible_moves
import os
import pickle
import atexit

# Hyperparameters
learning_rate = 0.2
discount_factor = 0.9
epsilon = 0.15
last_state_action = {}

# --- BINARY SERIALIZATION & AUTO-CHECKPOINTING ENGINE ---
# To ensure training continuity, the agent integrates binary persistence via Pickle.
# An automated atexit hook hooks into the Python runtime interpreter to guarantee 
# that the dynamically updating tabular state space is safely written to disk upon termination.

Q_TABLE_FILE = "q_table.pkl"
Q_table = {}

def load_q_table():
    """Load Q-table from pickle binary file"""
    global Q_table
    if os.path.exists(Q_TABLE_FILE):
        try:
            with open(Q_TABLE_FILE, "rb") as f:
                Q_table = pickle.load(f)
            print(f"--> [Q-Table loaded via Pickle. States: {len(Q_table)}]")
        except Exception as e:
            print(f"--> [Error loading pickle file, starting fresh: {e}]")
            Q_table = {}
    else:
        print("--> [No previous memory found. Starting a new Q-table.]")
        Q_table = {}

def save_q_table():
    """Save Q-table to pickle binary file"""
    try:
        with open(Q_TABLE_FILE, "wb") as f:
            pickle.dump(Q_table, f)
        print(f"--> [Q-Table saved via Pickle. States: {len(Q_table)}]")
    except Exception as e:
        print(f"--> [Error saving pickle file: {e}]")

# Load on import, save on exit
load_q_table()
atexit.register(save_q_table)


def get_state_repr(player, opponent):
    """Encode game state as sorted banner tuples for both players"""
    p_banners = tuple(sorted(player.get_banners()))
    o_banners = tuple(sorted(opponent.get_banners()))
    return (p_banners, o_banners)


def choose_action(state, valid_moves):
    """Epsilon-greedy action selection strategy"""
    if random.random() < epsilon:
        return random.choice(valid_moves)
    else:
        q_values = [Q_table.get((state, move), 0) for move in valid_moves]
        max_val = max(q_values)
        best_moves = [valid_moves[i] for i, v in enumerate(q_values) if v == max_val]
        return random.choice(best_moves)


def update_q_value(state, action, reward, next_state, valid_moves):
    """Bellman equation update for Q-learning"""
    max_next_q = max([Q_table.get((next_state, a), 0) for a in valid_moves], default=0)
    old_q = Q_table.get((state, action), 0)
    Q_table[(state, action)] = old_q + learning_rate * (reward + discount_factor * max_next_q - old_q)


def calculate_immediate_reward(player_before, player_after):
    """Compute immediate reward based on banner count change"""
    return (len(player_after.get_banners()) - len(player_before.get_banners())) * 10


def get_move(cards, player1, player2, companion_cards, choose_companion):
    global last_state_action

    current_state = get_state_repr(player1, player2)

    # Companion selection phase
    if choose_companion:
        companions = list(companion_cards.keys())
        if not companions:
            return None

        for _ in range(15):
            companion = random.choice(companions)
            if companion in ['Jon', 'Sandor']:
                jon_moves = get_valid_jon_sandor_jaqan(cards)
                if jon_moves:
                    return [companion, random.choice(jon_moves)]
            elif companion == 'Jaqen':
                jaqen_moves = get_valid_jon_sandor_jaqan(cards)
                if jaqen_moves and len(companion_cards) > 1:
                    other_comps = [c for c in companions if c != 'Jaqen']
                    return ['Jaqen', random.choice(jaqen_moves), random.choice(jaqen_moves), random.choice(other_comps)]
            elif companion == 'Ramsay':
                ramsay_moves = get_valid_ramsay(cards)
                if len(ramsay_moves) >= 2:
                    return ['Ramsay'] + random.sample(ramsay_moves, 2)
                elif ramsay_moves:
                    return ['Ramsay'] + ramsay_moves
            else:
                return [companion]
        return [random.choice(companions)]

    # Standard board move
    valid_moves = get_valid_moves(cards)
    if not valid_moves:
        return None

    chosen_move = choose_action(current_state, valid_moves)

    # Online learning: simulate chosen move to compute reward and next state
    sim_cards = copy.deepcopy(cards)
    sim_player = copy.deepcopy(player1)
    make_move(sim_cards, chosen_move, sim_player)

    reward = calculate_immediate_reward(player1, sim_player)
    next_state = get_state_repr(sim_player, player2)
    next_valid_moves = get_valid_moves(sim_cards)

    update_q_value(current_state, chosen_move, reward, next_state, next_valid_moves)

    return chosen_move


# Standalone training routine
def train_q_agent(episodes=1000):
    from main import make_board, Player, calculate_winner as get_winner

    print(f"Training Q-agent for {episodes} episodes...")
    for ep in range(episodes):
        cards, companion_cards = make_board()
        p1 = Player("q_agent")
        p2 = Player("random")
        turn = 1
        game_over = False

        while not game_over:
            current_player = p1 if turn == 1 else p2
            opponent = p2 if turn == 1 else p1

            valid_moves = get_valid_moves(cards)
            if not valid_moves:
                game_over = True
                continue

            state = get_state_repr(current_player, opponent)

            if turn == 1:
                # Agent's turn: select via policy
                move = choose_action(state, valid_moves)
                player_before = copy.deepcopy(current_player)
                make_move(cards, move, current_player)

                reward = (len(current_player.get_banners()) - len(player_before.get_banners())) * 10

                next_state = get_state_repr(current_player, opponent)
                next_valid_moves = get_valid_moves(cards)
                update_q_value(state, move, reward, next_state, next_valid_moves)
            else:
                # Opponent's turn: random move
                move = random.choice(valid_moves)
                make_move(cards, move, current_player)

            turn = 2 if turn == 1 else 1
    print("Training finished. Q_table size:", len(Q_table))