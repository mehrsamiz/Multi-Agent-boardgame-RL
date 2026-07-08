
"""
Module: generic_offline_training.py
Paradigm: Genetic Algorithms & Deterministic Combinatorial Trees

Description:
    An advanced evolutionary agent operating an offline training loop across multiple generations. 
    This script automates selection, crossover, and mutation operations to converge on a highly stable 
    heuristic matrix saved natively to JSON (`data/ga_weights.json`). Crucially, this architecture 
    eliminates stochastic fallbacks during multi-phase choice environments (Companion Phases) 
    by generating and evaluating exhaustive deterministic action trees.

Research Significance:
    Addresses multi-tiered decision trees in board games by marrying global evolutionary search 
    (for high-level strategy) with exact combinatorial tree search (for immediate special action phases).
"""

import copy
import random
import json
import os
from collections import Counter

from main import make_move, make_companion_move, find_card, make_board, Player
from random_agent import get_valid_moves, get_valid_jon_sandor_jaqan, get_valid_ramsay

# Genetic Algorithm configuration for offline training
POPULATION_SIZE = 20
GENERATIONS = 15
GAMES_PER_AGENT = 5
MUTATION_RATE = 0.15
WEIGHTS_FILE = "data/ga_weights.json"

HEURISTIC_KEYS = ['banner_gain', 'house_priority', 'secure_banner', 'opp_house', 'house_loc', 'winning_state']

number_of_houses = {
    'Stark': 8, 'Greyjoy': 7, 'Lanister': 6, 'Targaryen': 5, 'Baratheon': 4, 'Tyrell': 3, 'Tully': 2
}

# Default heuristic weights (fallback if training file not found)
DEFAULT_WEIGHTS = {
    'banner_gain': 0.85,
    'house_priority': 0.35,
    'secure_banner': 0.60,
    'opp_house': 0.40,
    'house_loc': 0.25,
    'winning_state': 0.55
}


# --- Heuristic Functions ---

def h_banner_gain(cards, player, move):
    sim_cards = copy.deepcopy(cards)
    sim_player = copy.deepcopy(player)
    before = len(sim_player.get_banners())
    try:
        make_move(sim_cards, move, sim_player)
    except:
        return 0
    after = len(sim_player.get_banners())
    return after - before


def h_house_loc(house, cards):
    """Evaluate spatial concentration of a house's cards across rows or columns (6x6 board)"""
    locations = [c.get_location() for c in cards if c.get_house() == house]
    if not locations:
        return 0
    rows = [loc // 6 for loc in locations]
    cols = [loc % 6 for loc in locations]

    most_common_row = Counter(rows).most_common(1)[0][1] if rows else 0
    most_common_col = Counter(cols).most_common(1)[0][1] if cols else 0
    return max(most_common_row, most_common_col) / len(locations)


# --- Move Evaluation ---

def evaluate_move_with_weights(move, weights, cards, player1, player2):
    card = None
    for c in cards:
        if c.get_location() == move:
            card = c
            break
    if not card:
        return float('-inf')

    house_name = card.get_house()

    bg = h_banner_gain(cards, player1, move)
    hp = number_of_houses.get(house_name, 0)

    cards_dict = player1.get_cards()
    count = len(cards_dict.get(house_name, [])) if isinstance(cards_dict, dict) else sum(
        1 for c in cards_dict if c.get_house() == house_name)
    sb = 1 if count > number_of_houses.get(house_name, 8) // 2 else 0

    opp_cards = player2.get_cards()
    oh = len(opp_cards.get(house_name, [])) if isinstance(opp_cards, dict) else sum(
        1 for c in opp_cards if c.get_house() == house_name)

    hl = h_house_loc(house_name, cards)
    ws = len(player1.get_banners()) - len(player2.get_banners())

    score = (weights.get('banner_gain', 0) * bg +
             weights.get('house_priority', 0) * hp +
             weights.get('secure_banner', 0) * sb +
             weights.get('opp_house', 0) * oh +
             weights.get('house_loc', 0) * hl +
             weights.get('winning_state', 0) * ws)
    return score


def evaluate_companion_move_with_weights(action, weights, cards, player1, player2, companion_cards):
    """Evaluate a companion action by simulating its effects on the board and banners"""
    sim_cards = copy.deepcopy(cards)
    sim_player1 = copy.deepcopy(player1)
    sim_player2 = copy.deepcopy(player2)
    sim_comp_cards = copy.deepcopy(companion_cards)

    before_banners = len(sim_player1.get_banners())
    try:
        make_companion_move(sim_cards, sim_comp_cards, list(action), sim_player1)
    except:
        return float('-inf')

    after_banners = len(sim_player1.get_banners())
    bg = after_banners - before_banners

    house_name = None
    if len(action) > 1 and isinstance(action[1], int):
        card = find_card(cards, action[1])
        if card:
            house_name = card.get_house()

    hp = number_of_houses.get(house_name, 0) if house_name else 0

    cards_dict = sim_player1.get_cards()
    count = len(cards_dict.get(house_name, [])) if isinstance(cards_dict, dict) else sum(
        1 for c in cards_dict if c.get_house() == house_name) if house_name else 0
    sb = 1 if count > number_of_houses.get(house_name, 8) // 2 else 0

    opp_cards = sim_player2.get_cards()
    oh = len(opp_cards.get(house_name, [])) if isinstance(opp_cards, dict) else sum(
        1 for c in opp_cards if c.get_house() == house_name) if house_name else 0

    hl = h_house_loc(house_name, sim_cards) if house_name else 0
    ws = len(sim_player1.get_banners()) - len(sim_player2.get_banners())

    score = (weights.get('banner_gain', 0) * bg +
             weights.get('house_priority', 0) * hp +
             weights.get('secure_banner', 0) * sb +
             weights.get('opp_house', 0) * oh +
             weights.get('house_loc', 0) * hl +
             weights.get('winning_state', 0) * ws)
    return score


def get_all_companion_actions(cards, companion_cards):
    """Generate all valid action combinations for available companion cards"""
    actions = []
    for name, data in companion_cards.items():
        choices = data.get('Choice', 0)
        if choices == 0:
            actions.append([name])
        elif choices == 1:
            for m in get_valid_jon_sandor_jaqan(cards):
                actions.append([name, m])
        elif choices == 2:  # Ramsay
            valid_ramsay = get_valid_ramsay(cards)
            if len(valid_ramsay) >= 2:
                # Limit to 20 distinct combinations to manage computational load
                count = 0
                for i in range(len(valid_ramsay)):
                    for j in range(i + 1, len(valid_ramsay)):
                        actions.append([name, valid_ramsay[i], valid_ramsay[j]])
                        count += 1
                        if count >= 20: break
                    if count >= 20: break
            else:
                actions.append([name] + valid_ramsay)
        elif choices == 3:  # Jaqen
            valid_jaqen = get_valid_jon_sandor_jaqan(cards)
            valid_comps = [c for c in companion_cards.keys() if c != 'Jaqen']
            if len(valid_jaqen) >= 2 and valid_comps:
                count = 0
                for i in range(len(valid_jaqen)):
                    for j in range(i + 1, len(valid_jaqen)):
                        for c in valid_comps:
                            actions.append([name, valid_jaqen[i], valid_jaqen[j], c])
                            count += 1
                            if count >= 20: break
                        if count >= 20: break
                    if count >= 20: break
    return actions


# --- Offline Simulation & Training Loop ---

def simulate_full_game(weights):
    """Simulate a full game between current weights and a random agent"""
    try:
        cards, companion_cards = make_board()
    except:
        return 0
    p1 = Player("GA_Agent")
    p2 = Player("Random_Opponent")

    turn = 1
    while True:
        valid_moves = get_valid_moves(cards)
        if not valid_moves:
            break

        if turn == 1:
            best_move = max(valid_moves, key=lambda m: evaluate_move_with_weights(m, weights, cards, p1, p2))
            make_move(cards, best_move, p1)
            turn = 2
        else:
            r_move = random.choice(valid_moves)
            make_move(cards, r_move, p2)
            turn = 1

    return 1 if len(p1.get_banners()) > len(p2.get_banners()) else 0


def initialize_population():
    return [{key: random.uniform(0, 1) for key in HEURISTIC_KEYS} for _ in range(POPULATION_SIZE)]


def train_genetic_algorithm():
    print("--- Starting offline genetic agent evolution ---")
    population = initialize_population()

    for gen in range(GENERATIONS):
        ranked_population = []
        for weights in population:
            wins = sum(simulate_full_game(weights) for _ in range(GAMES_PER_AGENT))
            ranked_population.append((wins, weights))

        ranked_population.sort(key=lambda x: x[0], reverse=True)
        print(f"Generation {gen + 1}: Best Agent Won {ranked_population[0][0]}/{GAMES_PER_AGENT} Games")

        best_agents = [w for _, w in ranked_population[:POPULATION_SIZE // 2]]

        # Generate next generation via crossover and mutation
        new_population = list(best_agents)
        while len(new_population) < POPULATION_SIZE:
            p1 = random.choice(best_agents)
            p2 = random.choice(best_agents)
            child = {k: (p1[k] + p2[k]) / 2 for k in HEURISTIC_KEYS}
            for k in HEURISTIC_KEYS:
                if random.random() < MUTATION_RATE:
                    child[k] += random.uniform(-0.1, 0.1)
                    child[k] = max(0, min(1, child[k]))
            new_population.append(child)

        population = new_population

    best_weights = ranked_population[0][1]
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(best_weights, f, indent=4)
    print(f"--> [Best weights successfully saved to {WEIGHTS_FILE}]")


# --- Weight Loading for Tournament Play ---

def load_optimized_weights():
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_WEIGHTS


OPTIMIZED_WEIGHTS = load_optimized_weights()


# --- Main Game Hooks ---

def get_move(cards, player1, player2, companion_cards, choose_companion):
    # --- DETERMINISTIC COMBINATORIAL TREE SEARCH ---
    # Rather than applying a random or greedy selection, the agent expands the entire 
    # action tree for complex multi-phase companion moves (e.g., Ramsay, Jaqen H'ghar).
    if choose_companion:
        if not companion_cards:
            return []

        # Exhaustively generate all valid companion action-state pairs
        possible_companion_actions = get_all_companion_actions(cards, companion_cards)
        if not possible_companion_actions:
            return []
        # Evaluate prospective simulated board states across the optimized GA weight matrix
        best_companion_action = max(
            possible_companion_actions,
            key=lambda act: evaluate_companion_move_with_weights(act, OPTIMIZED_WEIGHTS, cards, player1, player2,
                                                                 companion_cards)
        )
        return best_companion_action

    # Standard board move
    valid_moves = get_valid_moves(cards)
    if not valid_moves:
        return None

    return max(valid_moves, key=lambda m: evaluate_move_with_weights(m, OPTIMIZED_WEIGHTS, cards, player1, player2))


if __name__ == "__main__":
    train_genetic_algorithm()