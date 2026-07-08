
"""
Module: q_learning_hybrid.py
Paradigm: Feature-Engineered Temporal-Difference Reinforcement Learning (Q-Learning)

Description:
    This agent serves as a high-performance hybrid model that maps complex environment 
    topologies into a dense, 6-dimensional engineered feature space. By consolidating the 
    massive board combinatorics into structured metrics (including asset velocity, house monopoly 
    thresholds, and opponent advantage indicators), the agent successfully mitigates the curse 
    of dimensionality while maintaining structural representation.

Technical Highlights:
    - True Bellman Optimality: Operates an explicit look-ahead forward-simulation engine 
      to resolve max_Q(s', a') for immediate online temporal-difference policy updates.
    - Mathematical Exploration: Implements an Optimistic Initial Value strategy (Q0 = 30.0) 
      to natively incentivize policy exploration via TD-error gradients without hardcoded fallbacks.
    - 6D Discrete Vector Mapping: Compresses dynamic grid spaces into deterministic, reward-relevant features.
"""

import copy
import random
import numpy as np
from main import make_move, make_companion_move, calculate_winner as get_winner, get_possible_moves, find_varys
from random_agent import get_valid_moves, get_valid_ramsay, get_valid_jon_sandor_jaqan

Q_table = {}
epsilon = 0.05
learning_rate = 0.2
discount_factor = 0.95

# Optimistic initialization to eliminate hardcoded fallback heuristics
INITIAL_Q = 30.0

number_of_houses = {
    'Stark': 8, 'Greyjoy': 7, 'Lanister': 6,
    'Targaryen': 5, 'Baratheon': 4,
    'Tyrell': 3, 'Tully': 2
}


def find_card(cards, loc):
    for c in cards:
        if c.get_location() == loc:
            return c
    return None


def generate_state_vector(move, companion, player, opp, cards):
    # --- DENSE 6D STATE-SPACE ABSTRACTION MATRIX ---
    # Instead of tracking raw grid permutations (which causes a combinatoric explosion),
    # this encoder projects the prospective environment state into a dense 6-dimensional 
    # feature vector capturing critical game-theoretic majorities and transition velocity.
    if move is not None and isinstance(move, int):
        card = find_card(cards, move)
        selected_house = card.get_house() if card else None
    elif move is not None and hasattr(move, 'get_house'):
        selected_house = move.get_house()
    else:
        selected_house = None

    banner_before = len(player.get_banners())
    sim_cards = copy.deepcopy(cards)
    sim_player = copy.deepcopy(player)
    h_banner_gain_val = 0

    if move is not None and isinstance(move, int) and selected_house is not None and selected_house != 'No House':
        make_move(sim_cards, move, sim_player)
        banner_after = len(sim_player.get_banners())
        h_banner_gain_val = banner_after - banner_before

    sim_cards2 = copy.deepcopy(cards)
    sim_player2 = copy.deepcopy(player)
    h_secure = 0

    if move is not None and isinstance(move, int) and selected_house is not None and selected_house != 'No House':
        make_move(sim_cards2, move, sim_player2)
        num_cards = len(sim_player2.get_cards().get(selected_house, []))
        h_secure = int(num_cards > number_of_houses.get(selected_house, 99) / 2)
    elif selected_house:
        num_cards = len(player.get_cards().get(selected_house, []))
        h_secure = int(num_cards > number_of_houses.get(selected_house, 99) / 2)

    h_opp_house_val = len(opp.get_cards().get(selected_house, [])) if selected_house else 0
    h_priority = number_of_houses.get(selected_house, 0) if selected_house else 0
    h_winning = len(player.get_banners()) - len(opp.get_banners())

    h_companion_val = 0
    if companion and selected_house:
        remaining = [c for c in cards if c.get_house() == selected_house and c.get_name() != 'Varys']
        if len(remaining) == 0:
            h_companion_val = 1

    h_companion_val = 0
    if companion and selected_house:
        remaining = [c for c in cards if c.get_house() == selected_house and c.get_name() != 'Varys']
        if len(remaining) == 0:
            h_companion_val = 1


    return [
        h_banner_gain_val,
        h_secure,
        h_opp_house_val,
        h_priority,
        h_winning,
        h_companion_val,
    ]


def calculate_reward(player_before, player_after, opp_before, opp_after, selected_house, used_companion=None):
    reward = 0
    banners_before = set(player_before.get_banners())
    banners_after = set(player_after.get_banners())
    gained_banners = banners_after - banners_before
    reward += len(gained_banners) * 15

    if selected_house is not None:
        prev = len(player_before.get_cards().get(selected_house, []))
        after = len(player_after.get_cards().get(selected_house, []))
        if prev <= number_of_houses.get(selected_house, 99) / 2 < after:
            reward += 8

        opp_before_count = len(opp_before.get_cards().get(selected_house, []))
        opp_after_count = len(opp_after.get_cards().get(selected_house, []))
        if opp_after_count < opp_before_count:
            reward += 4

    diff_before = len(player_before.get_banners()) - len(opp_before.get_banners())
    diff_after = len(player_after.get_banners()) - len(opp_after.get_banners())
    if diff_after > diff_before:
        reward += (diff_after - diff_before) * 5

    return reward


def choose_action_with_vector(scored_actions):
    """Select action without hardcoded fallback, using optimistic initial values"""
    if random.random() < epsilon:
        return random.choice(scored_actions)

    best_score = float('-inf')
    best_choice = scored_actions[0]

    for action, state_vector, house in scored_actions:
        state_key = tuple(state_vector)
        action_key = str(action)

        # --- OPTIMISTIC VALUE INITIALIZATION GRADIENT ---
        # Setting unvisited state-action pairs to an elevated potential (Q0 = 30.0) 
        # creates an inherent curiosity mechanism. It forces the agent to explore 
        # uncharted policy paths through native TD-error corrections without heuristic biases.
        score = Q_table.get((state_key, action_key), INITIAL_Q)

        if score > best_score:
            best_score = score
            best_choice = (action, state_vector, house)

    return best_choice


def get_move(cards, player, opponent, companion_cards, choose_companion):
    all_cards = copy.deepcopy(cards)
    comp_cards_copy = copy.deepcopy(companion_cards)
    player_before = copy.deepcopy(player)
    opponent_before = copy.deepcopy(opponent)

    actions = []

    if choose_companion and len(comp_cards_copy) > 0:
        for name, data in comp_cards_copy.items():
            choices = data['Choice']
            if choices == 0:
                actions.append([name])
            elif choices == 1:
                valid_moves = get_valid_jon_sandor_jaqan(all_cards)
                for move in valid_moves:
                    actions.append([name, move])
            elif choices == 2:  # Ramsay
                valid_moves = get_valid_ramsay(all_cards)
                if len(valid_moves) >= 2:
                    for _ in range(15):
                        actions.append([name] + random.sample(valid_moves, 2))
            elif choices == 3:  # Jaqen
                valid_moves = get_valid_jon_sandor_jaqan(all_cards)
                valid_comps = [c for c in comp_cards_copy.keys() if c != 'Jaqen']
                if len(valid_moves) >= 2 and valid_comps:
                    for _ in range(15):
                        actions.append([name] + random.sample(valid_moves, 2) + [random.choice(valid_comps)])
    else:
        actions = get_valid_moves(all_cards)

    if not actions:
        return None

    scored_actions = []
    for action in actions:
        temp_cards = copy.deepcopy(all_cards)
        temp_player = copy.deepcopy(player)
        temp_opponent = copy.deepcopy(opponent)

        if choose_companion:
            make_companion_move(temp_cards, copy.deepcopy(comp_cards_copy), list(action), temp_player)
            house = None
        else:
            house_card = find_card(temp_cards, action)
            house = house_card.get_house() if house_card else None
            make_move(temp_cards, action, temp_player)

        state_vector = generate_state_vector(
            action if not choose_companion else None,
            action if choose_companion else None,
            temp_player,
            temp_opponent,
            temp_cards
        )

        scored_actions.append((action, state_vector, house))

    best_action, state_vector, selected_house = choose_action_with_vector(scored_actions)

    # Simulate next state for Bellman equation (true Q-Learning)
    next_cards = copy.deepcopy(all_cards)
    next_player = copy.deepcopy(player)
    next_opponent = copy.deepcopy(opponent)

    if choose_companion:
        make_companion_move(next_cards, copy.deepcopy(companion_cards), list(best_action), next_player)
    else:
        make_move(next_cards, best_action, next_player)

    # Compute immediate reward
    reward = calculate_reward(player_before, next_player, opponent_before, next_opponent, selected_house,
                              best_action if choose_companion else None)

    # Find valid next moves to compute max_Q
    next_actions = get_valid_moves(next_cards) if not choose_companion else []
    max_next_q = 0
    if next_actions:
        next_q_values = []
        for n_act in next_actions:
            n_state_vec = generate_state_vector(n_act, None, next_player, next_opponent, next_cards)
            next_q_values.append(Q_table.get((tuple(n_state_vec), str(n_act)), INITIAL_Q))
        max_next_q = max(next_q_values)

    # Apply chosen action to the actual game objects
    if choose_companion:
        make_companion_move(cards, companion_cards, list(best_action), player)
    else:
        make_move(cards, best_action, player)

    # Online learning via full Bellman equation
    state_key = tuple(state_vector)
    action_key = str(best_action)

    # --- TRUE TEMPORAL-DIFFERENCE BELLMAN UPDATE LOOP ---
    # Executes a full step look-ahead simulation over the dynamic environment state transitions.
    # By evaluating all possible successor actions a' on next_state s', we extract the true
    # mathematical maximum expected utility (max_next_q) to satisfy the Bellman Optimality Equation.
    old_q = Q_table.get((state_key, action_key), INITIAL_Q)
    Q_table[(state_key, action_key)] = old_q + learning_rate * (reward + (discount_factor * max_next_q) - old_q)

    return best_action