

# Multi-Agent Reinforcement Learning and Evolutionary Computing for Combinatorial Board Game Optimization

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Pygame-green.svg)](https://www.pygame.org/)
[![Paradigm](https://img.shields.io/badge/paradigm-RL%20%26%20Evolutionary-orange.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)

This repository contains an advanced Artificial Intelligence benchmark suite implemented for the complex combinatorial board game **Hand of the King** (based on the original tabletop design by Bruno Cathala). The project explores the boundaries between model-free **Temporal-Difference Reinforcement Learning (Q-Learning)** and **Global Evolutionary Parameter Optimization (Genetic Algorithms)** when subjected to severe state-space combinatorics, non-linear heuristic spaces, and multi-phase stochastic decision boundaries (Companion Cards).

The core codebase shifts away from simple hand-tuned heuristic baselines toward highly adaptive agents that balance long-term expected returns with immediate multi-tiered tactical advantages.

---

## 🔬 Research Thesis: State-Space Abstraction vs. Exploding Combinatorics

When dealing with a grid-based combinatorial game on a $6 \times 6$ layout with 36 discrete cards partitioned across 7 distinct houses with varying frequencies, tracking the exact grid state configuration results in an unmanageable state-space explosion ($>36!$). This repository serves as an empirical study on mitigating the **Curse of Dimensionality** using two fundamentally distinct algorithmic paradigms:

```text
                                  [ Environment State (6x6 Grid) ]
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
    [ Paradigm 1: Evolutionary Policy ]                        [ Paradigm 2: Tabular Reinforcement Learning ]
                   │                                                           │
        Real-Valued Chromosomes                                   State Abstraction & Compression
    (Global Heuristic Parameter Search)                                        │
                   │                                     ┌─────────────────────┼─────────────────────┐
        ┌──────────┴──────────┐                          ▼                     ▼                     ▼
        ▼                     ▼                  [ Exact Tabular ]     [ Compact State ]     [ 6D Feature Hybrid ]
  [ Static Weight ]   [ Combinatorial Trees ]    Permutation-Invariant   3D Sparse Vector    Dense Topographic Mapping
    Vector Search        Deterministic Look-Ahead    Canonical Banners     Pure MDP Utilities      Shaped Multi-Tiered Rewards

```

### 1. The Reinforcement Learning Frontier

* **Canonical Macro-Structural Mapping (`q_learning_exact.py`)**: Exploits structural symmetry by reducing the grid to an ordered, permutation-invariant canonical form tracking only player and opponent banner distributions. This provides absolute structural tracking at macro levels while discarding localized grid noise.
* **Severe State Aggregation (`q_learning_compact.py`)**: Compresses thousands of physical states into a dense 3D discrete matrix focused solely on immediate capture potential and absolute banner counts. This maximizes policy generalization and forces immediate tabular convergence.
* **Feature-Engineered Approximation (`q_learning_hybrid.py`)**: Projects the board state onto a 6-dimensional topographic feature space, tracking house monopoly thresholds, asset velocities, and rival advantage metrics.

### 2. The Evolutionary Frontier

* **Artificial Chromosome Scaling (`genetic_heuristic.py`)**: Uses a population-driven search to discover optimal scalar parameters across a set of interconnected tactical dimensions (e.g., board flexibility, tactical blocking, and house rarity).
* **Deterministic Combinatorial Tree Search (`generic_offline_training.py`)**: Solves the problem of delayed multi-phase companion interactions by embedding an exact, exhaustive look-ahead tree search inside an evolutionary offline training loop. This eliminates random fallbacks and greedily optimizes complex card mechanics.

---

## 🛠️ Core Architecture & Agent Framework

### 1. Canonical Exact State Q-Learning (`q_learning_exact.py`)

* **Paradigm**: Exact Tabular Q-Learning utilizing macro-structural transitions.
* **State Encoding**: Employs an ordered layout of controlled houses to construct a permutation-invariant string representation:
$$s = \left( \text{sort}(B_{player}), \text{sort}(B_{opponent}) \right)$$


This structure eliminates redundant state configurations for identical asset portfolios.
* **Update Dynamics**: Implements off-policy Bellman bootstrapping:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$


* **Architecture**: Built with a dual-mode pipeline that transitions smoothly between real-time inference loops and a decoupled offline self-play training architecture.

### 2. Minimalist Compact Q-Learning (`q_learning_compact.py`)

* **Paradigm**: Tabular Reinforcement Learning via Extreme State Aggregation.
* **State Mapping**: Maps sparse board conditions into a low-dimensional 3D vector:
$$\phi(s) = \begin{bmatrix} N_{\text{player banners}} & N_{\text{opponent banners}} & I_{\text{banner gain bit}} \end{bmatrix}$$


This aggressive compression forces ultra-fast policy convergence by mapping highly distinct grid states to core strategic primitives.
* **Reward Signal**: Relies on an unshaped environmental reward metric based solely on banner differentials ($\pm 10$ points per banner transition), preventing local sub-optimal convergence caused by heuristic biases.
* **TD Update Isolation**: Utilizes a single-step local Temporal-Difference (TD-0) update rule. By omitting multi-step look-ahead chains, it prevents feature-variance propagation from destabilizing the heavily aggregated state space.

### 3. Feature-Engineered Hybrid Agent (`q_learning_hybrid.py`)

* **Paradigm**: Topographic Feature-Engineered Q-Learning.
* **State Abstraction**: Maps environment states into a dense 6-dimensional feature vector:
1. `h_banner_gain_val`: Instantaneous delta in player banner ownership.
2. `h_secure`: Binary flag indicating if the house acquisition surpasses the strict mathematical monopoly threshold ($>\lfloor N_{\text{house}} / 2 \rfloor$).
3. `h_opp_house_val`: Counter tracking opposing card concentration within the target house.
4. `h_priority`: Base house scarcity factor derived from the maximum initial card distribution.
5. `h_winning`: Absolute margin of banner dominance.
6. `h_companion_val`: End-state trigger checking if the move completely exhausts a house to yield a companion asset.


* **Reward Function**: Implements a comprehensive, shaped reward system:
$$R = 15 \cdot \Delta \mathcal{B} + 8 \cdot \mathbf{1}_{\text{monopoly}} + 4 \cdot \mathbf{1}_{\text{disrupt}} + 5 \cdot \Delta \text{margin}$$


* **Forward Simulation Engine**: Runs an explicit look-ahead forward simulation across the action space to compute $\max_{a'} Q(s', a')$ for true Bellman optimality updates.

### 4. Evolutionary Real-Valued Heuristic (`genetic_heuristic.py`)

* **Paradigm**: Real-Valued Genetic Algorithm for Parameter Optimization.
* **Chromosome Structure**: Models high-level gameplay strategies as a 7-gene dictionary of real-valued scalar weights:
$$\mathbf{w} = \{ \text{banners}, \text{strategic}, \text{block}, \text{rare}, \text{cards}, \text{flexibility}, \text{companion} \}$$


* **Heuristic Scoring Engine**: Evaluates immediate board moves by scaling key game features against the evolved chromosome:
* *Flexibility Metric*: Grants higher scores if Varys rests within the central $4 \times 4$ sub-grid, preserving high directional branching factors for future turns.
* *Tactical Blocking*: Detects and weights moves that interrupt opponent houses with $\geq 2$ cards.
* *Companion Acquisition*: Awards a bonus when a house is completely exhausted to trigger companion card actions.


* **Evolutionary Operators**: Uses a population size of 20 over 50 generations with uniform stochastic crossover and a random gene mutation rate of **0.1**.

### 5. Multi-Phase Evolutionary Combinatorial Tree (`generic_offline_training.py`)

* **Paradigm**: Hybrid Genetic Algorithm integrated with Exact Combinatorial Action Trees.
* **The Companion Phase Problem**: Traditional heuristic models fail when companion cards (e.g., *Jon Snow, Ramsay Bolton, Jaqen H'ghar*) introduce multi-phase action selections. This agent completely eliminates stochastic fallbacks during these complex choice environments.
* **Combinatorial Search Expansion**: When a companion phase triggers, the agent expands the entire valid action-state tree:
$$a^*_{\text{companion}} = \arg\max_{a \in A_{\text{tree}}} f_{\text{eval}}(a \,|\, \mathbf{w}_{\text{optimized}})$$


For high-cardinality splits (e.g., *Ramsay*), it caps the expansion at 20 distinct combinations to control computational overhead.
* **Spatial Concentration Evaluation (`h_house_loc`)**: Tracks rows and columns across the $6 \times 6$ layout to measure the physical concentration of target houses, allowing it to predict and prioritize sweep movements.

---

## 📈 Strategic Exploration & Convergence Mechanics

Rather than relying on fixed heuristic fallbacks or simple exploration constraints, the reinforcement learning models drive autonomous exploration through **Optimistic Initial Values**:

* **Optimistic Bounds**: Unvisited state-action pairs are initialized with an artificially elevated value ($Q_0 = 20.0$ or $Q_0 = 30.0$).
* **Exploration Drive**: This initialization creates an inherent, variance-free curiosity mechanism. The agent is mathematically driven to explore unvisited trajectories via negative TD-error gradients, eliminating the need for rigid $\epsilon$-greedy decay curves during training phases.

---

## 💾 Core Infrastructure Features

### Asynchronous Runtime Persistence Engine

To maintain training continuity across independent runs or long tournament phases, the reinforcement learning agents feature an automated, binary state serialization pipeline:

* Uses high-velocity binary serialization protocols (`pickle`) to back up internal tabular data.
* Integrates directly into the Python runtime interpreter using native `atexit` registries. This guarantees that dynamically updating tabular structures are safely saved to disk upon execution termination without manual checkpoint commands.

---

## 📊 Comparative Paradigm Analysis

| Algorithmic Agent | State Vector Type | Reward Signal Paradigm | Exploration Engine | Multi-Phase Companion Handling |
| --- | --- | --- | --- | --- |
| **Exact Q-Learning** (`q_learning_exact`) | Permutation-Invariant Sorted Tuples | Banner Delta Multiplier ($\times 10$) | $\epsilon$-Greedy Strategy (**0.15**) | Random Sample Selection Loop |
| **Compact Q-Learning** (`q_learning_compact`) | 3D Sparse Aggregation | Pure Environmental Unshaped | Optimistic Initial Value ($Q_0 = 20.0$) | Random Trajectory Sampling |
| **Hybrid Q-Learning** (`q_learning_hybrid`) | 6D Topographic Projection | Shaped Multi-Tiered Rewards | Optimistic Value Gradient ($Q_0 = 30.0$) | Forward Look-Ahead Evaluation |
| **Genetic Heuristic** (`genetic_heuristic`) | Real-Valued 7-Gene Chromosome | Scaled Feature Utility | Uniform Stochastic Mutation (**0.1**) | Spatial Flexibility Metric |
| **Evolutionary Tree** (`generic_offline_training`) | Linear Parameter Weight Matrix | Deterministic Structural Fitness | Generational Evolutionary Search (**0.15**) | Exhaustive Deterministic Tree |

---

## 🚀 Installation & Execution

### Prerequisites

* Python 3.10+
* NumPy
* Pygame

### Repository Setup

```bash
git clone [https://github.com/mehrsamiz/Multi-Agent-boardgame-RL.git](https://github.com/mehrsamiz/Multi-Agent-boardgame-RL.git)
cd Multi-Agent-boardgame-RL

```
