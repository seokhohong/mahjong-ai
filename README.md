# Mahjong AI

A simplified Mahjong game implementation with AI players using Monte Carlo Tree Search (MCTS) and neural networks.

## Features

- **Simplified Mahjong Rules**: Pinzu and Souzu tiles (1-9), 4 players
- **MCTS AI**: Advanced AI using Monte Carlo Tree Search with neural network value estimation
- **PQNetwork**: Neural network that outputs both policy and value using TensorFlow/Keras
- **Web Interface**: Interactive web-based game interface
- **Multiple AI Strategies**: Random players and MCTS AI players
- **Training Support**: Neural network integration for value estimation

## Quick Start

### Prerequisites

- Python 3.8+
- Flask
- NumPy
- TensorFlow (for PQNetwork)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mahjong-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the web application:
```bash
python src/web/app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## AI Implementation

### PQNetwork (Policy-Value Network)

The PQNetwork is a sophisticated neural network that outputs both policy (action probabilities) and value (state evaluation) for Mahjong game states. It uses TensorFlow/Keras with a convolutional architecture designed to leverage the spatial and relational structure of the game.

#### Key Features

- **4-dimensional tile embeddings**: Each tile is represented by a 4-dimensional vector
- **Convolutional architecture**: Uses Conv1D layers to process spatial relationships in hands and discard piles
- **Shared weights**: Same convolutional layers process both hands and discard piles for all players
- **Comprehensive input structure**:
  - **Hand features**: (12, 5) tensor - 12 tiles, 5 features (4 embedding + 1 called flag)
  - **Discard pile features**: (max_turns, 4) tensor for each player (4 players total)
  - **Game state features**: 50 additional features
  - **Total**: 6 input tensors (1 hand + 4 discard piles + 1 game state)

#### Architecture

```
Input Layers:
├── Hand Input: (batch_size, 12, 5)
├── Discard Input 0: (batch_size, max_turns, 4)
├── Discard Input 1: (batch_size, max_turns, 4)
├── Discard Input 2: (batch_size, max_turns, 4)
├── Discard Input 3: (batch_size, max_turns, 4)
└── Game State Input: (batch_size, 50)

Processing:
├── Hand Processing:
│   ├── Conv1D(64, kernel_size=3) + BatchNorm + ReLU
│   └── Conv1D(128, kernel_size=3) + BatchNorm + ReLU
│   └── GlobalMaxPooling1D
│
├── Discard Processing (shared for all players):
│   ├── Conv1D(32, kernel_size=3) + BatchNorm + ReLU
│   └── Conv1D(64, kernel_size=3) + BatchNorm + ReLU
│   └── GlobalMaxPooling1D
│
├── Player Feature Combination:
│   ├── Combine hand + discard for each player
│   └── Dense(256) + Dropout(0.3)
│
├── All Players Concatenation:
│   └── Concatenate all player features
│
├── Final Processing:
│   ├── Dense(128) + Dropout(0.3)
│   ├── Dense(64) + Dropout(0.3)
│   ├── Concatenate with game state features
│   └── Dense(64) + Dropout(0.3)
│
└── Output Heads:
    ├── Policy Head: Dense(200, softmax)
    └── Value Head: Dense(1, tanh)
```

#### Usage Example

```python
from core.game import PQNetwork, GamePerspective

# Create a PQNetwork with convolutional architecture
pq_network = PQNetwork(hidden_size=128, embedding_dim=4, max_turns=50)

# Evaluate a game state
game_state = GamePerspective(...)
policy, value = pq_network.evaluate(game_state)

# Get action probabilities
possible_actions = {
    'tsumo': [],
    'ron': [],
    'pon': [],
    'chi': []
}
action_probs = pq_network.get_action_probabilities(game_state, possible_actions)

# Train the network
training_data = [
    (game_state, target_policy, target_value),
    # ... more training examples
]
pq_network.train(training_data, epochs=10, batch_size=32)

# Save and load models
pq_network.save_model('my_model.keras')
pq_network.load_model('my_model.keras')
```

#### Key Advantages

1. **Spatial Awareness**: Convolutional layers capture spatial relationships between tiles in hands and discard piles
2. **Shared Representations**: Same convolutional weights process all players' hands and discards, enabling transfer learning
3. **Structured Input**: Hand and discard data are processed as 2D tensors rather than flat vectors
4. **Scalable**: Architecture can handle variable numbers of tiles and turns
5. **Interpretable**: Convolutional filters can learn meaningful tile patterns and relationships

### MCTS (Monte Carlo Tree Search)

The AI uses MCTS with the following components:

1. **Selection**: UCB1 formula for balancing exploration and exploitation
2. **Expansion**: Adding new child nodes for untried actions
3. **Simulation**: Random playouts to estimate state values
4. **Backpropagation**: Updating node statistics with simulation results

### Neural Network Integration

- **PQNetwork**: Policy-value network used by `AIPlayer` when TensorFlow is available
- **Value Estimation**: MCTS simulation results may be blended with PQNetwork value predictions

### AIPlayer Class

```python
from core.game import AIPlayer

# Create an AI player with custom parameters
ai_player = AIPlayer(
    player_id=0,
    simulation_count=1000,  # Number of MCTS simulations
    exploration_constant=1.414  # UCB1 exploration parameter
)
```

## Game Rules

### Simplified Mahjong

- **Objective**: Be the first to form a winning hand with exactly 12 tiles
- **Winning Hand**: Must consist of 4 sets of 3 tiles each:
  - **Triplets**: Three identical tiles (e.g., 3-3-3)
  - **Runs**: Three consecutive tiles (e.g., 1-2-3, 4-5-6)
- **Gameplay**: 
  - Each player starts with 11 tiles
  - On your turn, draw 1 tile and discard 1 tile
  - Declare "Tsumo" when you have a winning hand
  - Declare "Ron" when you can win with another player's discard

### Actions

- **Tsumo**: Declare win with current hand
- **Ron**: Declare win with discarded tile
- **Discard**: Discard a tile from hand
- **Pon**: Call to complete a triplet
- **Chi**: Call to complete a sequence (only from left player)

## Usage Examples

### Basic Game

```python
from core.game import SimpleJong, AIPlayer, Player

# Create players
players = [
    AIPlayer(0, simulation_count=500),  # MCTS AI
    Player(1),  # Random player
    Player(2),  # Random player
    Player(3),  # Random player
]

# Create and play game
game = SimpleJong(players)
winner = game.play_round()
print(f"Player {winner} wins!")
```

### PQNetwork Demo

```bash
python src/demo_pq_network.py
```

### MCTS Demo

```bash
python src/demo_mcts.py
```

### Running Tests

```bash
python test/test_pq_network.py
python test/test_mcts_ai.py
```

## Architecture

### Core Components

- `SimpleJong`: Main game engine
- `Player`: Base player class
- `AIPlayer`: MCTS AI implementation
- `MCTSNode`: MCTS tree node
- `PQNetwork`: Policy-value neural network

### Key Classes

- **PQNetwork**: Policy-value neural network
  - `evaluate()`: Get policy and value for a game state
  - `get_action_probabilities()`: Get action probabilities
  - `train()`: Train the network on data
  - `save_model()` / `load_model()`: Model persistence

- **MCTSNode**: Represents a node in the MCTS tree
  - `select_child()`: UCB1 selection
  - `expand()`: Add new child nodes
  - `simulate()`: Random playout
  - `backpropagate()`: Update statistics

- **AIPlayer**: MCTS AI player
  - `play()`: Main decision method
  - `_mcts_search()`: MCTS algorithm
  - `_create_game_copy()`: State copying for simulations

## Configuration

### MCTS Parameters

- `simulation_count`: Number of MCTS simulations (default: 1000)
- `exploration_constant`: UCB1 exploration parameter (default: 1.414)

### PQNetwork Parameters

- `hidden_size`: Hidden layer size (default: 128)
- `embedding_dim`: Tile embedding dimension (default: 4)

## Performance

The MCTS AI performance depends on:

1. **Simulation Count**: More simulations = better decisions (but slower)
2. **Exploration Constant**: Higher values = more exploration
3. **Neural Network**: Provides value estimation for non-terminal states

### Recommended Settings

- **Fast Play**: 100-200 simulations
- **Balanced**: 500-1000 simulations  
- **High Quality**: 1000+ simulations

## Extending the AI

### Adding New Features

1. **Feature Engineering**: Modify `_extract_features()` in PQNetwork
2. **Neural Network Architecture**: Update `_build_model()` method
3. **MCTS Enhancements**: Add domain-specific knowledge to simulation

### Training the PQNetwork

To train the PQNetwork:

1. Collect game data (states and outcomes)
2. Prepare training data in the format: `(game_state, target_policy, target_value)`
3. Use the `train()` method
4. Save trained models with `save_model()`

Example training:
```python
# Collect training data from games
training_data = []
for game_state, target_policy, target_value in collected_data:
    training_data.append((game_state, target_policy, target_value))

# Train the network
pq_network.train(training_data, epochs=10, batch_size=32)
pq_network.save_model('trained_model.keras')
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License. 