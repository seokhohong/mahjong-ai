# Training Data Generator

The `TrainingDataGenerator` class generates training data for the Mahjong AI using MCTS simulations. It's designed to work with both random policies (generation 0) and PQNetwork integration for later generations.

## Features

- **MCTS Integration**: Uses Monte Carlo Tree Search for game simulation
- **Random Policies**: Generation 0 uses random policies for quick data generation
- **PQNetwork Support**: Can integrate with PQNetwork for policy and value estimation
- **Progress Tracking**: Uses tqdm for progress bars during generation
- **Data Persistence**: Saves data in numpy arrays with metadata
- **Intermediate Saving**: Saves only new samples since the previous save (incremental slices) and cleans up intermediate files after the final combined save

## Usage

### Basic Usage (Generation 0 - Random Policies)

```python
from src.training_data_generator import TrainingDataGenerator

# Create a training data generator with random policies
generator = TrainingDataGenerator(
    pq_network=None,  # Use random policies for generation 0
    simulation_count=1000,  # Number of MCTS simulations per move
    exploration_constant=1.414,  # UCB1 exploration parameter
    max_games=1000,  # Maximum number of games to simulate
    save_interval=100  # Save data every N games
)

# Generate training data
features, policies, values = generator.generate_training_data(
    output_file="training_data_gen0.npz"
)
```

### Advanced Usage (With PQNetwork)

```python
from src.training_data_generator import TrainingDataGenerator
from src.core.game import PQNetwork

# Create a PQNetwork
pq_network = PQNetwork(hidden_size=128, embedding_dim=4, max_turns=50)

# Create a training data generator with PQNetwork
generator = TrainingDataGenerator(
    pq_network=pq_network,  # Use PQNetwork for policy/value estimation
    simulation_count=1000,
    exploration_constant=1.414,
    max_games=1000,
    save_interval=100
)

# Generate training data
features, policies, values = generator.generate_training_data(
    output_file="training_data_gen1.npz"
)
```

## Parameters

### TrainingDataGenerator Parameters

- **pq_network** (`Optional[PQNetwork]`): PQNetwork instance for policy/value estimation. If `None`, uses random policies.
- **simulation_count** (`int`): Number of MCTS simulations per move (default: 1000)
- **exploration_constant** (`float`): UCB1 exploration constant (default: 1.414)
- **max_games** (`int`): Maximum number of games to simulate (default: 1000)
- **save_interval** (`int`): Save data every N games (default: 100)

## Output Format

The training data generator outputs three numpy arrays:

1. **features** (`np.ndarray`): Feature matrix with shape `(n_samples, n_features)`
   - For random policies: 102 features (simple encoding)
   - For PQNetwork: 430 features (convolutional encoding)

2. **policies** (`np.ndarray`): Policy labels with shape `(n_samples, 41)`
   - Concatenated multi-head policy: `[action(5) | tile1(18) | tile2(18)]`
   - Each head is a probability distribution that sums to 1.0 per sample

3. **values** (`np.ndarray`): Value labels with shape `(n_samples,)`
   - State values between -1.0 and 1.0

## Data Storage

### File Formats

- **Main data**: `.npz` files (compressed numpy arrays)
- **Metadata**: `.pkl` files (pickled dictionaries)

### File Structure (incremental intermediates)

```
training_data_gen0.npz          # Main training data
training_data_gen0_metadata.pkl  # Metadata
training_data_gen0_intermediate_0_1234.npz   # Slice [start:end) of samples
training_data_gen0_intermediate_1234_2345.npz
...
training_data_gen0.npz                     # Final combined data (intermediates auto-cleaned)
```

### Metadata Structure

```python
metadata = {
    'games_played': 1000,
    'moves_recorded': 28000,
    'feature_shape': (28000, 102),
    'policy_shape': (28000, 41),
    'value_shape': (28000,),
    'simulation_count': 1000,
    'exploration_constant': 1.414,
    'used_pqnetwork': False
}
```

## Feature Extraction

### Simple Features (Random Policies)

When using random policies, the generator extracts simple features:

- **Hand features** (72): One-hot encoding of tiles in hand
- **Visible tiles count** (1): Normalized count of visible tiles
- **Remaining tiles count** (1): Normalized count of remaining tiles
- **Player position** (4): One-hot encoding of player position
- **Called sets count** (1): Normalized count of called sets
- **Can call flag** (1): Whether player can make a call
- **Last discarded tile** (18): One-hot encoding of last discarded tile
- **Last discard player** (4): One-hot encoding of last discard player

**Total**: 102 features

### PQNetwork Features

When using PQNetwork, the generator extracts convolutional features:

- **Hand features** (60): Flattened (12, 5) tensor - 12 tiles, 5 features each
- **Discard features** (320): Flattened (4, 20, 4) tensor - 4 players, 20 turns, 4 features each
- **Game state features** (50): Additional game state information

**Total**: 430 features

## Integration with MCTS

The training data generator integrates with the existing MCTS implementation:

1. **Game Simulation**: Uses `SimpleJong` with `AIPlayer` instances
2. **MCTS Search**: Each player uses MCTS for decision making
3. **Data Recording**: Records game state, policy, and value at each move
4. **Action Application**: Applies actions to game state and continues

## Example Workflow

### Generation 0: Random Policies

```python
# Step 1: Create generator with random policies
generator = TrainingDataGenerator(
    pq_network=None,
    simulation_count=500,
    max_games=1000
)

# Step 2: Generate training data
features, policies, values = generator.generate_training_data(
    output_file="gen0_training_data.npz"
)

# Step 3: Train PQNetwork on the data
# Policies are stored as 41-length vectors [5 | 18 | 18].
# PQNetwork.train accepts either dicts for heads or the concatenated 41-vector.
pq_network = PQNetwork()
training_data = list(zip(features, policies, values))
pq_network.train(training_data, epochs=10)
```

### Generation 1: PQNetwork Integration

```python
# Step 1: Load trained PQNetwork
pq_network = PQNetwork()
pq_network.load_model("gen0_trained_model.keras")

# Step 2: Create generator with PQNetwork
generator = TrainingDataGenerator(
    pq_network=pq_network,
    simulation_count=1000,
    max_games=1000
)

# Step 3: Generate improved training data
features, policies, values = generator.generate_training_data(
    output_file="gen1_training_data.npz"
)
```

## Performance Considerations

### Simulation Count

- **Fast generation**: 100-200 simulations per move
- **Balanced**: 500-1000 simulations per move
- **High quality**: 1000+ simulations per move

### Memory Usage (approximate, float32)

- **Features**: ~size depends on encoder (simple vs PQ)
- **Policies**: ~1.6MB per 10,000 samples (41 entries per sample)
- **Values**: ~0.04MB per 10,000 samples

### Generation Speed

- **Random policies**: ~2-3 games per second (100 simulations)
- **PQNetwork**: ~0.5-1 game per second (1000 simulations)

## Error Handling

The training data generator includes robust error handling:

- **Game failures**: Continues to next game if a game fails
- **Feature extraction**: Falls back to simple features if PQNetwork fails
- **Data recording**: Skips moves that fail to record
- **File saving**: Continues if intermediate saves fail

## Testing

### Run tests

```bash
python -m pytest -q test/
```

Unit tests include generation-0 and PQ-integrated data generation, and PQNetwork functionality. PQ tests are skipped if TensorFlow is not available.

## Policy Space Mapping (Multi-Head)

- **Action head (5)**: `[Discard, Ron, Tsumo, Pon, Chi]`
- **Tile1 head (18)**: tile distribution for single-tile actions (Discard, Pon). Ignored for Ron/Tsumo.
- **Tile2 head (18)**: second tile for Chi only; ignored for other actions.

Scoring examples:
- Discard X: `P(Discard) * P(tile1=X)`
- Ron: `P(Ron)`
- Tsumo: `P(Tsumo)`
- Pon on X: `P(Pon) * P(tile1=X)`
- Chi [A, B]: `P(Chi) * P(tile1=A) * P(tile2=B)`

## Dependencies

- `numpy`: Numerical computing
- `tqdm`: Progress bars
- `tensorflow`: PQNetwork integration (optional)
- `pickle`: Metadata storage

## Future Enhancements

1. **Parallel Processing**: Multi-threaded game generation
2. **Distributed Generation**: Multi-machine data generation
3. **Advanced Features**: More sophisticated feature engineering
4. **Quality Metrics**: Data quality assessment and filtering
5. **Incremental Training**: Continuous data generation and model updates
