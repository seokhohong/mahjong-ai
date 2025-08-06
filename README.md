# Mahjong AI - Simple Jong

A visual web-based implementation of a simplified Mahjong game using only Pinzu tiles (1-9). Play against 3 AI players in a beautiful, modern interface.

## Features

- **Visual Game Interface**: Modern, responsive web interface with beautiful tile graphics
- **1 Human vs 3 AI Players**: Play against AI opponents using random strategy
- **Real-time Game State**: See your hand, discarded tiles, and other players' hand sizes
- **Interactive Gameplay**: Click tiles to discard them, or declare Tsumo when you can win
- **Extensible Architecture**: Easy to extend with new AI strategies and full Mahjong rules

## Project Structure

```
mahjong-ai/
├── src/                    # Source code
│   ├── core/              # Core game logic
│   │   ├── __init__.py
│   │   └── game.py        # Main game engine and classes
│   ├── web/               # Web interface
│   │   ├── __init__.py
│   │   ├── app.py         # Flask application
│   │   ├── templates/     # HTML templates
│   │   └── static/        # CSS, JS, and static assets
│   ├── __init__.py
│   └── demo_ron.py        # Demo script for Ron functionality
├── test/                  # Test files
│   ├── __init__.py
│   └── test_game.py       # Unit tests for game logic
├── run.py                 # Main entry point
├── run_tests.py           # Test runner script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

1. Start the Flask application:
   ```bash
   python run.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Click "New Game" to start playing!

### Running Tests

Run the test suite:
```bash
python run_tests.py
```

Or run tests directly:
```bash
python -m unittest test.test_game -v
```

### Running Demo

Run the Ron demonstration:
```bash
python src/demo_ron.py
```

## How to Play

### Game Rules (Simplified Mahjong)

- **Objective**: Be the first to form a winning hand with exactly 9 tiles
- **Winning Hand**: Must consist of 3 sets of 3 tiles each:
  - **Triplets**: Three identical tiles (e.g., 3-3-3)
  - **Runs**: Three consecutive tiles (e.g., 1-2-3, 4-5-6)
- **Gameplay**: 
  - Each player starts with 8 tiles
  - On your turn, draw 1 tile and discard 1 tile
  - Declare "Tsumo" when you have a winning hand

### Interface

- **Your Hand**: Displayed at the bottom of the screen
- **AI Players**: Shown at the top, left, and right with hand sizes
- **Discarded Tiles**: Visible in the center area
- **Game Info**: Shows current player, remaining tiles, and game status

### Controls

- **Click a tile**: Select and automatically discard it
- **Declare Tsumo**: Click the "Declare Tsumo" button when you can win
- **New Game**: Start a new game (available when current game ends)

## Architecture

### Backend (Python/Flask)

- `src/core/game.py`: Core game engine and rules
- `src/web/app.py`: Main Flask application with web interface
- `test/test_game.py`: Unit tests for game logic

### Frontend (HTML/CSS/JavaScript)

- `src/web/templates/index.html`: Main game interface
- `src/web/static/css/style.css`: Modern, responsive styling
- `src/web/static/js/game.js`: Interactive game logic

### Key Classes

- `SimpleJong`: Core game engine
- `Player`: Base player class
- `HumanPlayer`: Human player with web interface
- `AIPlayer`: AI player with random strategy
- `GameManager`: Manages game state and web interactions

## Extending the App

### Adding New AI Strategies

1. Create a new AI player class inheriting from `Player`:
   ```python
   class SmartAIPlayer(Player):
       def play(self, game_state):
           # Implement your AI strategy here
           pass
   ``` 
- Improving the user interface
- Adding new game features
- Fixing bugs or improving performance
- Writing additional tests 