class MahjongGame {
    constructor() {
        this.gameState = null;
        this.selectedTile = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.updateStatus('Welcome to Mahjong AI! Click "New Game" to start.', 'info');
    }

    bindEvents() {
        document.getElementById('new-game-btn').addEventListener('click', () => this.startNewGame());
        document.getElementById('tsumo-btn').addEventListener('click', () => this.declareTsumo());
    }

    async startNewGame() {
        try {
            const response = await fetch('/api/new_game', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.updateStatus(`New game started! Game ID: ${data.game_id}`, 'success');
                await this.updateGameState();
            } else {
                this.updateStatus(`Error: ${data.error}`, 'error');
            }
        } catch (error) {
            this.updateStatus(`Error starting game: ${error.message}`, 'error');
        }
    }

    async updateGameState() {
        try {
            const response = await fetch('/api/game_state');
            const data = await response.json();
            
            if (response.ok) {
                this.gameState = data;
                this.renderGame();
            } else {
                this.updateStatus(`Error: ${data.error}`, 'error');
            }
        } catch (error) {
            this.updateStatus(`Error updating game state: ${error.message}`, 'error');
        }
    }

    renderGame() {
        if (!this.gameState) return;

        // Check if there's an error in the game state
        if (this.gameState.error) {
            this.updateStatus(`Error: ${this.gameState.error}`, 'error');
            return;
        }

        // Update game info
        document.getElementById('game-id').textContent = `Game ID: ${this.gameState.game_id}`;
        document.getElementById('remaining-tiles').textContent = `Remaining: ${this.gameState.remaining_tiles}`;
        document.getElementById('current-player').textContent = `Current Player: ${this.gameState.current_player + 1}`;
        document.getElementById('remaining-tiles-display').textContent = `Remaining: ${this.gameState.remaining_tiles}`;

        // Update AI players (mapping: backend index → display position)
        // Backend: [0=human, 1=left, 2=top, 3=right] → Display: [1=left, 2=top, 3=right]
        const playerMapping = {1: 0, 2: 1, 3: 2}; // display ID → backend other_hands index
        
        for (let displayId = 1; displayId <= 3; displayId++) {
            const backendIndex = playerMapping[displayId];
            const handData = this.gameState.other_hands[backendIndex];
            const handSize = Array.isArray(handData) ? handData.length : handData;
            document.getElementById(`ai-hand-${displayId}`).textContent = handSize;
            this.renderAIHand(displayId, handData);
        }

        // Update human player
        this.renderHumanHand();
        this.renderAllDiscards();

        // Update controls
        this.updateControls();

        // Update status
        if (this.gameState.game_over) {
            if (this.gameState.winner !== null) {
                const winner = this.gameState.winner;
                const win_type = this.gameState.win_type;
                if (winner === 0) {
                    this.updateStatus(`🎉 Congratulations! You won by ${win_type}!`, 'success');
                } else {
                    this.updateStatus(`Game Over! AI Player ${winner} won by ${win_type}!`, 'error');
                }
            } else {
                this.updateStatus('Game Over! No winner.', 'info');
            }
        } else if (this.gameState.is_human_turn) {
            this.updateStatus('Your turn! Click on a tile to discard it, or click "Declare Tsumo" if you can win.', 'info');
        } else {
            this.updateStatus('AI players are thinking...', 'info');
        }
    }

    renderAIHand(playerId, handData) {
        const handContainer = document.getElementById(`ai-hand-${playerId}-tiles`);
        handContainer.innerHTML = '';

        if (Array.isArray(handData)) {
            // If handData is an array, it's a winning hand, so render the tiles
            handData.forEach(tileStr => {
                const tile = document.createElement('div');
                tile.className = 'tile';
                tile.dataset.value = tileStr;

                const img = document.createElement('img');
                const tileNumber = tileStr.replace('p', '');
                img.src = `/static/images/tiles/Pin${tileNumber}.svg`;
                img.alt = `Pinzu ${tileNumber}`;
                tile.appendChild(img);

                handContainer.appendChild(tile);
            });
        } else {
            // Otherwise, it's the hand size, so render face-down tiles
            for (let i = 0; i < handData; i++) {
                const tile = document.createElement('div');
                tile.className = 'face-down-tile';
                handContainer.appendChild(tile);
            }
        }
    }

    renderHumanHand() {
        const handContainer = document.getElementById('human-hand');
        const handSizeSpan = document.getElementById('human-hand-size');
        
        handContainer.innerHTML = '';
        handSizeSpan.textContent = this.gameState.human_hand.length;

        // Hand is already sorted from backend, just separate newly drawn tile
        let regularHand = [...this.gameState.human_hand];
        let newlyDrawnTile = null;

        if (this.gameState.newly_drawn_tile) {
            // Remove newly drawn tile from regular hand if it exists
            const drawnIndex = regularHand.indexOf(this.gameState.newly_drawn_tile);
            if (drawnIndex !== -1) {
                newlyDrawnTile = regularHand.splice(drawnIndex, 1)[0];
            } else {
                // If not found in hand, it means it's an additional tile
                newlyDrawnTile = this.gameState.newly_drawn_tile;
            }
        }

        // Create a wrapper for the regular hand tiles
        const regularHandWrapper = document.createElement('div');
        regularHandWrapper.className = 'regular-hand-wrapper';
        regularHandWrapper.style.display = 'flex';
        regularHandWrapper.style.gap = '8px';
        
        // Render regular hand tiles in a horizontal row
        regularHand.forEach((tileStr, index) => {
            const tile = document.createElement('div');
            tile.className = 'tile';
            tile.dataset.tile = tileStr;
            tile.dataset.value = tileStr;
            tile.dataset.index = index;
            
            // Create img element for the tile
            const img = document.createElement('img');
            const tileNumber = tileStr.replace('p', '');
            img.src = `/static/images/tiles/Pin${tileNumber}.svg`;
            img.alt = `Pinzu ${tileNumber}`;
            tile.appendChild(img);
            
            tile.addEventListener('click', () => this.selectTile(tile));
            
            regularHandWrapper.appendChild(tile);
        });
        
        handContainer.appendChild(regularHandWrapper);

        // Render newly drawn tile separately on the right with clear separation
        if (newlyDrawnTile) {
            const newlyDrawnElement = document.createElement('div');
            newlyDrawnElement.className = 'tile newly-drawn';
            newlyDrawnElement.dataset.tile = newlyDrawnTile;
            newlyDrawnElement.dataset.value = newlyDrawnTile;
            newlyDrawnElement.dataset.index = regularHand.length;
            
            // Create img element for the newly drawn tile
            const img = document.createElement('img');
            const tileNumber = newlyDrawnTile.replace('p', '');
            img.src = `/static/images/tiles/Pin${tileNumber}.svg`;
            img.alt = `Pinzu ${tileNumber}`;
            newlyDrawnElement.appendChild(img);
            
            newlyDrawnElement.addEventListener('click', () => this.selectTile(newlyDrawnElement));
            
            handContainer.appendChild(newlyDrawnElement);
        }
    }

    renderAllDiscards() {
        // Clear all discard areas
        for (let i = 1; i <= 3; i++) {
            const aiDiscardArea = document.getElementById(`ai-discards-${i}`);
            aiDiscardArea.innerHTML = '';
        }
        const humanDiscardArea = document.getElementById('human-discards');
        humanDiscardArea.innerHTML = '';

        // Render discards for each player
        if (this.gameState.player_discards) {
            // Human player discards (player 0)
            this.renderPlayerDiscards(0, humanDiscardArea);
            
            // AI player discards (mapping display positions to backend players)
            const discardMapping = {1: 1, 2: 2, 3: 3}; // display ID → backend player ID
            for (let displayId = 1; displayId <= 3; displayId++) {
                const backendPlayerId = discardMapping[displayId];
                const aiDiscardArea = document.getElementById(`ai-discards-${displayId}`);
                this.renderPlayerDiscards(backendPlayerId, aiDiscardArea);
            }
        }
    }

    renderPlayerDiscards(playerId, discardArea) {
        if (this.gameState.player_discards && this.gameState.player_discards[playerId]) {
            this.gameState.player_discards[playerId].forEach(tileStr => {
                const tile = document.createElement('div');
                tile.className = 'tile discarded';
                tile.dataset.value = tileStr;
                
                // Create img element for the discarded tile
                const img = document.createElement('img');
                const tileNumber = tileStr.replace('p', '');
                img.src = `/static/images/tiles/Pin${tileNumber}.svg`;
                img.alt = `Pinzu ${tileNumber}`;
                tile.appendChild(img);
                
                discardArea.appendChild(tile);
            });
        }
    }

    renderDiscardedTiles() {
        // This method is now replaced by renderAllDiscards
        this.renderAllDiscards();
    }

    selectTile(tileElement) {
        if (!this.gameState.is_human_turn || this.gameState.game_over) {
            return;
        }

        // Remove previous selection
        const previousSelected = document.querySelector('.tile.selected');
        if (previousSelected) {
            previousSelected.classList.remove('selected');
        }

        // Select new tile
        tileElement.classList.add('selected');
        this.selectedTile = tileElement.dataset.tile;

        // Auto-discard after a short delay
        setTimeout(() => {
            this.discardTile();
        }, 500);
    }

    async discardTile() {
        if (!this.selectedTile || !this.gameState.is_human_turn) {
            return;
        }

        try {
            const response = await fetch('/api/discard', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ tile: this.selectedTile })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.updateStatus(`Discarded ${this.selectedTile}`, 'success');
                this.selectedTile = null;
                await this.updateGameState();
            } else {
                this.updateStatus(`Error: ${data.message}`, 'error');
            }
        } catch (error) {
            this.updateStatus(`Error discarding tile: ${error.message}`, 'error');
        }
    }

    async declareTsumo() {
        if (!this.gameState.is_human_turn || this.gameState.game_over) {
            return;
        }

        try {
            const response = await fetch('/api/tsumo', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.updateStatus('🎉 Tsumo! You won!', 'success');
                await this.updateGameState();
            } else {
                this.updateStatus(`Error: ${data.message}`, 'error');
            }
        } catch (error) {
            this.updateStatus(`Error declaring tsumo: ${error.message}`, 'error');
        }
    }

    updateControls() {
        const tsumoBtn = document.getElementById('tsumo-btn');
        const newGameBtn = document.getElementById('new-game-btn');

        if (this.gameState.game_over) {
            tsumoBtn.disabled = true;
            newGameBtn.disabled = false;
        } else if (this.gameState.is_human_turn) {
            tsumoBtn.disabled = false;
            newGameBtn.disabled = true;
        } else {
            tsumoBtn.disabled = true;
            newGameBtn.disabled = true;
        }
    }

    updateStatus(message, type = 'info') {
        const statusElement = document.getElementById('status-message');
        statusElement.textContent = message;
        statusElement.className = `status-message ${type}`;
    }
}

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.mahjongGame = new MahjongGame();
}); 