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
        document.getElementById('play-game-btn').addEventListener('click', () => this.startPlayGame());
        document.getElementById('watch-game-btn').addEventListener('click', () => this.startWatchGame());
    }

    async startPlayGame() {
        try {
            const response = await fetch('/api/play_game', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.updateStatus(`New play game started! Game ID: ${data.game_id}`, 'success');
                await this.updateGameState();
            } else {
                this.updateStatus(`Error: ${data.error}`, 'error');
            }
        } catch (error) {
            this.updateStatus(`Error starting play game: ${error.message}`, 'error');
        }
    }

    async startWatchGame() {
        try {
            const response = await fetch('/api/watch_game', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.updateStatus(`New watch game started! Game ID: ${data.game_id}`, 'success');
                await this.updateGameState();
                // In watch mode, start the AI sequence immediately
                await this.playAISequence();
            } else {
                this.updateStatus(`Error: ${data.error}`, 'error');
            }
        } catch (error) {
            this.updateStatus(`Error starting watch game: ${error.message}`, 'error');
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

        // Handle different game modes
        if (this.gameState.game_mode === 'watch') {
            // Watch mode: all players are AI
            this.renderWatchMode();
        } else {
            // Play mode: human player + 3 AI players
            this.renderPlayMode();
        }

        // Update action buttons
        this.updateActionButtons();

        // Update status
        if (this.gameState.game_over) {
            if (this.gameState.winner !== null) {
                const winner = this.gameState.winner;
                const win_type = this.gameState.win_type;
                if (this.gameState.game_mode === 'play' && winner === 0) {
                    this.updateStatus(`🎉 Congratulations! You won by ${win_type}!`, 'success');
                } else {
                    this.updateStatus(`Game Over! AI Player ${winner} won by ${win_type}!`, 'error');
                }
            } else {
                this.updateStatus('Game Over! No winner.', 'info');
            }
        } else if (this.gameState.is_human_turn) {
            this.updateStatus('Your turn! Click on a tile to discard it.', 'info');
        } else {
            this.updateStatus('AI players are thinking...', 'info');
        }
    }

    renderWatchMode() {
        // In watch mode, show all AI players' hands
        // Backend sends: [AI0, AI1, AI2, AI3] in other_hands
        // Display positions: 1=left, 2=top, 3=right, 0=bottom
        const playerMapping = {0: 0, 1: 1, 2: 2, 3: 3}; // backend index → display ID
        
        for (let backendIndex = 0; backendIndex < 4; backendIndex++) {
            const displayId = playerMapping[backendIndex];
            const handData = this.gameState.other_hands[backendIndex];
            this.renderAIHand(displayId, handData);
        }

        // Update player labels for watch mode
        document.getElementById('top-player-label').textContent = 'AI Player 2';
        document.getElementById('left-player-label').textContent = 'AI Player 1';
        document.getElementById('right-player-label').textContent = 'AI Player 3';
        document.getElementById('bottom-player-label').textContent = 'AI Player 0';

        // Show AI player 0 elements and hide human elements
        const humanHandContainer = document.getElementById('human-hand');
        const aiHand0Container = document.getElementById('ai-hand-0-tiles');
        const humanDiscardsContainer = document.getElementById('human-discards');
        const aiDiscards0Container = document.getElementById('ai-discards-0');
        const humanCalledSetsContainer = document.getElementById('human-called-sets');
        const aiCalledSets0Container = document.getElementById('ai-called-sets-0');

        humanHandContainer.style.display = 'none';
        aiHand0Container.style.display = 'flex';
        humanDiscardsContainer.style.display = 'none';
        aiDiscards0Container.style.display = 'block';
        humanCalledSetsContainer.style.display = 'none';
        aiCalledSets0Container.style.display = 'block';

        // Show all discards and called sets
        this.renderAllDiscards();
        this.renderCalledSets();
    }

    renderPlayMode() {
        // Play mode: show 3 AI players (mapping: backend index → display position)
        // Backend: [0=human, 1=left, 2=top, 3=right] → Display: [1=left, 2=top, 3=right]
        const playerMapping = {1: 0, 2: 1, 3: 2}; // display ID → backend other_hands index
        
        for (let displayId = 1; displayId <= 3; displayId++) {
            const backendIndex = playerMapping[displayId];
            const handData = this.gameState.other_hands[backendIndex];
            this.renderAIHand(displayId, handData);
        }

        // Update player labels for play mode
        document.getElementById('top-player-label').textContent = 'AI Player 2 (Toimen)';
        document.getElementById('left-player-label').textContent = 'AI Player 1 (Kamicha)';
        document.getElementById('right-player-label').textContent = 'AI Player 3 (Shimocha)';
        document.getElementById('bottom-player-label').textContent = 'You (Human)';

        // Show human player elements and hide AI player 0 elements
        const humanHandContainer = document.getElementById('human-hand');
        const aiHand0Container = document.getElementById('ai-hand-0-tiles');
        const humanDiscardsContainer = document.getElementById('human-discards');
        const aiDiscards0Container = document.getElementById('ai-discards-0');
        const humanCalledSetsContainer = document.getElementById('human-called-sets');
        const aiCalledSets0Container = document.getElementById('ai-called-sets-0');

        humanHandContainer.style.display = 'flex';
        aiHand0Container.style.display = 'none';
        humanDiscardsContainer.style.display = 'block';
        aiDiscards0Container.style.display = 'none';
        humanCalledSetsContainer.style.display = 'block';
        aiCalledSets0Container.style.display = 'none';

        // Show human player
        this.renderHumanHand();

        // Show all discards and called sets
        this.renderAllDiscards();
        this.renderCalledSets();
    }
    
    getTileImage(tileStr) {
        const suit = tileStr.slice(-1);
        const number = tileStr.slice(0, -1);
        let suitName = '';
        if (suit === 'p') {
            suitName = 'Pin';
        } else if (suit === 's') {
            suitName = 'Sou';
        }
        return `/static/images/tiles/${suitName}${number}.svg`;
    }

    renderAIHand(playerId, handData) {
        // In watch mode, playerId is 1-4, in play mode it's 1-3
        const handContainer = document.getElementById(`ai-hand-${playerId}-tiles`);
        if (!handContainer) {
            console.warn(`Hand container not found for player ${playerId}`);
            return;
        }
        
        handContainer.innerHTML = '';

        if (Array.isArray(handData)) {
            // If handData is an array, it's a winning hand, so render the tiles
            handData.forEach(tileStr => {
                const tile = document.createElement('div');
                tile.className = 'tile';
                tile.dataset.value = tileStr;

                const img = document.createElement('img');
                img.src = this.getTileImage(tileStr);
                img.alt = tileStr;
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
        
        handContainer.innerHTML = '';

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
            img.src = this.getTileImage(tileStr);
            img.alt = tileStr;
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
            img.src = this.getTileImage(newlyDrawnTile);
            img.alt = newlyDrawnTile;
            newlyDrawnElement.appendChild(img);
            
            newlyDrawnElement.addEventListener('click', () => this.selectTile(newlyDrawnElement));
            
            handContainer.appendChild(newlyDrawnElement);
        }
    }

    renderAllDiscards() {
        // Clear all discard areas
        for (let i = 0; i <= 3; i++) {
            const aiDiscardArea = document.getElementById(`ai-discards-${i}`);
            if (aiDiscardArea) {
                aiDiscardArea.innerHTML = '';
            }
        }
        const humanDiscardArea = document.getElementById('human-discards');
        if (humanDiscardArea) {
            humanDiscardArea.innerHTML = '';
        }

        // Render discards for each player
        if (this.gameState.player_discards) {
            if (this.gameState.game_mode === 'watch') {
                // Watch mode: render all 4 AI players
                // Backend: [AI0, AI1, AI2, AI3] → Display: [0=bottom, 1=left, 2=top, 3=right]
                const discardMapping = {0: 0, 1: 1, 2: 2, 3: 3}; // backend index → display ID
                for (let backendIndex = 0; backendIndex < 4; backendIndex++) {
                    const displayId = discardMapping[backendIndex];
                    const discardArea = document.getElementById(`ai-discards-${displayId}`);
                    if (discardArea) {
                        this.renderPlayerDiscards(backendIndex, discardArea);
                    }
                }
            } else {
                // Play mode: render human player (0) and 3 AI players (1, 2, 3)
                this.renderPlayerDiscards(0, humanDiscardArea);
                
                // AI player discards (mapping display positions to backend players)
                const discardMapping = {1: 1, 2: 2, 3: 3}; // display ID → backend player ID
                for (let displayId = 1; displayId <= 3; displayId++) {
                    const backendPlayerId = discardMapping[displayId];
                    const aiDiscardArea = document.getElementById(`ai-discards-${displayId}`);
                    if (aiDiscardArea) {
                        this.renderPlayerDiscards(backendPlayerId, aiDiscardArea);
                    }
                }
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
                img.src = this.getTileImage(tileStr);
                img.alt = tileStr;
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
                
                // If AI turns are needed, start the delayed AI sequence
                if (data.ai_turn_needed) {
                    await this.playAISequence();
                }
            } else {
                this.updateStatus(`Error: ${data.message}`, 'error');
            }
        } catch (error) {
            this.updateStatus(`Error discarding tile: ${error.message}`, 'error');
        }
    }

    async playAISequence() {
        // Add a small delay before starting AI turns
        await this.delay(300);
        
        while (true) {
            try {
                const response = await fetch('/api/play_ai_turn', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                const data = await response.json();
                
                if (data.game_over) {
                    // Update game state and stop the sequence
                    await this.updateGameState();
                    break;
                }
                
                if (data.human_turn && this.gameState.game_mode === 'play') {
                    // In play mode, stop and wait for human action
                    await this.updateGameState();
                    break;
                }
                
                if (data.continue_ai || this.gameState.game_mode === 'watch') {
                    // Update game state to show the AI move
                    await this.updateGameState();
                    // Add delay before next AI move
                    await this.delay(500);
                } else {
                    // No more AI moves needed
                    await this.updateGameState();
                    break;
                }
            } catch (error) {
                this.updateStatus(`Error during AI turn: ${error.message}`, 'error');
                break;
            }
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
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

    renderCalledSets() {
        if (!this.gameState.called_sets) return;
        
        // Render called sets for each player
        if (this.gameState.game_mode === 'watch') {
            // Watch mode: render all 4 AI players
            // Backend: [AI0, AI1, AI2, AI3] → Display: [0=bottom, 1=left, 2=top, 3=right]
            const calledSetsMapping = {0: 0, 1: 1, 2: 2, 3: 3}; // backend index → display ID
            for (let backendIndex = 0; backendIndex < 4; backendIndex++) {
                const displayId = calledSetsMapping[backendIndex];
                this.renderPlayerCalledSets(backendIndex, `ai-called-sets-${displayId}`);
            }
        } else {
            // Play mode: render human player (player 0) and 3 AI players
            this.renderPlayerCalledSets(0, 'human-called-sets');
            
            // AI players (mapping display positions to backend players)
            const calledSetsMapping = {1: 1, 2: 2, 3: 3}; // display ID → backend player ID
            for (let displayId = 1; displayId <= 3; displayId++) {
                const backendPlayerId = calledSetsMapping[displayId];
                this.renderPlayerCalledSets(backendPlayerId, `ai-called-sets-${displayId}`);
            }
        }
    }
    
    renderPlayerCalledSets(playerId, containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        
        if (this.gameState.called_sets && this.gameState.called_sets[playerId]) {
            this.gameState.called_sets[playerId].forEach(calledSet => {
                const setElement = document.createElement('div');
                setElement.className = 'called-set';
                
                calledSet.tiles.forEach((tileStr, index) => {
                    const tile = document.createElement('div');
                    tile.className = 'tile';
                    
                    // Check if this is the called tile (horizontal)
                    if (tileStr === calledSet.called_tile) {
                        tile.classList.add('called-horizontal');
                        
                        // Determine positioning based on source
                        const relativeSource = this.getRelativePosition(playerId, calledSet.source_position);
                        if (relativeSource === 'opposite') {
                            tile.classList.add('from-opposite');
                        } else if (relativeSource === 'left') {
                            tile.classList.add('from-left');
                        } else if (relativeSource === 'right') {
                            tile.classList.add('from-right');
                        }
                    }
                    
                    const img = document.createElement('img');
                    img.src = this.getTileImage(tileStr);
                    img.alt = tileStr;
                    tile.appendChild(img);
                    
                    setElement.appendChild(tile);
                });
                
                container.appendChild(setElement);
            });
        }
    }
    
    getRelativePosition(callerPos, sourcePos) {
        // Calculate relative position from caller's perspective
        const diff = (sourcePos - callerPos + 4) % 4;
        switch (diff) {
            case 1: return 'right';
            case 2: return 'opposite';
            case 3: return 'left';
            default: return 'self';
        }
    }

    updateActionButtons() {
        const actionButtonsContainer = document.getElementById('action-buttons');
        actionButtonsContainer.innerHTML = ''; // Clear existing buttons

        if (this.gameState.game_over || this.gameState.game_mode === 'watch') {
            return; // No actions if game is over or in watch mode
        }

        const actions = this.gameState.possible_actions;

        // Tsumo button on player's turn
        if (this.gameState.is_human_turn && actions.tsumo) {
            const tsumoBtn = this.createButton('Tsumo', 'btn-success', () => this.declareTsumo());
            actionButtonsContainer.appendChild(tsumoBtn);
        }

        // Actions on other players' discards
        if (!this.gameState.is_human_turn && this.gameState.last_discarded_tile) {
            let hasAction = false;
            
            // Ron takes priority - if Ron is available, only show Ron and Skip
            if (actions.ron && actions.ron.length > 0) {
                const ronBtn = this.createButton('Ron', 'btn-danger', () => this.handleAction('ron', actions.ron));
                actionButtonsContainer.appendChild(ronBtn);
                hasAction = true;
                
                // When Ron is available, skip showing Pon/Chi buttons
                if (hasAction) {
                    const passBtn = this.createButton('Skip', 'btn-secondary', () => this.handleAction('pass'));
                    actionButtonsContainer.appendChild(passBtn);
                }
            } else {
                // Only show Pon/Chi if Ron is not available
                if (actions.pon && actions.pon.length > 0) {
                    const ponBtn = this.createButton('Pon', 'btn-primary', () => this.handleAction('pon', actions.pon));
                    actionButtonsContainer.appendChild(ponBtn);
                    hasAction = true;
                }
                if (actions.chi && actions.chi.length > 0) {
                    const chiBtn = this.createButton('Chi', 'btn-primary', () => this.handleAction('chi', actions.chi));
                    actionButtonsContainer.appendChild(chiBtn);
                    hasAction = true;
                }

                if (hasAction) {
                    const passBtn = this.createButton('Skip', 'btn-secondary', () => this.handleAction('pass'));
                    actionButtonsContainer.appendChild(passBtn);
                }
            }
        }
    }

    createButton(text, className, onClick) {
        const button = document.createElement('button');
        button.textContent = text;
        button.className = `btn ${className}`;
        button.addEventListener('click', onClick);
        return button;
    }
    
    async handleAction(actionType, tiles = []) {
        try {
            const response = await fetch('/api/action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action_type: actionType, tiles: tiles.map(t => t.map(t2 => t2)) })
            });
            const data = await response.json();
            if (data.success) {
                this.updateStatus(data.message, 'success');
                await this.updateGameState();
                if (data.ai_turn_needed) {
                    await this.playAISequence();
                }
            } else {
                this.updateStatus(`Error: ${data.message}`, 'error');
            }
        } catch (error) {
            this.updateStatus(`Error: ${error.message}`, 'error');
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