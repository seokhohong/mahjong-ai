from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Reuse the same keras/tf import strategy as in core.game
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
    TENSORFLOW_AVAILABLE = True
except Exception:  # pragma: no cover - tested elsewhere already
    TENSORFLOW_AVAILABLE = False

# Import types we need from the core engine
from ..game import (
    GamePerspective,
    Tile,
    TileType,
    Suit,
    Player,
)


class PurePolicyNetwork:
    """
    Pure policy network (no value head needed for training target, but we keep value head
    available for optional auxiliary learning). Architecture and feature encoding mirror PQNetwork.
    """

    def __init__(self, hidden_size: int = 128, embedding_dim: int = 4, max_turns: int = 50):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for PurePolicyNetwork. Please install tensorflow.")
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.max_turns = max_turns
        self.model = self._build_model()

    def _build_model(self):
        # Updated inputs: integer indices for tiles with shared embedding
        hand_idx = keras.Input(shape=(12,), dtype='int32', name='hand_idx')
        disc_idx = keras.Input(shape=(4, self.max_turns), dtype='int32', name='disc_idx')
        from ..constants import GAME_STATE_VEC_LEN, MAX_CALLED_SETS_PER_PLAYER
        called_idx = keras.Input(shape=(4, MAX_CALLED_SETS_PER_PLAYER, 3), dtype='int32', name='called_sets_idx')
        game_state_input = keras.Input(shape=(GAME_STATE_VEC_LEN,), name='game_state_input')

        vocab_size = 18 + 1  # 18 tile categories + PAD=0
        tile_emb = layers.Embedding(input_dim=vocab_size, output_dim=self.embedding_dim, mask_zero=True, name='tile_embedding')

        # Hand conv stack on embeddings
        hand_emb = tile_emb(hand_idx)  # (batch, 12, embed)
        hand_conv1 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(hand_emb)
        hand_conv1 = layers.BatchNormalization()(hand_conv1)
        hand_conv2 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(hand_conv1)
        hand_conv2 = layers.BatchNormalization()(hand_conv2)
        hand_pool = layers.GlobalMaxPooling1D()(hand_conv2)

        # Discard stacks per player on embeddings
        discard_features = []
        disc_slices = [layers.Lambda(lambda x, i=i: x[:, i, :], name=f'disc_slice_{i}')(disc_idx) for i in range(4)]
        for sl in disc_slices:
            di_emb = tile_emb(sl)  # (batch, maxT, embed)
            d1 = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(di_emb)
            d1 = layers.BatchNormalization()(d1)
            d2 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(d1)
            d2 = layers.BatchNormalization()(d2)
            dp = layers.GlobalMaxPooling1D()(d2)
            discard_features.append(dp)

        # Called sets stacks per player
        called_features = []
        cs_slices = [layers.Lambda(lambda x, i=i: x[:, i, :, :], name=f'called_slice_{i}')(called_idx) for i in range(4)]
        for cs in cs_slices:
            # Embed (sets,3)
            cs_emb = tile_emb(cs)  # (batch, sets, 3, embed)
            # Merge sets and triplet dims
            cs_shape = (-1,)
            cs_flat = layers.Reshape((-1, self.embedding_dim))(layers.Lambda(lambda t: tf.reshape(t, (tf.shape(t)[0], tf.shape(t)[1]*tf.shape(t)[2], tf.shape(t)[3])))(cs_emb))
            cp = layers.GlobalMaxPooling1D()(cs_flat)
            called_features.append(cp)

        # Per-player combine
        player_features = []
        for i in range(4):
            if i == 0:
                player_hand = hand_pool
            else:
                player_hand = layers.Dense(128, activation='relu')(layers.Dense(128, activation='relu')(game_state_input))
            combined = layers.Concatenate()([player_hand, discard_features[i], called_features[i]])
            pc = layers.Dense(256, activation='relu')(combined)
            pc = layers.Dropout(0.3)(pc)
            player_features.append(pc)

        all_players = layers.Concatenate()(player_features)
        x = layers.Dense(self.hidden_size, activation='relu')(all_players)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(self.hidden_size // 2, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Concatenate()([x, game_state_input])
        x = layers.Dense(self.hidden_size // 2, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # Single flattened policy head over all action classes
        # Import here to avoid circular import
        from .pure_policy_dataset import get_num_actions  # type: ignore
        policy_flat = layers.Dense(get_num_actions(), activation='softmax', name='policy_flat')(x)
        outputs = [policy_flat]
        losses = { 'policy_flat': 'categorical_crossentropy' }
        metrics = { 'policy_flat': 'accuracy' }

        model = keras.Model(
            inputs=[hand_idx, disc_idx, called_idx, game_state_input],
            outputs=outputs,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=losses,
            metrics=metrics,
        )
        return model

    # Feature extraction mirrors PQNetwork
    def _get_tile_index(self, tile: Tile) -> int:
        return (tile.tile_type.value - 1) * 2 + (0 if tile.suit == Suit.PINZU else 1)

    def _get_tile_embedding(self, tile: Tile) -> np.ndarray:
        embedding = np.zeros(self.embedding_dim)
        tile_idx = self._get_tile_index(tile)
        np.random.seed(tile_idx)
        embedding = np.random.randn(self.embedding_dim) * 0.1
        np.random.seed()
        return embedding

    def _encode_hand_convolutional(self, hand: List[Tile], called_sets: List[Any]) -> np.ndarray:
        hand_tensor = np.zeros((12, 5))
        called_tiles = set()
        for called_set in called_sets:
            for tile in called_set.tiles:
                called_tiles.add(tile)
        for i, tile in enumerate(hand[:12]):
            hand_tensor[i, :4] = self._get_tile_embedding(tile)
            hand_tensor[i, 4] = 1.0 if tile in called_tiles else 0.0
        return hand_tensor

    def _encode_discard_pile_convolutional(self, discards: List[str]) -> np.ndarray:
        discard_tensor = np.zeros((self.max_turns, self.embedding_dim))
        for i, tile_str in enumerate(discards[:self.max_turns]):
            try:
                tile_type = int(tile_str[:-1])
                suit = Suit(tile_str[-1])
                tile = Tile(suit, TileType(tile_type))
                discard_tensor[i] = self._get_tile_embedding(tile)
            except Exception:
                continue
        return discard_tensor

    def _get_player_discards(self, game_state: GamePerspective, player_id: int) -> List[str]:
        if hasattr(game_state, 'player_discards') and player_id in game_state.player_discards:  # type: ignore[attr-defined]
            return game_state.player_discards[player_id]  # type: ignore[index]
        return []

    def _extract_additional_features(self, game_state: GamePerspective) -> np.ndarray:
        # Use same ordering and invariances as dataset encoder (_encode_game_state_50)
        from ..constants import (
            TOTAL_TILES,
            MAX_HAND_TILES,
            MAX_CALLED_SETS_PER_PLAYER,
            MAX_CALLED_SETS_ALL_OPPONENTS,
            GAME_STATE_VEC_LEN,
        )

        vec = np.zeros((GAME_STATE_VEC_LEN,), dtype=np.float32)

        # Scalars
        vec[0] = float(game_state.remaining_tiles) / float(TOTAL_TILES)
        vec[1] = 1.0 if game_state.can_call else 0.0
        vec[2] = float(len(game_state.player_hand)) / float(MAX_HAND_TILES)
        your_called_sets = game_state.called_sets.get(game_state.player_id, [])
        vec[3] = float(len(your_called_sets)) / float(MAX_CALLED_SETS_PER_PLAYER)
        total_opponent_sets = float(sum(len(sets) for pid, sets in game_state.called_sets.items() if pid != game_state.player_id))
        vec[4] = total_opponent_sets / float(MAX_CALLED_SETS_ALL_OPPONENTS)
        vec[5] = float(len(getattr(game_state, 'visible_tiles', []))) / float(TOTAL_TILES)

        # Last discard embedding (fixed 4 dims) at positions 6..9
        if game_state.last_discarded_tile is not None:
            emb = self._get_tile_embedding(game_state.last_discarded_tile).astype(np.float32)
        else:
            emb = np.zeros((self.embedding_dim,), dtype=np.float32)
        vec[6:6 + self.embedding_dim] = emb[: self.embedding_dim]

        # Last discard player one-hot relative to viewer at last 4 positions
        if game_state.last_discard_player is not None:
            rel = (int(game_state.last_discard_player) - int(game_state.player_id)) % 4
            vec[-4 + rel] = 1.0

        return vec

    def _extract_features(self, game_state: GamePerspective) -> List[np.ndarray]:
        hand_features = self._encode_hand_convolutional(
            game_state.player_hand,
            game_state.called_sets.get(game_state.player_id, []),
        )
        discard_features = [
            self._encode_discard_pile_convolutional(self._get_player_discards(game_state, i))
            for i in range(4)
        ]
        game_state_features = self._extract_additional_features(game_state)
        return [hand_features] + discard_features + [game_state_features]

    def evaluate(self, game_state: GamePerspective) -> Dict[str, np.ndarray]:
        features = self._extract_features(game_state)
        batched = [np.expand_dims(f, axis=0) for f in features]
        pred_flat = self.model.predict(batched, verbose=0)[0]
        return { 'policy': pred_flat }

    def save_model(self, filepath: str) -> None:
        if not filepath.endswith('.keras'):
            filepath += '.keras'
        self.model.save(filepath)

    def load_model(self, filepath: str) -> None:
        if not filepath.endswith('.keras'):
            filepath += '.keras'
        # Allow Lambda layer deserialization used for slicing discards
        try:
            self.model = keras.models.load_model(filepath, safe_mode=False)
        except TypeError:
            # Older TF may not support safe_mode param
            self.model = keras.models.load_model(filepath)


class PurePolicyRecorder(Player):
    """
    Wraps a `Player` and records (state, action, value) tuples for each decision.
    For now, value is a placeholder float provided optionally by caller (e.g., outcome or heuristic).
    """

    def __init__(self, wrapped: Player):
        super().__init__(wrapped.player_id)
        # Use __dict__ directly to avoid recursion in __setattr__
        self.__dict__['wrapped'] = wrapped
        self.__dict__['records'] = []  # type: ignore[assignment]

    def record(self, game_state: GamePerspective, player: Player, kind: str = 'turn', value: float = 0.0) -> None:
        # This hook is triggered by player's notify_* methods. We don't know action yet.
        # We'll store the state now and pair with the action once chosen via interceptors below.
        pass

    def clear(self) -> None:
        self.records.clear()

    # Intercept player's decisions by delegating with logging
    def play(self, game_state: GamePerspective) -> Any:
        action = self.wrapped.play(game_state)
        # Default value 0.0; downstream can relabel with returns if desired
        self.records.append((game_state, action, 0.0))
        return action

    def choose_reaction(self, game_state: GamePerspective, options: Dict[str, List[List[Tile]]]) -> Any:
        action = self.wrapped.choose_reaction(game_state, options)
        self.records.append((game_state, action, 0.0))
        return action

    # Delegate unknown attributes/methods to wrapped player (e.g., heuristics helpers)
    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial forwarding
        return getattr(self.__dict__['wrapped'], name)

    # Ensure game back-reference is kept in sync
    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        if name == '_game':
            # Set on both recorder and wrapped player
            self.__dict__['_game'] = value
            try:
                setattr(self.__dict__['wrapped'], '_game', value)
            except Exception:
                pass
            return
        self.__dict__[name] = value


