from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from src.core.constants import MAX_TURNS
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm  # type: ignore

import numpy as np

# Use PyTorch for the learning stack; keep legacy flag name for compatibility with tests
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False

# Backward-compat alias
TENSORFLOW_AVAILABLE = TORCH_AVAILABLE

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

    def __init__(self, hidden_size: int = 128, embedding_dim: int = 4, max_turns: int = MAX_TURNS):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PurePolicyNetwork. Please install torch.")
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.max_turns = max_turns
        # Determine action space size once
        from .pure_policy_dataset import get_num_actions  # type: ignore
        self._num_actions = int(get_num_actions())
        # Convolutional feature towers over structured inputs
        from ..constants import TOTAL_TILES, NUM_PLAYERS, GAME_STATE_VEC_LEN as GSV
        dealt = 11 * int(NUM_PLAYERS)
        self._max_discards_per_player = max(1, (int(TOTAL_TILES) - dealt) // int(NUM_PLAYERS))
        self._max_called_tiles_per_player = 9  # up to 3 melds fully; clamp to 9

        conv_ch1, conv_ch2 = 32, 64

        class _PolicyNet(nn.Module):
            def __init__(self, outer: 'PurePolicyNetwork') -> None:
                super().__init__()
                self.outer = outer
                # Convs and MLP belong to this module so that .to(device) migrates all weights
                self.hand_conv = nn.Sequential(
                    nn.Conv1d(outer.embedding_dim, conv_ch1, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(conv_ch1, conv_ch2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                )
                self.calls_conv = nn.Sequential(
                    nn.Conv1d(outer.embedding_dim, conv_ch1, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(conv_ch1, conv_ch2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                )
                self.disc_conv = nn.Sequential(
                    nn.Conv1d(outer.embedding_dim, conv_ch1, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(conv_ch1, conv_ch2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                )
                self.mlp = nn.Sequential(
                    nn.Linear((conv_ch2 * 3) + GSV, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size // 2, self.outer._num_actions),
                )

            def forward(self, hand_seq: torch.Tensor, calls_seq: torch.Tensor, disc_seq: torch.Tensor, gsv: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
                h = self.hand_conv(hand_seq).squeeze(-1)
                c = self.calls_conv(calls_seq).squeeze(-1)
                d = self.disc_conv(disc_seq).squeeze(-1)
                x = torch.cat([h, c, d, gsv], dim=1)
                return self.mlp(x)

        self._net = _PolicyNet(self)
        # Device and eval mode by default
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._net.to(self._device)
        self._net.eval()
        # Precompute deterministic tile embeddings table for indices [0..18] (0=PAD)
        self._embedding_table = np.zeros((19, self.embedding_dim), dtype=np.float32)
        for idx in range(1, 19):
            rng = np.random.RandomState(seed=idx)
            self._embedding_table[idx] = (rng.randn(self.embedding_dim) * 0.1).astype(np.float32)
        # Provide a Keras-like wrapper exposing fit/predict/save/load used in tests
        self.model = _KerasLikeWrapper(self)

    # Keras-like fit/predict wrapper implemented below; no separate builder required

    # Feature extraction mirrors PQNetwork
    def _get_tile_index(self, tile: Tile) -> int:
        return (tile.tile_type.value - 1) * 2 + (0 if tile.suit == Suit.PINZU else 1)

    def _get_tile_embedding(self, tile: Tile) -> np.ndarray:
        # Kept for compatibility; now uses precomputed table
        tile_idx = self._get_tile_index(tile)
        if 0 <= tile_idx + 1 < self._embedding_table.shape[0]:
            return self._embedding_table[tile_idx + 1].copy()
        return np.zeros((self.embedding_dim,), dtype=np.float32)

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
        # Legacy path not performance critical; batch path uses vectorized table lookup
        discard_tensor = np.zeros((self.max_turns, self.embedding_dim), dtype=np.float32)
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
        # Use same ordering and invariances as dataset encoder (50-length game-state vector)
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

        # Explicit flags before last-discard embedding (to mirror dataset order)
        vec[6] = 1.0 if game_state.can_tsumo() else 0.0
        vec[7] = 1.0 if game_state.can_ron() else 0.0

        # Last discard embedding (fixed 4 dims) follows at positions 8..11
        if game_state.last_discarded_tile is not None:
            emb = self._get_tile_embedding(game_state.last_discarded_tile).astype(np.float32)
        else:
            emb = np.zeros((self.embedding_dim,), dtype=np.float32)
        vec[8:8 + self.embedding_dim] = emb[: self.embedding_dim]


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
        # Reuse predict path to ensure identical preprocessing
        from .pure_policy_dataset import serialize_state, extract_indexed_state  # type: ignore
        sd = serialize_state(game_state)
        idx = extract_indexed_state(sd)
        probs = self.model.predict([
            idx['hand_idx'][None, :],
            idx['disc_idx'][None, :, :],
            idx.get('called_sets_idx', np.zeros((4, 4, 3), dtype=np.int32))[None, :, :, :],
            idx['game_state'][None, :],
        ], verbose=0)[0]
        return {'policy': probs}

    def _extract_features_from_indexed(self, hand_idx: np.ndarray, disc_idx: np.ndarray, called_idx: np.ndarray, game_state_vec: np.ndarray) -> List[np.ndarray]:
        """Vectorized single-sample path using precomputed embedding table."""
        # Hand embeddings and called flag
        hand_idx_safe = np.asarray(hand_idx, dtype=np.int32)
        hand_emb = self._embedding_table[np.clip(hand_idx_safe, 0, 18)]  # (12, embed)
        # Called sets presence for current player = row 0 of called_idx
        if called_idx.ndim == 3 and called_idx.shape[0] >= 1:
            cs0 = np.asarray(called_idx[0], dtype=np.int32)  # (sets,3)
            # Presence of each code in [0..18]
            codes = np.arange(19, dtype=np.int32)
            presence = (cs0[..., None] == codes[None, None, :]).any(axis=(0, 1))  # (19,)
        else:
            presence = np.zeros((19,), dtype=bool)
        called_flag = presence[np.clip(hand_idx_safe, 0, 18)] & (hand_idx_safe > 0)
        hand_tensor = np.concatenate([hand_emb, called_flag.astype(np.float32)[..., None]], axis=-1)

        # Discard embeddings per player
        disc_idx_safe = np.asarray(disc_idx, dtype=np.int32)
        disc_emb = self._embedding_table[np.clip(disc_idx_safe, 0, 18)]  # (4, maxT, embed)

        # Game state vec
        gs = np.asarray(game_state_vec, dtype=np.float32)
        from ..constants import GAME_STATE_VEC_LEN as GSV
        if gs.shape[0] < GSV:
            gs = np.pad(gs, (0, GSV - gs.shape[0]))
        elif gs.shape[0] > GSV:
            gs = gs[:GSV]

        return [hand_tensor] + [disc_emb[i] for i in range(min(4, disc_emb.shape[0]))] + [gs]

    def save_model(self, filepath: str) -> None:
        """Persist a self-describing checkpoint containing hyperparams and weights.

        Stores:
          - init kwargs (hidden_size, embedding_dim, max_turns)
          - network state_dict
          - embedding table
        """
        if not filepath.endswith('.pt'):
            filepath += '.pt'
        payload = {
            'format': 'pure_policy_full_v1',
            'init': {
                'hidden_size': int(self.hidden_size),
                'embedding_dim': int(self.embedding_dim),
                'max_turns': int(self.max_turns),
            },
            'state_dict': self._net.state_dict(),
            'embedding_table': self._embedding_table,
        }
        torch.save(payload, filepath)

    def load_model(self, filepath: str) -> None:
        """Load model from file, supporting both full-object and state_dict checkpoints.

        - If the checkpoint is a PurePolicyNetwork instance, copy its fields into self
          so existing references (e.g., in players) remain valid.
        - If it's a state_dict, attempt to load into the current network; fall back to
          strict=False for partial compatibility.
        """
        if not filepath.endswith('.pt'):
            filepath += '.pt'
        ckpt = torch.load(filepath, map_location='cpu')
        # Self-describing payload
        if isinstance(ckpt, dict) and ckpt.get('format') == 'pure_policy_full_v1':
            init = ckpt.get('init', {})
            state_dict = ckpt.get('state_dict', {})
            emb_table = ckpt.get('embedding_table')
            # Rebuild a fresh network with saved hyperparams
            rebuilt = PurePolicyNetwork(
                hidden_size=int(init.get('hidden_size', self.hidden_size)),
                embedding_dim=int(init.get('embedding_dim', self.embedding_dim)),
                max_turns=int(init.get('max_turns', self.max_turns)),
            )
            try:
                rebuilt._net.load_state_dict(state_dict)
            except Exception:
                rebuilt._net.load_state_dict(state_dict, strict=False)
            if emb_table is not None:
                rebuilt._embedding_table = np.asarray(emb_table, dtype=np.float32)
            # Copy rebuilt internals into self
            self.hidden_size = rebuilt.hidden_size
            self.embedding_dim = rebuilt.embedding_dim
            self.max_turns = rebuilt.max_turns
            self._num_actions = rebuilt._num_actions
            self._net = rebuilt._net
            self._embedding_table = rebuilt._embedding_table
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._net.to(self._device)
            self._net.eval()
            return
        # Legacy state_dict checkpoint (no hyperparams)
        if isinstance(ckpt, dict):
            try:
                self._net.load_state_dict(ckpt)
            except Exception:
                self._net.load_state_dict(ckpt, strict=False)
            return
        raise ValueError(f"Unsupported checkpoint format in {filepath}: {type(ckpt)}")

    @staticmethod
    def load_from_file(filepath: str) -> 'PurePolicyNetwork':
        """Load and return a PurePolicyNetwork instance from file."""
        if not filepath.endswith('.pt'):
            filepath += '.pt'
        obj = torch.load(filepath, map_location='cpu')
        if isinstance(obj, dict) and obj.get('format') == 'pure_policy_full_v1':
            init = obj.get('init', {})
            inst = PurePolicyNetwork(
                hidden_size=int(init.get('hidden_size', 128)),
                embedding_dim=int(init.get('embedding_dim', 4)),
                max_turns=int(init.get('max_turns', MAX_TURNS)),
            )
            try:
                inst._net.load_state_dict(obj.get('state_dict', {}))
            except Exception:
                inst._net.load_state_dict(obj.get('state_dict', {}), strict=False)
            if obj.get('embedding_table') is not None:
                inst._embedding_table = np.asarray(obj.get('embedding_table'), dtype=np.float32)
            inst._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            inst._net.to(inst._device)
            inst._net.eval()
            return inst
        # Legacy: state_dict only
        inst = PurePolicyNetwork()
        try:
            inst._net.load_state_dict(obj)  # type: ignore[arg-type]
        except Exception:
            inst._net.load_state_dict(obj, strict=False)  # type: ignore[arg-type]
        inst._net.eval()
        return inst

    # --- PyTorch interop helpers ---
    def to(self, device: torch.device) -> 'PurePolicyNetwork':
        """Move underlying torch module to the specified device and update internal device."""
        self._device = device
        self._net.to(device)
        return self

    def parameters(self):
        """Expose torch parameters for external inspection or optimization."""
        return self._net.parameters()

    @property
    def torch_module(self) -> nn.Module:  # pragma: no cover - thin accessor
        return self._net

class _KerasLikeWrapper:
    """Minimal Keras-like facade with fit/predict using PyTorch, with GPU support.

    This wrapper precomputes flattened features once per fit call for speed.
    """

    def __init__(self, owner: PurePolicyNetwork):
        self._owner = owner

    def _precompute_cnn(self, hands: np.ndarray, discs: np.ndarray, called: np.ndarray, gss: np.ndarray):
        """Vectorized precompute of CNN inputs from indexed features.

        Returns:
          - hand_seq: (N, embed, 12)
          - calls_seq: (N, embed, 36)  # 4 players * 9 called tiles
          - disc_seq: (N, embed, 4*K)  # concatenated discards per player
          - gsv: (N, GSV)
        """
        hands = np.asarray(hands, dtype=np.int32)
        discs = np.asarray(discs, dtype=np.int32)
        from ..constants import NUM_PLAYERS, MAX_CALLED_SETS_PER_PLAYER, MAX_TILES_PER_CALLED_SET
        if called is None:
            called = np.zeros((hands.shape[0], int(NUM_PLAYERS), int(MAX_CALLED_SETS_PER_PLAYER), int(MAX_TILES_PER_CALLED_SET)), dtype=np.int32)
        else:
            called = np.asarray(called, dtype=np.int32)

        from ..constants import GAME_STATE_VEC_LEN as GSV
        gss = np.asarray(gss, dtype=np.float32)
        if gss.ndim == 1:
            gss = gss[None, :]
        if gss.shape[1] < GSV:
            pad = np.zeros((gss.shape[0], GSV - gss.shape[1]), dtype=np.float32)
            gss = np.concatenate([gss, pad], axis=1)
        elif gss.shape[1] > GSV:
            gss = gss[:, :GSV]

        # --- Hand sequence: sort non-zero ascending, zeros to end ---
        hand_vals = np.where(hands > 0, hands, 999)
        hand_sorted = np.sort(hand_vals, axis=1)
        hand_sorted = np.where(hand_sorted == 999, 0, hand_sorted)
        hand_seq = self._owner._embedding_table[np.clip(hand_sorted, 0, 18)]  # (N,12,embed)
        hand_seq = np.transpose(hand_seq, (0, 2, 1))  # (N,embed,12)

        # --- Calls per player: flatten (4,3) per set -> 12, push zeros to end preserving order, take first 9 ---
        called_flat = called.reshape(called.shape[0], int(NUM_PLAYERS), -1)  # (N,4, 4*3)
        zero_first = (called_flat == 0).astype(np.int32)
        order = np.argsort(zero_first, axis=2, kind='stable')  # non-zero (0) before zero (1)
        called_reordered = np.take_along_axis(called_flat, order, axis=2)
        called_top9 = called_reordered[:, :, :self._owner._max_called_tiles_per_player]  # (N,players,9)
        calls_emb = self._owner._embedding_table[np.clip(called_top9, 0, 18)]  # (N,4,9,embed)
        calls_seq = calls_emb.reshape(calls_emb.shape[0], -1, calls_emb.shape[-1])  # (N,36,embed)
        calls_seq = np.transpose(calls_seq, (0, 2, 1))  # (N,embed,36)

        # --- Discards per player: truncate/pad to K then concat players ---
        K = self._owner._max_discards_per_player
        maxT = discs.shape[2] if discs.ndim >= 3 else 0
        if maxT >= K:
            disc_slice = discs[:, :, :K]
        else:
            pad = np.zeros((discs.shape[0], discs.shape[1], K - maxT), dtype=np.int32)
            disc_slice = np.concatenate([discs, pad], axis=2)
        disc_emb = self._owner._embedding_table[np.clip(disc_slice, 0, 18)]  # (N,4,K,embed)
        disc_seq = disc_emb.reshape(disc_emb.shape[0], -1, disc_emb.shape[-1])  # (N,4*K,embed)
        disc_seq = np.transpose(disc_seq, (0, 2, 1))  # (N,embed,4*K)

        return hand_seq.astype(np.float32), calls_seq.astype(np.float32), disc_seq.astype(np.float32), gss.astype(np.float32)

    def fit(
        self,
        x_list: List[np.ndarray],
        y: Dict[str, np.ndarray],
        epochs: int = 1,
        batch_size: int = 8,
        verbose: int = 0,
        shuffle: bool = True,
        sample_weight: np.ndarray = None,
        early_stopping_patience: int = 5,
        val_x_list: Optional[List[np.ndarray]] = None,
        val_targets: Optional[Dict[str, np.ndarray]] = None,
        val_sample_weight: Optional[np.ndarray] = None,
        legality_masks: np.ndarray = None,
        val_legality_masks: Optional[np.ndarray] = None,
        ):
        """
        Enhanced policy gradient training with better metrics and stability.
        """
        try:
            hands, discs, called, gss = x_list
        except Exception:
            return self
        
        y_flat = y.get('policy_flat') if isinstance(y, dict) else None
        if y_flat is None:
            return self

        # Precompute CNN tensors and labels
        hand_seq_np, calls_seq_np, disc_seq_np, gsv_np = self._precompute_cnn(hands, discs, called, gss)
        labels_np = np.argmax(y_flat, axis=1).astype(np.int64)
        
        # Require rewards for policy-gradient training
        if sample_weight is None:
            raise ValueError("sample_weight (rewards) is required for policy-gradient training")
        # Require legality masks for training to enforce rules during learning
        if legality_masks is None:
            raise ValueError("legality_masks is required for training")
        # Use RAW rewards for both training and metrics (no normalization)
        original_rewards = sample_weight.astype(np.float32)
        
        device = self._owner._device
        hand_seq = torch.from_numpy(hand_seq_np).to(device)
        calls_seq = torch.from_numpy(calls_seq_np).to(device)
        disc_seq = torch.from_numpy(disc_seq_np).to(device)
        gsv_t = torch.from_numpy(gsv_np).to(device)
        yb_all = torch.from_numpy(labels_np).to(device)
        # Legality masks (required)
        try:
            lm_all = torch.from_numpy(legality_masks.astype(bool)).to(device)
            if lm_all.ndim != 2 or lm_all.shape[0] != hand_seq.shape[0] or lm_all.shape[1] != self._owner._num_actions:
                raise ValueError("legality_masks must have shape (N, num_actions)")
            # Assert every sample has at least one legal action
            per_row_any = lm_all.any(dim=1)
            if not bool(per_row_any.all().item()):
                # Collect a few offending indices for debugging
                bad_idx = torch.nonzero(~per_row_any, as_tuple=False).view(-1).tolist()
                preview = bad_idx[:10]
                raise AssertionError(
                    f"Found {len(bad_idx)} samples with no legal actions in legality_masks. "
                    f"Example indices: {preview}"
                )
        except Exception as e:
            raise ValueError(f"Invalid legality_masks: {e}")
        
        # Use RAW rewards for training (policy gradient) and for categorization/metrics
        orig_rewards_all = torch.from_numpy(original_rewards).to(device)
        wb_all = orig_rewards_all

        # Setup optimizer with optional learning rate scheduling
        optimizer = torch.optim.Adam(self._owner._net.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=False
        )
        
        def policy_gradient_loss(logits: torch.Tensor, actions: torch.Tensor, 
                                advantages: torch.Tensor, entropy_weight: float = 0.01):
            """
            Policy gradient loss with entropy regularization.
            Returns loss, entropy, and action probabilities for metrics.
            """
            # Clamp logits for stability
            logits = torch.clamp(logits, min=-10.0, max=10.0)
            
            # Calculate log probabilities and probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            
            # Get action log probabilities
            action_log_probs = log_probs.gather(1, actions.view(-1, 1)).squeeze(1)
            action_probs = probs.gather(1, actions.view(-1, 1)).squeeze(1)
            
            # Policy gradient loss
            pg_loss = -(action_log_probs * advantages).mean()
            
            # Entropy for exploration
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            
            # Combined loss
            total_loss = pg_loss - entropy_weight * entropy
            
            return total_loss, entropy, action_probs
        
        self._owner._net.train()

        from tqdm import tqdm  # type: ignore

        num_samples = hand_seq.shape[0]
        bs = max(1, min(batch_size, num_samples))

        # Tracking metrics
        best_metric = -float('inf')
        epochs_no_improve = 0
        history = {
            'loss': [], 'entropy': [], 'win_prob': [], 'lose_prob': [], 
            'neutral_prob': [], 'performance': [], 'win_count': [], 'lose_count': [], 'neutral_count': []
        }

        # Precompute validation features/labels/weights if provided
        val_labels: Optional[np.ndarray] = None
        val_weights: Optional[np.ndarray] = None
        if val_x_list is not None and val_targets is not None:
            vh, vd, vc, vg = val_x_list
            vy_flat = val_targets.get('policy_flat') if isinstance(val_targets, dict) else None
            if vy_flat is not None:
                # we re-precompute CNN tensors later during validation step
                val_labels = np.argmax(vy_flat, axis=1).astype(np.int64)
                if val_sample_weight is not None:
                    val_weights = val_sample_weight.astype(np.float32)
        
        for ep in range(max(1, epochs)):
            indices = torch.arange(num_samples, device=device)
            if shuffle:
                indices = indices[torch.randperm(num_samples, device=device)]
            
            total_batches = (num_samples + bs - 1) // bs
            
            # Epoch metrics
            epoch_loss = 0.0
            epoch_entropy = 0.0
            win_probs, lose_probs, neutral_probs = [], [], []
            # For top-K actions over win samples
            win_action_prob_sum = np.zeros((self._owner._num_actions,), dtype=np.float64)
            win_correct = 0
            win_samples = 0
            # Track CE/PG on train
            ce_sum = 0.0
            pg_sum = 0.0
            
            # Progress bar
            bar = None
            if verbose:
                bar = tqdm(range(total_batches), desc=f"Epoch {ep+1}/{epochs}", leave=False) if tqdm else None
                if bar is None:
                    print(f"Epoch {ep+1}/{epochs}")
            
            batch_idx = 0
            for start in range(0, num_samples, bs):
                end = min(num_samples, start + bs)
                idx = indices[start:end]
                # slices prepared below for each input tensor
                yb = yb_all.index_select(0, idx)
                
                # Use RAW rewards for training (policy gradient)
                wb = wb_all.index_select(0, idx)
                orig_rewards = orig_rewards_all.index_select(0, idx)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass
                logits = self._owner._net(
                    hand_seq.index_select(0, idx),
                    calls_seq.index_select(0, idx),
                    disc_seq.index_select(0, idx),
                    gsv_t.index_select(0, idx),
                )
                # Apply legality mask to logits
                lm = lm_all.index_select(0, idx)
                logits = logits.masked_fill(~lm, -1e9)
                loss, entropy, action_probs = policy_gradient_loss(logits, yb, wb)
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._owner._net.parameters(), max_norm=1.0)
                optimizer.step()

                # Collect metrics
                with torch.no_grad():
                    epoch_loss += loss.item()
                    epoch_entropy += entropy.item()
                    
                    # Predictions for metrics
                    preds = torch.argmax(logits, dim=-1)
                    eq = (preds == yb).float()

                    # CE/PG loss components on train
                    neg_logp = -torch.log(action_probs + 1e-8)
                    ce_sum += neg_logp.sum().item()
                    
                    # CRITICAL FIX: Use ORIGINAL rewards for categorization, not normalized advantages

                    # Win actions (reward = 1)
                    win_mask = (orig_rewards == 1.0)
                    if win_mask.any():
                        win_probs.extend(action_probs[win_mask].cpu().numpy())
                        win_correct += eq[win_mask].sum().item()
                        win_samples += win_mask.sum().item()
                        # Accumulate full-policy probabilities for top-K analysis
                        probs_full = F.softmax(logits, dim=-1)
                        win_action_prob_sum += probs_full[win_mask].sum(dim=0).cpu().numpy()
                    # PG loss term: -(reward * log pi(a|s))
                    pg_sum += (-(orig_rewards * torch.log(action_probs + 1e-8))).sum().item()

                    # Lose actions (reward = -1)
                    lose_mask = (orig_rewards == -1.0)
                    if lose_mask.any():
                        lose_probs.extend(action_probs[lose_mask].cpu().numpy())

                    # Neutral actions (reward = 0)
                    neutral_mask = (orig_rewards == 0.0)
                    if neutral_mask.any():
                        neutral_probs.extend(action_probs[neutral_mask].cpu().numpy())
                
                batch_idx += 1
                if verbose and bar is not None:
                    avg_win_prob = np.mean(win_probs) if win_probs else 0.0
                    avg_lose_prob = np.mean(lose_probs) if lose_probs else 0.0
                    
                    bar.set_postfix({
                        "loss": f"{epoch_loss/batch_idx:.4f}",
                        "win_π": f"{avg_win_prob:.3f}",
                        "lose_π": f"{avg_lose_prob:.3f}",
                    })
                    bar.update(1)
            
            # Calculate epoch statistics
            epoch_loss /= max(1, batch_idx)
            epoch_entropy /= max(1, batch_idx)
            
            # Calculate average probabilities
            avg_win_prob = np.mean(win_probs) if win_probs else 0.0
            avg_lose_prob = np.mean(lose_probs) if lose_probs else 0.0
            avg_neutral_prob = np.mean(neutral_probs) if neutral_probs else 0.0
            
            # Count samples in each category
            win_count = len(win_probs)
            lose_count = len(lose_probs)
            neutral_count = len(neutral_probs)
            
            # Key performance metric: difference between win and lose probabilities
            performance = avg_win_prob - avg_lose_prob
            
            # Store history
            history['loss'].append(epoch_loss)
            history['entropy'].append(epoch_entropy)
            history['win_prob'].append(avg_win_prob)
            history['lose_prob'].append(avg_lose_prob)
            history['neutral_prob'].append(avg_neutral_prob)
            history['performance'].append(performance)
            history['win_count'].append(win_count)
            history['lose_count'].append(lose_count)
            history['neutral_count'].append(neutral_count)
            
            # Update learning rate based on performance
            scheduler.step(performance)
            
            if verbose:
                if bar is not None:
                    bar.close()
                
                # Normalize by total samples seen in epoch
                total_samples = max(1, num_samples)
                train_ce = ce_sum / total_samples
                train_pg = pg_sum / total_samples
                print(f"\nEpoch {ep+1}/{epochs} Summary:")
                print(f"  Loss: {epoch_loss:.4f} | Entropy: {epoch_entropy:.4f}")
                print(f"  [TRAIN] CE: {train_ce:.4f} | PG: {train_pg:.4f}")
                print(f"  [TRAIN] Sample Counts -> Win: {win_count} | Lose: {lose_count} | Neutral: {neutral_count}")
                print(f"  [TRAIN] Policy Probs -> Win: {avg_win_prob:.4f} | Lose: {avg_lose_prob:.4f} | Neutral: {avg_neutral_prob:.4f}")
                # Top-10 actions by average probability over winning samples
                if win_samples > 0:
                    avg_action_probs_win = (win_action_prob_sum / max(1, int(win_samples))).astype(np.float64)
                    top_idx = np.argsort(avg_action_probs_win)[-10:][::-1]
                    try:
                        from .pure_policy_dataset import get_action_labels  # type: ignore
                        labels = get_action_labels()
                    except Exception:
                        labels = [str(i) for i in range(len(avg_action_probs_win))]

                    # Make labels human readable: translate discard_i and pon_i to tile strings
                    def _pretty(label: str) -> str:
                        if label.startswith('discard_'):
                            try:
                                ti = int(label.split('_', 1)[1])
                                rank = (ti // 2) + 1
                                suit = 'p' if (ti % 2) == 0 else 's'
                                return f'discard_{rank}{suit}'
                            except Exception:
                                return label
                        if label.startswith('pon_'):
                            try:
                                ti = int(label.split('_', 1)[1])
                                rank = (ti // 2) + 1
                                suit = 'p' if (ti % 2) == 0 else 's'
                                return f'pon_{rank}{suit}'
                            except Exception:
                                return label
                        if label.startswith('chi_'):
                            # chi_{tileIdx}_{variant}
                            try:
                                _, ti, var = label.split('_', 2)
                                ti = int(ti)
                                rank = (ti // 2) + 1
                                suit = 'p' if (ti % 2) == 0 else 's'
                                return f'chi_{rank}{suit}_{var}'
                            except Exception:
                                return label
                        return label

                    print("  [TRAIN] (wins) top actions:")
                    for rank_i, ai in enumerate(top_idx, 1):
                        print(f"    {rank_i}. {_pretty(labels[ai])}: {avg_action_probs_win[ai]:.4f}")
                    # Also report ron and tsumo average probabilities over wins
                    try:
                        ron_idx = labels.index('ron')
                        tsumo_idx = labels.index('tsumo')
                    except ValueError:
                        ron_idx, tsumo_idx = 1, 2  # Fallback to standard indices
                    print(f"  [TRAIN] (wins) ron: {avg_action_probs_win[ron_idx]:.4f} | tsumo: {avg_action_probs_win[tsumo_idx]:.4f}")
                
                if avg_win_prob > 0 and avg_lose_prob > 0:
                    ratio = avg_win_prob / avg_lose_prob
                    print(f"  [TRAIN] Win/Lose Ratio: {ratio:.2f}x | Performance: {performance:.4f}")
                elif avg_lose_prob == 0:
                    print(f"  [TRAIN] WARNING: No lose samples found in this epoch! Check data distribution.")
                
                print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

                # Validation metrics per epoch if provided
                if val_labels is not None:
                    with torch.no_grad():
                        vh, vd, vc, vg = val_x_list
                        h_np, c_np, d_np, g_np = self._precompute_cnn(vh, vd, vc, vg)
                        logits_val = self._owner._net(
                            torch.from_numpy(h_np).to(device),
                            torch.from_numpy(c_np).to(device),
                            torch.from_numpy(d_np).to(device),
                            torch.from_numpy(g_np).to(device),
                        )
                        # Apply validation legality mask if provided
                        if val_legality_masks is not None:
                            try:
                                lm_val = torch.from_numpy(val_legality_masks.astype(bool)).to(logits_val.device)
                                if lm_val.shape == logits_val.shape:
                                    logits_val = logits_val.masked_fill(~lm_val, -1e9)
                                elif lm_val.ndim == 2 and lm_val.shape[1] == logits_val.shape[1]:
                                    logits_val = logits_val.masked_fill(~lm_val, -1e9)
                            except Exception:
                                pass
                        log_probs_val = F.log_softmax(logits_val, dim=-1)
                        # CE and chosen action probabilities
                        idx = torch.arange(len(val_labels), device=logits_val.device)
                        labels_t = torch.from_numpy(val_labels).to(logits_val.device)
                        true_logp_val = log_probs_val[idx, labels_t]
                        ce_val = float((-true_logp_val).mean().cpu().item()) if val_labels.size > 0 else 0.0
                        action_prob_val = torch.exp(true_logp_val).cpu().numpy()
                        # PG
                        if val_weights is not None and val_weights.size == val_labels.size:
                            vw = torch.from_numpy(val_weights).to(logits_val.device)
                            pg_val = float((-(vw * true_logp_val)).mean().cpu().item())
                            win_mask_val = (vw == 1.0).cpu().numpy()
                            lose_mask_val = (vw == -1.0).cpu().numpy()
                            neutral_mask_val = (vw == 0.0).cpu().numpy()
                        else:
                            pg_val = ce_val
                            win_mask_val = np.zeros_like(val_labels, dtype=bool)
                            lose_mask_val = np.zeros_like(val_labels, dtype=bool)
                            neutral_mask_val = np.zeros_like(val_labels, dtype=bool)
                    # Aggregate holdout policy probs and counts
                    ho_win_prob = float(np.mean(action_prob_val[win_mask_val])) if np.any(win_mask_val) else 0.0
                    ho_lose_prob = float(np.mean(action_prob_val[lose_mask_val])) if np.any(lose_mask_val) else 0.0
                    ho_neutral_prob = float(np.mean(action_prob_val[neutral_mask_val])) if np.any(neutral_mask_val) else 0.0
                    ho_win_count = int(np.sum(win_mask_val))
                    ho_lose_count = int(np.sum(lose_mask_val))
                    ho_neutral_count = int(np.sum(neutral_mask_val))
                    ho_performance = ho_win_prob - ho_lose_prob
                    # Print holdout metrics aligned with train
                    print(f"  [HOLDOUT] CE: {ce_val:.4f} | PG: {pg_val:.4f}")
                    print(f"  [HOLDOUT] Sample Counts -> Win: {ho_win_count} | Lose: {ho_lose_count} | Neutral: {ho_neutral_count}")
                    print(f"  [HOLDOUT] Policy Probs -> Win: {ho_win_prob:.4f} | Lose: {ho_lose_prob:.4f} | Neutral: {ho_neutral_prob:.4f}")
                    if ho_win_prob > 0 and ho_lose_prob > 0:
                        print(f"  [HOLDOUT] Win/Lose Ratio: {ho_win_prob/ho_lose_prob:.2f}x | Performance: {ho_performance:.4f}")
                    elif ho_lose_prob == 0:
                        print(f"  [HOLDOUT] WARNING: No lose samples found in HOLDOUT! Check data distribution.")
            
            # Early stopping based on performance metric (avg_win_prob - avg_lose_prob)
            if performance > best_metric:
                best_metric = performance
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {ep+1}:")
                        print(f"  No improvement in {early_stopping_patience} epochs")
                        print(f"  Best performance: {best_metric:.4f}")
                    break
        
        # Store training history for analysis
        self.training_history = history
        
        if verbose:
            print("\n" + "="*50)
            print("Training Complete!")
            print(f"Final Performance: {history['performance'][-1]:.4f}")
            if history['lose_prob'][-1] > 0:
                print(f"Final Win/Lose Ratio: {history['win_prob'][-1]/max(0.001, history['lose_prob'][-1]):.2f}x")
            else:
                print("Warning: No lose samples in final epoch")
            print("="*50)
        
        self._owner._net.eval()
        return self

    def predict(self, x_list: List[np.ndarray], verbose: int = 0) -> np.ndarray:
        hands, discs, called, gss = x_list
        hand_seq_np, calls_seq_np, disc_seq_np, gsv_np = self._precompute_cnn(hands, discs, called, gss)
        with torch.no_grad():
            logits = self._owner._net(
                torch.from_numpy(hand_seq_np).to(self._owner._device),
                torch.from_numpy(calls_seq_np).to(self._owner._device),
                torch.from_numpy(disc_seq_np).to(self._owner._device),
                torch.from_numpy(gsv_np).to(self._owner._device),
            )
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

