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
        # Build a minimal MLP over flattened features extracted by this class
        # Feature dimension: (12*5) + (4*max_turns*embedding_dim) + 50
        self._flat_dim = (12 * 5) + (4 * self.max_turns * self.embedding_dim) + 50
        self._net = nn.Sequential(
            nn.Linear(self._flat_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, self._num_actions),
        )
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
        # Flatten to match MLP input
        parts: List[np.ndarray] = [features[0].reshape(-1)]
        for i in range(1, 5):
            parts.append(features[i].reshape(-1))
        parts.append(features[5].reshape(-1))
        x = np.concatenate(parts, axis=0)[None, :].astype(np.float32)
        with torch.no_grad():
            logits = self._net(torch.from_numpy(x).to(self._device))
            probs = F.softmax(logits, dim=-1).numpy()
        return {'policy': probs[0]}

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
        if not filepath.endswith('.pt'):
            filepath += '.pt'
        torch.save(self._net.state_dict(), filepath)

    def load_model(self, filepath: str) -> None:
        if not filepath.endswith('.pt'):
            filepath += '.pt'
        state = torch.load(filepath, map_location='cpu')
        self._net.load_state_dict(state)

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

    def _precompute_flat(self, hands: np.ndarray, discs: np.ndarray, called: np.ndarray, gss: np.ndarray) -> np.ndarray:
        """Vectorized precompute of flattened features for an entire batch."""
        # Hands: (N, 12) -> embeddings (N,12,embed), called flag (N,12,1)
        hands = np.asarray(hands, dtype=np.int32)
        hand_emb = self._owner._embedding_table[np.clip(hands, 0, 18)]  # (N,12,embed)
        # Called flags: build presence per sample from called sets (N,4,sets,3)
        called = np.asarray(called, dtype=np.int32)
        if called.ndim == 4:
            # Only row 0 (current player perspective) is used for flags
            cs0 = called[:, 0, :, :]  # (N, sets, 3)
            # Convert to presence [N,19]
            codes = np.arange(19, dtype=np.int32)
            presence = (cs0[..., None] == codes[None, None, None, :]).any(axis=(1, 2))  # (N,19)
            called_flag = presence[np.arange(hands.shape[0])[:, None], np.clip(hands, 0, 18)] & (hands > 0)
        else:
            called_flag = np.zeros(hands.shape, dtype=bool)
        hand_feat = np.concatenate([hand_emb, called_flag.astype(np.float32)[..., None]], axis=-1)  # (N,12,5)

        # Discards: (N,4,maxT) -> embeddings (N,4,maxT,embed)
        discs = np.asarray(discs, dtype=np.int32)
        disc_emb = self._owner._embedding_table[np.clip(discs, 0, 18)]  # (N,4,maxT_in,embed)
        # Conform time dimension to network's max_turns via slice/pad
        maxT = self._owner.max_turns
        curT = disc_emb.shape[2] if disc_emb.ndim >= 3 else 0
        if curT > maxT:
            disc_emb = disc_emb[:, :, :maxT, :]
        elif curT < maxT:
            pad_amt = maxT - curT
            pad = np.zeros((disc_emb.shape[0], disc_emb.shape[1], pad_amt, disc_emb.shape[3]), dtype=np.float32)
            disc_emb = np.concatenate([disc_emb, pad], axis=2)

        # Game state: (N,GSV)
        from ..constants import GAME_STATE_VEC_LEN as GSV
        gss = np.asarray(gss, dtype=np.float32)
        if gss.ndim == 1:
            gss = gss[None, :]
        if gss.shape[1] < GSV:
            pad = np.zeros((gss.shape[0], GSV - gss.shape[1]), dtype=np.float32)
            gss = np.concatenate([gss, pad], axis=1)
        elif gss.shape[1] > GSV:
            gss = gss[:, :GSV]

        # Flatten to single vector per sample
        N = hands.shape[0]
        from ..constants import GAME_STATE_VEC_LEN as GSV
        flat_dim = (12 * 5) + (4 * self._owner.max_turns * self._owner.embedding_dim) + GSV
        flat = np.empty((N, flat_dim), dtype=np.float32)
        # Hand
        flat[:, 0:12*5] = hand_feat.reshape(N, 12 * 5)
        # Discards per player
        offset = 12 * 5
        for i in range(4):
            start = offset + i * (self._owner.max_turns * self._owner.embedding_dim)
            end = start + (self._owner.max_turns * self._owner.embedding_dim)
            if i < disc_emb.shape[1]:
                flat[:, start:end] = disc_emb[:, i, :, :].reshape(N, -1)
            else:
                flat[:, start:end] = 0.0
        # Game state
        flat[:, -GSV:] = gss[:, :GSV]
        return flat

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

        # Precompute flattened features and labels
        xb_np = self._precompute_flat(hands, discs, called, gss)
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
        xb = torch.from_numpy(xb_np).to(device)
        yb_all = torch.from_numpy(labels_np).to(device)
        # Legality masks (required)
        try:
            lm_all = torch.from_numpy(legality_masks.astype(bool)).to(device)
            if lm_all.ndim != 2 or lm_all.shape[0] != xb.shape[0] or lm_all.shape[1] != self._owner._num_actions:
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

        num_samples = xb.shape[0]
        bs = max(1, min(batch_size, num_samples))

        # Tracking metrics
        best_metric = -float('inf')
        epochs_no_improve = 0
        history = {
            'loss': [], 'entropy': [], 'win_prob': [], 'lose_prob': [], 
            'neutral_prob': [], 'performance': [], 'win_count': [], 'lose_count': [], 'neutral_count': []
        }

        # Precompute validation features/labels/weights if provided
        val_flat: Optional[np.ndarray] = None
        val_labels: Optional[np.ndarray] = None
        val_weights: Optional[np.ndarray] = None
        if val_x_list is not None and val_targets is not None:
            vh, vd, vc, vg = val_x_list
            vy_flat = val_targets.get('policy_flat') if isinstance(val_targets, dict) else None
            if vy_flat is not None:
                val_flat = self._precompute_flat(vh, vd, vc, vg)
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
                xb_b = xb.index_select(0, idx)
                yb = yb_all.index_select(0, idx)
                
                # Use RAW rewards for training (policy gradient)
                wb = wb_all.index_select(0, idx)
                orig_rewards = orig_rewards_all.index_select(0, idx)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass
                logits = self._owner._net(xb_b)
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
                print(f"  Train -> CE: {train_ce:.4f} | PG: {train_pg:.4f}")
                print(f"  Sample Counts -> Win: {win_count} | Lose: {lose_count} | Neutral: {neutral_count}")
                print(f"  Policy Probs -> Win: {avg_win_prob:.4f} | Lose: {avg_lose_prob:.4f} | Neutral: {avg_neutral_prob:.4f}")
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

                    print("  Train (wins) top actions:")
                    for rank_i, ai in enumerate(top_idx, 1):
                        print(f"    {rank_i}. {_pretty(labels[ai])}: {avg_action_probs_win[ai]:.4f}")
                    # Also report ron and tsumo average probabilities over wins
                    try:
                        ron_idx = labels.index('ron')
                        tsumo_idx = labels.index('tsumo')
                    except ValueError:
                        ron_idx, tsumo_idx = 1, 2  # Fallback to standard indices
                    print(f"  Train (wins) ron: {avg_action_probs_win[ron_idx]:.4f} | tsumo: {avg_action_probs_win[tsumo_idx]:.4f}")
                
                if avg_win_prob > 0 and avg_lose_prob > 0:
                    ratio = avg_win_prob / avg_lose_prob
                    print(f"  Win/Lose Ratio: {ratio:.2f}x | Performance: {performance:.4f}")
                elif avg_lose_prob == 0:
                    print(f"  WARNING: No lose samples found in this epoch! Check data distribution.")
                
                print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

                # Validation metrics per epoch if provided
                if val_flat is not None and val_labels is not None:
                    with torch.no_grad():
                        logits_val = self._owner._net(torch.from_numpy(val_flat).to(device))
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
                    print(f"  Holdout -> CE: {ce_val:.4f} | PG: {pg_val:.4f}")
                    print(f"            Sample Counts -> Win: {ho_win_count} | Lose: {ho_lose_count} | Neutral: {ho_neutral_count}")
                    print(f"            Policy Probs -> Win: {ho_win_prob:.4f} | Lose: {ho_lose_prob:.4f} | Neutral: {ho_neutral_prob:.4f}")
                    if ho_win_prob > 0 and ho_lose_prob > 0:
                        print(f"            Win/Lose Ratio: {ho_win_prob/ho_lose_prob:.2f}x | Performance: {ho_performance:.4f}")
                    elif ho_lose_prob == 0:
                        print(f"            WARNING: No lose samples found in HOLDOUT! Check data distribution.")
            
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
        xb_np = self._precompute_flat(hands, discs, called, gss)
        with torch.no_grad():
            logits = self._owner._net(torch.from_numpy(xb_np).to(self._owner._device))
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

