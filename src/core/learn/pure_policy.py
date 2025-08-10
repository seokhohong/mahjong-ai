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
        sample_weight: Optional[Dict[str, np.ndarray]] = None,
        early_stopping_patience: int = 5,
    ):
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
        weights_np = None
        if sample_weight and 'policy_flat' in sample_weight:
            weights_np = sample_weight['policy_flat'].astype(np.float32)

        device = self._owner._device
        xb = torch.from_numpy(xb_np).to(device)
        yb_all = torch.from_numpy(labels_np).to(device)
        wb_all = torch.from_numpy(weights_np).to(device) if weights_np is not None else None

        # Optimizer and policy-gradient loss
        optimizer = torch.optim.Adam(self._owner._net.parameters(), lr=1e-4)
        def policy_gradient_loss(logits: torch.Tensor, actions: torch.Tensor, advantages) -> torch.Tensor:
            # Clamp logits to avoid extreme gradients
            logits = torch.clamp(logits, min=-5.0, max=5.0)
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, actions.view(-1, 1)).squeeze(1)
            return -(action_log_probs * advantages).mean()
        self._owner._net.train()

        from tqdm import tqdm  # type: ignore

        num_samples = xb.shape[0]
        bs = max(1, min(batch_size, num_samples))

        best_acc: float = -1.0
        epochs_no_improve: int = 0
        for ep in range(max(1, epochs)):
            indices = torch.arange(num_samples, device=device)
            if shuffle:
                indices = indices[torch.randperm(num_samples, device=device)]
            total_batches = (num_samples + bs - 1) // bs
            running_loss = 0.0
            naive_correct = 0.0
            naive_samples = 0
            win_correct = 0.0
            win_samples = 0
            bar = None
            if verbose:
                bar = tqdm(range(total_batches), desc=f"Epoch {ep+1}/{epochs}", leave=False) if tqdm else None
                if bar is None:
                    print(f"Epoch {ep+1}/{epochs}")
            bi = 0
            for start in range(0, num_samples, bs):
                end = min(num_samples, start + bs)
                idx = indices[start:end]
                xb_b = xb.index_select(0, idx)
                yb = yb_all.index_select(0, idx)
                wb = wb_all.index_select(0, idx)
                optimizer.zero_grad(set_to_none=True)
                logits = self._owner._net(xb_b)
                loss = policy_gradient_loss(logits, yb, wb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._owner._net.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += float(loss.detach().cpu().item())
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=-1)
                    eq = (preds == yb).float()
                    naive_correct += float(eq.sum().item())
                    naive_samples += int(yb.shape[0])
                    # win_acc: only count samples with reward weight exactly +1
                    if wb is not None:
                        win_mask = (wb == 1.0)
                        if win_mask.any():
                            win_correct += float(eq[win_mask].sum().item())
                            win_samples += int(win_mask.sum().item())

                bi += 1
                if verbose:
                    naive_val = naive_correct / max(1.0, float(naive_samples))
                    win_val = (win_correct / float(win_samples)) if win_samples > 0 else 0.0
                    if bar is not None:
                        bar.set_postfix({
                            "loss": f"{running_loss/bi:.4f}",
                            "naive_acc": f"{naive_val:.4f}",
                            "win_acc": f"{win_val:.4f}",
                        })
                        bar.update(1)
                    elif bi % 10 == 0:
                        print(f"  step {bi:4d}/{total_batches} - loss: {running_loss/bi:.4f} - naive_acc: {naive_val:.4f} - win_acc: {win_val:.4f}")
            epoch_naive = (naive_correct / max(1.0, float(naive_samples)))
            epoch_win = (win_correct / float(win_samples)) if win_samples > 0 else 0.0
            if verbose and bar is not None:
                bar.close()
                print(f"Epoch {ep+1}/{epochs} - avg loss: {running_loss/max(1, bi):.4f} - naive_acc: {epoch_naive:.4f} - win_acc: {epoch_win:.4f}")

            # Early stopping on win_acc plateau
            if epoch_win > best_acc:
                best_acc = epoch_win
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {ep+1}: no win_acc improvement in {early_stopping_patience} epoch(s). Best win_acc={best_acc:.4f}")
                    break

        self._owner._net.eval()
        return self

    def predict(self, x_list: List[np.ndarray], verbose: int = 0) -> np.ndarray:
        hands, discs, called, gss = x_list
        xb_np = self._precompute_flat(hands, discs, called, gss)
        with torch.no_grad():
            logits = self._owner._net(torch.from_numpy(xb_np).to(self._owner._device))
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs


class OptimizedKerasLikeWrapper:
    """Optimized wrapper with proper PyTorch training pipeline."""

    def __init__(self, owner):
        self._owner = owner

    def fit(self, x_list: List[np.ndarray], y: Dict[str, np.ndarray], 
            epochs: int = 1, batch_size: int = 8, verbose: int = 0, 
            shuffle: bool = True, sample_weight: Optional[Dict[str, np.ndarray]] = None):
        
        hands, discs, called, gss = x_list
        
        y_flat = y.get('policy_flat') if isinstance(y, dict) else None
        if y_flat is None:
            return self

        print("Preprocessing features (this should happen once)...")
        
        # OPTIMIZATION 1: Preprocess all features ONCE, not per epoch
        all_features = self._preprocess_all_features(hands, discs, called, gss)
        all_labels = torch.from_numpy(np.argmax(y_flat, axis=1)).long()
        
        # Handle sample weights
        weights_tensor = None
        if sample_weight and 'policy_flat' in sample_weight:
            weights_tensor = torch.from_numpy(sample_weight['policy_flat']).float()

        # OPTIMIZATION 2: Use PyTorch DataLoader for efficient batching
        if weights_tensor is not None:
            dataset = TensorDataset(all_features, all_labels, weights_tensor)
        else:
            dataset = TensorDataset(all_features, all_labels)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            pin_memory=True,  # Faster CPU->GPU transfers
            num_workers=0     # Can increase if CPU preprocessing is heavy
        )

        # Setup training
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(self._owner._net.parameters(), lr=1e-3)
        self._owner._net.train()

        for epoch_idx in range(epochs):
            running_loss = 0.0
            naive_correct = 0.0
            naive_samples = 0
            win_correct = 0.0
            win_samples = 0
            
            if verbose:
                pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1}/{epochs}")
            else:
                pbar = dataloader

            for batch_data in pbar:
                if weights_tensor is not None:
                    xb, yb, wb = batch_data
                    wb = wb.to(self._owner._device, non_blocking=True)
                else:
                    xb, yb = batch_data
                    wb = None
                
                # OPTIMIZATION 3: Move to GPU with non_blocking for speed
                xb = xb.to(self._owner._device, non_blocking=True)
                yb = yb.to(self._owner._device, non_blocking=True)
                
                # Forward pass
                optimizer.zero_grad(set_to_none=True)
                logits = self._owner._net(xb)
                
                # Loss calculation
                loss_vec = criterion(logits, yb)
                if wb is not None:
                    loss = (loss_vec * wb).mean()
                else:
                    loss = loss_vec.mean()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                batch_size_actual = xb.size(0)
                running_loss += loss.item() * batch_size_actual
                
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=-1)
                    eq = (preds == yb).float()
                    naive_correct += float(eq.sum().item())
                    naive_samples += int(batch_size_actual)
                    if wb is not None:
                        win_mask = (wb == 1.0)
                        if win_mask.any():
                            win_correct += float(eq[win_mask].sum().item())
                            win_samples += int(win_mask.sum().item())
                
                if verbose:
                    naive_acc = naive_correct / max(1, naive_samples)
                    win_acc = (win_correct / win_samples) if win_samples > 0 else 0.0
                    pbar.set_postfix({
                        "loss": f"{running_loss/max(1, naive_samples):.4f}", 
                        "naive_acc": f"{naive_acc:.4f}",
                        "win_acc": f"{win_acc:.4f}"
                    })
            
            if verbose:
                final_loss = running_loss / max(1, naive_samples)
                final_naive = naive_correct / max(1, naive_samples)
                final_win = (win_correct / win_samples) if win_samples > 0 else 0.0
                print(f"Epoch {epoch_idx+1}/{epochs} - loss: {final_loss:.4f} - naive_acc: {final_naive:.4f} - win_acc: {final_win:.4f}")

        self._owner._net.eval()
        return self

    def _preprocess_all_features(self, hands: np.ndarray, discs: np.ndarray, 
                                called: np.ndarray, gss: np.ndarray) -> torch.Tensor:
        """
        OPTIMIZATION 4: Vectorized feature preprocessing
        Convert all samples to features in one go, not sample by sample
        """
        num_samples = hands.shape[0]
        flat_features = []
        
        # Process in larger chunks for memory efficiency
        chunk_size = min(1000, num_samples)
        
        for start_idx in range(0, num_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, num_samples)
            chunk_features = []
            
            for i in range(start_idx, end_idx):
                # This is still per-sample, but at least it's done once
                feats = self._owner._extract_features_from_indexed(
                    hands[i], discs[i], called[i], gss[i]
                )
                # Flatten features
                parts = [feats[0].reshape(-1)]
                for j in range(1, 5):
                    parts.append(feats[j].reshape(-1))
                parts.append(feats[5].reshape(-1))
                
                chunk_features.append(np.concatenate(parts, axis=0))
            
            flat_features.extend(chunk_features)
        
        # Convert to tensor and move to GPU once
        features_tensor = torch.from_numpy(
            np.array(flat_features, dtype=np.float32)
        ).to(self._owner._device)
        
        return features_tensor

    def predict(self, x_list: List[np.ndarray], verbose: int = 0) -> np.ndarray:
        """Optimized prediction with batching"""
        hands, discs, called, gss = x_list
        
        # Preprocess features once
        all_features = self._preprocess_all_features(hands, discs, called, gss)
        
        # Use DataLoader for efficient batching
        dataset = TensorDataset(all_features)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        results = []
        self._owner._net.eval()
        
        with torch.no_grad():
            for (xb,) in dataloader:
                logits = self._owner._net(xb)
                probs = torch.softmax(logits, dim=-1)
                results.append(probs.cpu().numpy())
        
        return np.concatenate(results, axis=0)


# OPTIMIZATION 5: Even better - precompute features completely outside training
class PrecomputedDataset(torch.utils.data.Dataset):
    """Dataset that holds precomputed features to avoid repeated computation"""
    
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, weights: Optional[torch.Tensor] = None):
        self.features = features
        self.labels = labels
        self.weights = weights
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.weights is not None:
            return self.features[idx], self.labels[idx], self.weights[idx]
        return self.features[idx], self.labels[idx]


def precompute_features_once(pure_policy_net, hands, discs, called, gss, y_flat, sample_weight=None):
    """
    BEST OPTIMIZATION: Compute features once and reuse for all epochs
    """
    print("Computing features once for all epochs...")
    
    num_samples = hands.shape[0]
    flat_features = []
    
    for i in tqdm(range(num_samples), desc="Feature extraction"):
        feats = pure_policy_net._extract_features_from_indexed(
            hands[i], discs[i], called[i], gss[i]
        )
        parts = [feats[0].reshape(-1)]
        for j in range(1, 5):
            parts.append(feats[j].reshape(-1))
        parts.append(feats[5].reshape(-1))
        flat_features.append(np.concatenate(parts, axis=0))
    
    # Convert everything to tensors
    features_tensor = torch.from_numpy(np.array(flat_features, dtype=np.float32))
    labels_tensor = torch.from_numpy(np.argmax(y_flat, axis=1)).long()
    
    weights_tensor = None
    if sample_weight and 'policy_flat' in sample_weight:
        weights_tensor = torch.from_numpy(sample_weight['policy_flat']).float()
    
    return PrecomputedDataset(features_tensor, labels_tensor, weights_tensor)


# Example usage:
def efficient_training_example(pure_policy_net, hands, discs, called, gss, y_flat):
    # Method 1: Use optimized wrapper
    optimized_model = OptimizedKerasLikeWrapper(pure_policy_net)
    optimized_model.fit([hands, discs, called, gss], {'policy_flat': y_flat}, 
                       epochs=10, batch_size=32, verbose=1)
    
    # Method 2: Even better - precompute once
    dataset = precompute_features_once(pure_policy_net, hands, discs, called, gss, y_flat)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
    
    # Now training is just pure PyTorch - very fast!
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pure_policy_net._net.parameters(), lr=1e-3)
    
    for epoch in range(10):
        for batch in dataloader:
            if len(batch) == 3:
                features, labels, weights = batch
                features, labels, weights = features.cuda(), labels.cuda(), weights.cuda()
            else:
                features, labels = batch
                features, labels = features.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            logits = pure_policy_net._net(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()