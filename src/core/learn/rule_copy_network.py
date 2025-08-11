from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False

from src.core.constants import MAX_TURNS
from src.core.learn.pure_policy import PurePolicyNetwork


class RuleCopyNetwork:
    """
    Wrapper around PurePolicyNetwork that keeps the exact same architecture and
    feature structure but trains with standard, unweighted cross-entropy to copy
    a rules-based policy (supervised imitation).

    Exposes a `.model` facade with Keras-like `fit` and `predict` compatible
    with existing utilities, but the loss is plain cross-entropy and does not
    consume reward/sample weights. Legality masks are still honored during
    training by masking logits of illegal actions.
    """

    def __init__(self, hidden_size: int = 128, embedding_dim: int = 4, max_turns: int = MAX_TURNS):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RuleCopyNetwork. Please install torch.")
        # Underlying network is identical
        self._ppn = PurePolicyNetwork(hidden_size=hidden_size, embedding_dim=embedding_dim, max_turns=max_turns)
        # Provide a minimal Keras-like interface
        self.model = _KerasLikeCEWrapper(self._ppn)

    # Convenience delegates
    def save_model(self, filepath: str) -> None:
        self._ppn.save_model(filepath)

    def load_model(self, filepath: str) -> None:
        self._ppn.load_model(filepath)

    def to(self, device: torch.device) -> 'RuleCopyNetwork':  # pragma: no cover - thin accessor
        self._ppn.to(device)
        return self

    # For players/tests that want direct predict
    def predict(self, x_list: List[np.ndarray], verbose: int = 0) -> np.ndarray:
        return self._ppn.model.predict(x_list, verbose=verbose)


class _KerasLikeCEWrapper:
    """
    Keras-like facade over the underlying PurePolicyNetwork using unweighted
    cross-entropy loss to imitate labels (rule-based actions).
    """

    def __init__(self, owner: PurePolicyNetwork):
        self._owner = owner

    # Reuse the exact CNN precompute used by PurePolicyNetwork for parity
    def _precompute_cnn(self, hands: np.ndarray, discs: np.ndarray, called: np.ndarray, gss: np.ndarray):
        # Call the original wrapper's precompute to avoid divergence
        return self._owner.model._precompute_cnn(hands, discs, called, gss)  # type: ignore[attr-defined]

    def fit(
        self,
        x_list: List[np.ndarray],
        y: Dict[str, np.ndarray],
        epochs: int = 1,
        batch_size: int = 32,
        verbose: int = 0,
        shuffle: bool = True,
        sample_weight: Optional[np.ndarray] = None,  # ignored by design
        early_stopping_patience: int = 3,
        val_x_list: Optional[List[np.ndarray]] = None,
        val_targets: Optional[Dict[str, np.ndarray]] = None,
        val_sample_weight: Optional[np.ndarray] = None,  # ignored
        legality_masks: Optional[np.ndarray] = None,
        val_legality_masks: Optional[np.ndarray] = None,
    ) -> '_KerasLikeCEWrapper':
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

        device = self._owner._device  # type: ignore[attr-defined]
        hand_seq = torch.from_numpy(hand_seq_np).to(device)
        calls_seq = torch.from_numpy(calls_seq_np).to(device)
        disc_seq = torch.from_numpy(disc_seq_np).to(device)
        gsv_t = torch.from_numpy(gsv_np).to(device)
        yb_all = torch.from_numpy(labels_np).to(device)

        lm_all: Optional[torch.Tensor] = None
        if legality_masks is not None:
            lm_all = torch.from_numpy(legality_masks.astype(bool)).to(device)
            if lm_all.ndim != 2 or lm_all.shape[0] != hand_seq.shape[0] or lm_all.shape[1] != self._owner._num_actions:  # type: ignore[attr-defined]
                raise ValueError("legality_masks must have shape (N, num_actions)")

        # Optimizer
        optimizer = torch.optim.Adam(self._owner._net.parameters(), lr=1e-4)  # type: ignore[attr-defined]

        # Validation
        val_labels: Optional[np.ndarray] = None
        if val_x_list is not None and val_targets is not None and 'policy_flat' in val_targets:
            val_labels = np.argmax(val_targets['policy_flat'], axis=1).astype(np.int64)

        self._owner._net.train()  # type: ignore[attr-defined]
        best_val: float = float('inf')
        epochs_no_improve = 0

        num_samples = hand_seq.shape[0]
        bs = max(1, min(batch_size, num_samples))

        for ep in range(max(1, epochs)):
            # Shuffle indices
            indices = torch.arange(num_samples, device=device)
            if shuffle:
                indices = indices[torch.randperm(num_samples, device=device)]

            epoch_loss = 0.0
            batch_count = 0
            # Accuracy accumulators
            train_correct = 0
            train_total = 0

            if verbose:
                try:
                    from tqdm import tqdm as _tqdm  # type: ignore
                except Exception:
                    _tqdm = None  # type: ignore
                bar = _tqdm(range((num_samples + bs - 1) // bs), desc=f"Epoch {ep+1}/{epochs}", leave=False) if _tqdm else None
            else:
                bar = None

            bidx = 0
            for start in range(0, num_samples, bs):
                end = min(num_samples, start + bs)
                idx = indices[start:end]
                yb = yb_all.index_select(0, idx)

                optimizer.zero_grad(set_to_none=True)
                logits = self._owner._net(  # type: ignore[attr-defined]
                    hand_seq.index_select(0, idx),
                    calls_seq.index_select(0, idx),
                    disc_seq.index_select(0, idx),
                    gsv_t.index_select(0, idx),
                )
                # Mask illegal logits if provided
                if lm_all is not None:
                    lm = lm_all.index_select(0, idx)
                    logits = logits.masked_fill(~lm, -1e9)

                # Cross-entropy
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._owner._net.parameters(), max_norm=1.0)  # type: ignore[attr-defined]
                optimizer.step()

                epoch_loss += float(loss.item())
                batch_count += 1
                # Train accuracy
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=-1)
                    train_correct += int((preds == yb).sum().item())
                    train_total += int(yb.shape[0])
                bidx += 1
                if verbose and bar is not None:
                    bar.set_postfix({"loss": f"{epoch_loss/max(1,batch_count):.4f}"})
                    bar.update(1)

            if verbose and bar is not None:
                bar.close()

            # Validation CE
            val_ce = None
            val_acc = None
            if val_labels is not None and val_x_list is not None:
                with torch.no_grad():
                    vh, vd, vc, vg = val_x_list
                    h_np, c_np, d_np, g_np = self._precompute_cnn(vh, vd, vc, vg)
                    logits_val = self._owner._net(  # type: ignore[attr-defined]
                        torch.from_numpy(h_np).to(device),
                        torch.from_numpy(c_np).to(device),
                        torch.from_numpy(d_np).to(device),
                        torch.from_numpy(g_np).to(device),
                    )
                    if val_legality_masks is not None:
                        try:
                            lm_val = torch.from_numpy(val_legality_masks.astype(bool)).to(logits_val.device)
                            if lm_val.ndim == 2 and lm_val.shape[1] == logits_val.shape[1]:
                                logits_val = logits_val.masked_fill(~lm_val, -1e9)
                        except Exception:
                            pass
                    labels_t = torch.from_numpy(val_labels).to(logits_val.device)
                    val_ce = float(F.cross_entropy(logits_val, labels_t).cpu().item()) if labels_t.numel() > 0 else None
                    if labels_t.numel() > 0:
                        preds_val = torch.argmax(logits_val, dim=-1)
                        val_acc = float((preds_val == labels_t).float().mean().cpu().item())

            # Early stopping on validation CE if available; else on train loss
            metric_now = val_ce if val_ce is not None else (epoch_loss / max(1, batch_count))
            if metric_now < best_val:
                best_val = metric_now
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {ep+1}: best CE={best_val:.4f}")
                    break

            # Report epoch summary with accuracies
            if verbose:
                train_acc = (train_correct / max(1, train_total)) if train_total > 0 else 0.0
                acc_msg = f"train_acc={train_acc*100:.2f}%"
                if val_acc is not None:
                    acc_msg += f" | val_acc={val_acc*100:.2f}%"
                print(f"Epoch {ep+1}/{epochs} | CE={epoch_loss/max(1,batch_count):.4f} | {acc_msg}")

        self._owner._net.eval()  # type: ignore[attr-defined]
        return self

    def predict(self, x_list: List[np.ndarray], verbose: int = 0) -> np.ndarray:
        # Delegate to PurePolicyNetwork predict for identical behavior
        return self._owner.model.predict(x_list, verbose=verbose)



