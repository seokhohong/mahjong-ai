from __future__ import annotations

import os
from typing import Tuple

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix  # type: ignore
    import joblib  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError('scikit-learn is required for policy_trainer_alt. Please install scikit-learn.') from e


def _tabular_from_states(states_obj: np.ndarray) -> np.ndarray:
    """Flatten indexed state dicts into a single tabular feature vector.

    Features (concatenated):
    - hand_idx: 12 ints
    - disc_idx: 4x50 ints (flattened)
    - game_state: 50 floats
    Returns shape: (N, 12 + 200 + 50) = (N, 262)
    """
    features: list[list[float]] = []
    for s in states_obj:
        sd = s.item() if hasattr(s, 'item') else s
        hand = np.asarray(sd['hand_idx'], dtype=np.int32)  # (12,)
        disc = np.asarray(sd['disc_idx'], dtype=np.int32).reshape(-1)  # (200,)
        gs = np.asarray(sd['game_state'], dtype=np.float32)  # (50,)
        fv = np.concatenate([
            hand.astype(np.float32),
            disc.astype(np.float32),
            gs,
        ], axis=0)
        features.append(fv.tolist())
    return np.asarray(features, dtype=np.float32)


def train_random_forest(
    dataset_path: str,
    model_out: str | None = None,
    n_estimators: int = 200,
    max_depth: int | None = 25,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[float, str | None]:
    """Train a RandomForest on the pure-policy dataset and report action-head accuracy.

    Returns (test_accuracy, saved_model_path_or_None).
    """
    if not dataset_path.endswith('.npz'):
        raise ValueError('dataset_path must be a .npz file produced by generate_pure_policy_dataset')
    data = np.load(dataset_path, allow_pickle=True)

    states = data['states']            # object array of dicts
    y_action_1hot = data['y_action']   # (N, 5)

    X = _tabular_from_states(states)   # (N, 262)
    y = np.argmax(y_action_1hot, axis=1).astype(np.int32)  # (N,)

    if X.shape[0] < 2:
        # Too small to split; train and evaluate on the same set
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
        )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    labels = [0, 1, 2, 3, 4]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    saved_path = None
    if model_out:
        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        if not model_out.endswith('.joblib'):
            model_out += '.joblib'
        joblib.dump(clf, model_out)
        saved_path = model_out

    print(f'RandomForest action-head accuracy: {acc:.4f}')
    print(f'RandomForest action-head F1 (macro): {f1_macro:.4f}')
    print('Action classes: 0=discard, 1=ron, 2=tsumo, 3=pon, 4=chi')
    print('Confusion matrix (rows=true, cols=pred):')
    # Pretty print confusion matrix
    with np.printoptions(linewidth=200):
        print(cm)
    if saved_path:
        print(f'Saved RandomForest model to: {saved_path}')
    return acc, saved_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a RandomForest on pure-policy dataset and report accuracy')
    parser.add_argument('--data', required=True, help='Path to .npz dataset from generate_pure_policy_dataset')
    parser.add_argument('--out', default=None, help='Output model path (.joblib). Optional')
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=25)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    acc, path = train_random_forest(
        dataset_path=args.data,
        model_out=args.out,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        test_size=args.test_size,
        random_state=args.seed,
    )
    print(f'Accuracy: {acc:.4f}')


