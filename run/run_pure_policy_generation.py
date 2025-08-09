from core.learn.pure_policy_dataset import generate_pure_policy_dataset
import os

if __name__ == "__main__":
    generate_pure_policy_dataset(3, 123, out_path=os.path.join(os.path.dirname(__file__), 'training_data','debug_pure_policy_test.npz'))