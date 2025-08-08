#!/usr/bin/env python3
"""
Example script demonstrating how to use the TrainingDataGenerator
"""

import sys
import os
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training_data_generator import TrainingDataGenerator
from core.game import PQNetwork, TENSORFLOW_AVAILABLE


def example_generation_0():
    """Example: Generate training data with random policies (Generation 0)"""
    print("=== Generation 0: Random Policies ===\n")
    
    # Create a training data generator with random policies
    generator = TrainingDataGenerator(
        generation=0,
        pq_network=None,  # Use random policies for generation 0
        simulation_count=200,  # Moderate simulation count
        exploration_constant=1.414,
        max_games=50,  # Small number for example
        save_interval=10
    )
    
    print(f"Generation directory: {generator.generation_dir}")
    print(f"Data directory: {generator.data_dir}")
    print("Generating training data with random policies...")
    
    features, policies, values = generator.generate_training_data(
        output_file="example_gen0_data.npz"
    )
    
    print(f"Generated {len(features)} training samples")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Policy labels shape: {policies.shape}")
    print(f"Value labels shape: {values.shape}")
    
    return features, policies, values


def example_generation_1():
    """Example: Generate training data with PQNetwork (Generation 1)"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Skipping PQNetwork example.")
        return None
    
    print("\n=== Generation 1: PQNetwork Integration ===\n")
    
    try:
        # Create a PQNetwork
        pq_network = PQNetwork(hidden_size=64, embedding_dim=4, max_turns=20)
        print("PQNetwork created successfully!")
        
        # Create a training data generator with PQNetwork
        generator = TrainingDataGenerator(
            generation=1,
            pq_network=pq_network,  # Use PQNetwork for policy/value estimation
            simulation_count=300,  # Higher simulation count for better quality
            exploration_constant=1.414,
            max_games=30,  # Smaller number due to slower generation
            save_interval=10
        )
        
        print(f"Generation directory: {generator.generation_dir}")
        print(f"Data directory: {generator.data_dir}")
        print("Generating training data with PQNetwork...")
        
        features, policies, values = generator.generate_training_data(
            output_file="example_gen1_data.npz"
        )
        
        print(f"Generated {len(features)} training samples")
        print(f"Feature matrix shape: {features.shape}")
        print(f"Policy labels shape: {policies.shape}")
        print(f"Value labels shape: {values.shape}")
        
        return features, policies, values
        
    except Exception as e:
        print(f"Error in PQNetwork example: {e}")
        return None


def example_load_and_analyze():
    """Example: Load and analyze generated training data"""
    print("\n=== Loading and Analyzing Training Data ===\n")
    
    try:
        # Load generation 0 data
        gen0_data_path = "training_data/generation_0/data/example_gen0_data.npz"
        if os.path.exists(gen0_data_path):
            data = np.load(gen0_data_path)
            features = data['features']
            policies = data['policies']
            values = data['values']
            
            print("Generation 0 Data Analysis:")
            print(f"  Features: {features.shape}")
            print(f"  Policies: {policies.shape}")
            print(f"  Values: {values.shape}")
            print(f"  Feature range: [{np.min(features):.4f}, {np.max(features):.4f}]")
            print(f"  Policy sum to 1: {np.allclose(np.sum(policies, axis=1), 1.0)}")
            print(f"  Value range: [{np.min(values):.4f}, {np.max(values):.4f}]")
            
            # Calculate some statistics
            print(f"  Average value: {np.mean(values):.4f}")
            print(f"  Value std: {np.std(values):.4f}")
            print(f"  Most common actions: {np.argmax(policies, axis=1)[:10]}")
        
        # Load generation 1 data
        gen1_data_path = "training_data/generation_1/data/example_gen1_data.npz"
        if os.path.exists(gen1_data_path):
            data = np.load(gen1_data_path)
            features = data['features']
            policies = data['policies']
            values = data['values']
            
            print("\nGeneration 1 Data Analysis:")
            print(f"  Features: {features.shape}")
            print(f"  Policies: {policies.shape}")
            print(f"  Values: {values.shape}")
            print(f"  Feature range: [{np.min(features):.4f}, {np.max(features):.4f}]")
            print(f"  Policy sum to 1: {np.allclose(np.sum(policies, axis=1), 1.0)}")
            print(f"  Value range: [{np.min(values):.4f}, {np.max(values):.4f}]")
            
            # Calculate some statistics
            print(f"  Average value: {np.mean(values):.4f}")
            print(f"  Value std: {np.std(values):.4f}")
            print(f"  Most common actions: {np.argmax(policies, axis=1)[:10]}")
            
    except Exception as e:
        print(f"Error loading data: {e}")


def example_directory_structure():
    """Example: Show the directory structure"""
    print("\n=== Directory Structure ===\n")
    
    base_dir = "training_data"
    if os.path.exists(base_dir):
        print("Training data directory structure:")
        for root, dirs, files in os.walk(base_dir):
            level = root.replace(base_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    else:
        print("Training data directory not found yet.")


def main():
    """Main function running all examples"""
    print("=== Training Data Generator Examples ===\n")
    
    # Example 1: Generation 0 with random policies
    gen0_data = example_generation_0()
    
    # Example 2: Generation 1 with PQNetwork
    gen1_data = example_generation_1()
    
    # Example 3: Show directory structure
    example_directory_structure()
    
    # Example 4: Load and analyze data
    example_load_and_analyze()
    
    print("\n=== Examples Completed ===")
    print("Check the generated files in the training_data directory:")
    print("  - training_data/generation_0/data/")
    print("  - training_data/generation_1/data/")
    print("  - training_data/models/")
    print("  - training_data/logs/")


if __name__ == "__main__":
    main()
