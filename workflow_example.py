#!/usr/bin/env python3
"""
Complete workflow example demonstrating the training pipeline
"""

import sys
import os
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training_data_generator import TrainingDataGenerator
from core.game import PQNetwork, TENSORFLOW_AVAILABLE


def workflow_generation_0():
    """Step 1: Generate training data with random policies (Generation 0)"""
    print("=== Step 1: Generation 0 - Random Policies ===\n")
    
    # Create generator for generation 0
    generator = TrainingDataGenerator(
        generation=0,
        pq_network=None,  # Use random policies
        simulation_count=500,  # Moderate simulation count
        max_games=100,  # Small number for example
        save_interval=20
    )
    
    print(f"Creating generation 0 in: {generator.generation_dir}")
    
    # Generate training data
    features, policies, values = generator.generate_training_data(
        output_file="gen0_training_data.npz"
    )
    
    print(f"✓ Generated {len(features)} training samples")
    print(f"✓ Data saved in: {generator.data_dir}")
    
    return generator, features, policies, values


def workflow_train_pqnetwork(generator, features, policies, values):
    """Step 2: Train PQNetwork on generation 0 data"""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Skipping PQNetwork training.")
        return None
    
    print("\n=== Step 2: Training PQNetwork ===\n")
    
    try:
        # Create PQNetwork
        pq_network = PQNetwork(hidden_size=64, embedding_dim=4, max_turns=20)
        print("✓ PQNetwork created")
        
        # Prepare training data
        # Note: This is a simplified example. In practice, you'd need to convert
        # the numpy arrays back to GameState objects for PQNetwork training
        print("⚠️  Note: This is a simplified example. Full training would require")
        print("   converting numpy arrays back to GameState objects.")
        
        # Save the model
        model_filename = "gen0_trained_model.keras"
        generator.save_model(pq_network, model_filename)
        print(f"✓ Model saved as: {model_filename}")
        
        return pq_network
        
    except Exception as e:
        print(f"✗ Error training PQNetwork: {e}")
        return None


def workflow_generation_1(generator, pq_network):
    """Step 3: Generate training data with trained PQNetwork (Generation 1)"""
    if not pq_network:
        print("No PQNetwork available. Skipping generation 1.")
        return None
    
    print("\n=== Step 3: Generation 1 - PQNetwork Integration ===\n")
    
    # Create generator for generation 1
    gen1_generator = TrainingDataGenerator(
        generation=1,
        pq_network=pq_network,  # Use trained PQNetwork
        simulation_count=800,  # Higher simulation count
        max_games=80,  # Smaller number due to slower generation
        save_interval=20
    )
    
    print(f"Creating generation 1 in: {gen1_generator.generation_dir}")
    
    # Generate improved training data
    features, policies, values = gen1_generator.generate_training_data(
        output_file="gen1_training_data.npz"
    )
    
    print(f"✓ Generated {len(features)} improved training samples")
    print(f"✓ Data saved in: {gen1_generator.data_dir}")
    
    return gen1_generator, features, policies, values


def workflow_compare_generations():
    """Step 4: Compare data from different generations"""
    print("\n=== Step 4: Comparing Generations ===\n")
    
    try:
        # Load generation 0 data
        gen0_path = "training_data/generation_0/data/gen0_training_data.npz"
        if os.path.exists(gen0_path):
            gen0_data = np.load(gen0_path)
            gen0_features = gen0_data['features']
            gen0_policies = gen0_data['policies']
            gen0_values = gen0_data['values']
            
            print("Generation 0 Statistics:")
            print(f"  Samples: {len(gen0_features)}")
            print(f"  Features: {gen0_features.shape[1]}")
            print(f"  Value range: [{np.min(gen0_values):.4f}, {np.max(gen0_values):.4f}]")
            print(f"  Average value: {np.mean(gen0_values):.4f}")
        
        # Load generation 1 data
        gen1_path = "training_data/generation_1/data/gen1_training_data.npz"
        if os.path.exists(gen1_path):
            gen1_data = np.load(gen1_path)
            gen1_features = gen1_data['features']
            gen1_policies = gen1_data['policies']
            gen1_values = gen1_data['values']
            
            print("\nGeneration 1 Statistics:")
            print(f"  Samples: {len(gen1_features)}")
            print(f"  Features: {gen1_features.shape[1]}")
            print(f"  Value range: [{np.min(gen1_values):.4f}, {np.max(gen1_values):.4f}]")
            print(f"  Average value: {np.mean(gen1_values):.4f}")
            
            if os.path.exists(gen0_path):
                print("\nComparison:")
                print(f"  Feature dimensions: {gen0_features.shape[1]} → {gen1_features.shape[1]}")
                print(f"  Value improvement: {np.mean(gen1_values) - np.mean(gen0_values):.4f}")
        
    except Exception as e:
        print(f"✗ Error comparing generations: {e}")


def workflow_show_directory_structure():
    """Step 5: Show the final directory structure"""
    print("\n=== Step 5: Directory Structure ===\n")
    
    base_dir = "training_data"
    if os.path.exists(base_dir):
        print("Final training data directory structure:")
        for root, dirs, files in os.walk(base_dir):
            level = root.replace(base_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    else:
        print("Training data directory not found.")


def main():
    """Main workflow function"""
    print("=== Complete Training Pipeline Workflow ===\n")
    
    # Step 1: Generation 0
    gen0_generator, gen0_features, gen0_policies, gen0_values = workflow_generation_0()
    
    # Step 2: Train PQNetwork
    pq_network = workflow_train_pqnetwork(gen0_generator, gen0_features, gen0_policies, gen0_values)
    
    # Step 3: Generation 1
    gen1_generator, gen1_features, gen1_policies, gen1_values = workflow_generation_1(gen0_generator, pq_network)
    
    # Step 4: Compare generations
    workflow_compare_generations()
    
    # Step 5: Show directory structure
    workflow_show_directory_structure()
    
    print("\n=== Workflow Completed ===")
    print("✓ Training data organized by generation")
    print("✓ Models saved in training_data/models/")
    print("✓ Logs saved in training_data/logs/")
    print("✓ Data saved in training_data/generation_X/data/")


if __name__ == "__main__":
    main()
