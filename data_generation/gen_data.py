import functions as ag
import os
import math
# udělat rozmezí min, max délka výstupu, otestovat počty samplů pro každý problém (min. velikost kdy se naučí a kdy ne)
# Reset the example registry at the beginning to start fresh
ag.reset_example_registry()

# Example usage
if __name__ == "__main__":
    # Define the output directory
    output_dir = "data/mix_only/binary_addition"
    
    # Define the total number of examples
    total_train = 12800
    total_test = 1024
    
    # Define the generator ratios (must sum to 1.0)
    generator_ratios = {
        ag.generate_binary_format: 0.68,
        ag.generate_original_format: 0.32,  
    }
    
    # Generate the mixed dataset with multiple test sets
    ag.generate_mixed_dataset(
        output_dir=output_dir,
        total_train=total_train,
        total_test=total_test,
        generator_ratios=generator_ratios,
        # Create separate test sets for each format
        test_generators=[ag.generate_original_format, ag.generate_binary_format],
        # Use a fixed random seed for reproducibility
        random_seed=42
    )