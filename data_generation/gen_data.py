import functions as ag
import os
import math
# Example usage
if __name__ == "__main__":
    # Define the output directory

    output_dir = "data/sort_addition"
    
    # Define the total number of examples
    total_train = 12800
    total_test = 1024
    
    # Define the generator ratios (must sum to 1.0)
    generator_ratios = {
        ag.generate_sorting_format: 0.96,
        ag.generate_original_format: 0.04,  
    }
    
    # Generate the mixed dataset with multiple test sets
    ag.generate_mixed_dataset(
        output_dir=output_dir,
        total_train=total_train,
        total_test=total_test,
        generator_ratios=generator_ratios,
        # Create separate test sets for each format
        test_generators=[ag.generate_original_format, ag.generate_sorting_format]
    )

    output_dir = "data/sort_only"
    
    # Define the base number of examples to generate
    base_train = 512
    target_train = 12800  # Target total after oversampling
    
    # Calculate the oversample factor (rounded up to ensure we reach the target)
    oversample_factor = math.ceil(target_train / base_train)
    
    # Define the total number of examples
    total_test = 1024
    
    # Define the generator ratios (must sum to 1.0)
    generator_ratios = {
        ag.generate_sorting_format: 1.00,
        # ag.generate_original_format: 0.00,  
    }
    
    # Generate the mixed dataset with multiple test sets
    ag.generate_mixed_dataset(
        output_dir=output_dir,
        total_train=base_train,  # Generate 614 base examples
        total_test=total_test,
        generator_ratios=generator_ratios,
        # test_generators=[ag.generate_sorting_format],
        oversample_factor=oversample_factor  # Then oversample to reach target count
    )
    
    # Print final counts
    print(f"Base examples: {base_train}")
    print(f"Oversample factor: {oversample_factor}")
    print(f"Expected final count: {base_train * oversample_factor}")
    print(f"Target count: {target_train}")
    
    # Alternative: If you want just one specific test generator
    # ag.generate_mixed_dataset(
    #     output_dir="data/roman_only_test",
    #     total_train=10000,
    #     total_test=1000,
    #     generator_ratios=generator_ratios,
    #     test_generators=ag.generate_roman_format
    # )
    
    # Alternative: If you want a mixed test set with the same ratio as training
    # ag.generate_mixed_dataset(
    #     output_dir="data/mixed_test",
    #     total_train=10000,
    #     total_test=1000,
    #     generator_ratios=generator_ratios,
    #     test_generators=None  # Uses the same ratios as training
    # )