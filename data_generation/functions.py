import json
import os
import random

def create_dirs(*dirs):
    """Create directories if they don't exist."""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

# ---- Utility functions ----

def int_to_roman_tokens(num):
    """Convert an integer to a list of Roman numeral tokens."""
    # Map of values to Roman numerals
    val_to_roman = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"),
        (1, "I")
    ]
    
    roman_tokens = []
    i = 0
    
    while num > 0 and i < len(val_to_roman):
        symbol = val_to_roman[i][1]
        value = val_to_roman[i][0]
        
        if num >= value:
            roman_tokens.append(symbol)
            num -= value
        else:
            i += 1
    
    return roman_tokens

def can_represent_in_five_tokens(num):
    """Check if a number can be represented with 5 or fewer Roman numeral tokens."""
    return len(int_to_roman_tokens(num)) <= 5

def has_exactly_five_roman_tokens(num):
    """Check if a number can be represented with exactly 5 Roman numeral tokens."""
    tokens = int_to_roman_tokens(num)
    return len(tokens) == 5

# ---- Formatting functions ----

def format_original(a, b, c):
    """Format addition in original format: 1 2 3 4 5 + 1 2 3 4 5 = 2 4 6 9 0"""
    a_spaced = ' '.join(str(a))
    b_spaced = ' '.join(str(b))
    c_spaced = ' '.join(str(c))
    
    input_text = f"{a_spaced} + {b_spaced}"
    output_text = f"{c_spaced}"
    
    return input_text, output_text

def format_roman(a, b, c):
    """Format addition in Roman numeral format with spaces between tokens"""
    # Convert to Roman tokens
    a_roman = int_to_roman_tokens(a)
    b_roman = int_to_roman_tokens(b)
    c_roman = int_to_roman_tokens(c)
    
    # Join with spaces
    a_spaced = ' '.join(a_roman)
    b_spaced = ' '.join(b_roman)
    c_spaced = ' '.join(c_roman)
    
    input_text = f"{a_spaced} + {b_spaced}"
    output_text = f"{c_spaced}"
    
    return input_text, output_text

def format_sorted_list(unsorted_list, sorted_list):
    """Format sorting task: unsorted list -> sorted list"""
    # Convert numbers to strings and join with spaces
    unsorted_str = ' '.join(str(num) for num in unsorted_list)
    sorted_str = ' '.join(str(num) for num in sorted_list)
    
    input_text = f"S {unsorted_str}"
    output_text = f"{sorted_str}"
    
    return input_text, output_text

# ---- Generator functions ----

def generate_sorting_format(num_examples):
    """Generate examples for sorting lists of random numbers.
       - Lists contain numbers 0-9
       - List length is randomly between 4 and 10
       - Output is the sorted version of the input list
    """
    examples = []
    
    while len(examples) < num_examples:
        # Generate a random list length between 4 and 10
        list_length = random.randint(4, 10)
        
        # Generate a list of random numbers from 0 to 9
        unsorted_list = [random.randint(0, 9) for _ in range(list_length)]
        
        # Sort the list
        sorted_list = sorted(unsorted_list)
        
        # Format the example
        input_text, output_text = format_sorted_list(unsorted_list, sorted_list)
        
        examples.append({
            "input": input_text,
            "output": output_text
        })
    
    return examples

def generate_original_format(num_examples):
    """Generate examples in original format where:
       - Addends can have 1-5 digits each
       - Result must have exactly 5 digits
       - Arithmetic is correct
    """
    examples = []
    
    while len(examples) < num_examples:
        # Generate addends with 1-5 digits each
        a_digits = random.randint(1, 5)
        b_digits = random.randint(1, 5)
        
        # Generate the actual numbers with the specified number of digits
        a_min = 10 ** (a_digits - 1) if a_digits > 1 else 1  # Minimum is 1, not 0
        a_max = (10 ** a_digits) - 1
        a = random.randint(a_min, a_max)
        
        b_min = 10 ** (b_digits - 1) if b_digits > 1 else 1  # Minimum is 1, not 0
        b_max = (10 ** b_digits) - 1
        b = random.randint(b_min, b_max)
        
        # Calculate their sum (ensuring arithmetic correctness)
        c = a + b
        
        # Check if the sum has exactly 5 digits
        if 10000 <= c <= 99999:
            input_text, output_text = format_original(a, b, c)
            
            examples.append({
                "input": input_text,
                "output": output_text
            })
    
    return examples

def generate_roman_format(num_examples):
    """Generate examples in Roman format where:
       - Addends can have 1-5 tokens each
       - Result must have exactly 5 tokens
       - Arithmetic is correct
    """
    examples = []
    
    # Find numbers that can be represented with exactly 5 Roman tokens
    # This improves efficiency by precomputing possible sums
    possible_sums = []
    for i in range(1, 10000):
        if has_exactly_five_roman_tokens(i):
            possible_sums.append(i)
    
    if not possible_sums:
        raise ValueError("Could not find any numbers representable with exactly 5 Roman tokens")
    
    print(f"Found {len(possible_sums)} numbers representable with exactly 5 Roman tokens")
    
    while len(examples) < num_examples:
        # Select a number that can be represented with exactly 5 Roman tokens
        c = random.choice(possible_sums)
        
        # Try to find addends that sum to c
        found_valid_pair = False
        attempts = 0
        max_attempts = 100  # Increased attempts for better success rate
        
        while not found_valid_pair and attempts < max_attempts:
            attempts += 1
            
            # Random split of c into a and b
            a = random.randint(1, c-1)
            b = c - a
            
            # Check if both a and b can be represented with 5 or fewer Roman tokens
            if (can_represent_in_five_tokens(a) and can_represent_in_five_tokens(b)):
                input_text, output_text = format_roman(a, b, c)
                
                examples.append({
                    "input": input_text,
                    "output": output_text
                })
                found_valid_pair = True
        
        # If we couldn't find a valid pair after many attempts, try a different sum
        if not found_valid_pair:
            continue
    
    return examples

# ---- Main data generation function ----

def generate_mixed_dataset(
    output_dir,
    total_train,
    total_test,
    generator_ratios,
    test_generators=None
):
    """
    Generate a mixed dataset with arbitrary generators and ratios.
    
    Args:
        output_dir: Directory to save the dataset
        total_train: Total number of training examples
        total_test: Total number of test examples
        generator_ratios: Dict mapping generator functions to their ratios (must sum to 1.0)
        test_generators: List of generators to use for test sets or a single generator.
                         If None, uses the same ratio mix as training.
    """
    # Validate ratios
    ratio_sum = sum(generator_ratios.values())
    if not 0.999 <= ratio_sum <= 1.001:  # Allow for small floating-point errors
        raise ValueError(f"Generator ratios must sum to 1.0, got {ratio_sum}")
    
    # Create directories
    create_dirs(output_dir)
    
    # Generate training examples
    train_examples = []
    for generator, ratio in generator_ratios.items():
        num_examples = int(total_train * ratio)
        print(f"Generating {num_examples} training examples with {generator.__name__}...")
        examples = generator(num_examples)
        train_examples.extend(examples)
    
    # Shuffle training examples
    random.shuffle(train_examples)
    
    # Write training dataset to file
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_examples, f, indent=2)
    
    # Generate test examples
    if test_generators is None:
        # Use the same ratio mix for test set
        test_examples = []
        for generator, ratio in generator_ratios.items():
            num_examples = int(total_test * ratio)
            print(f"Generating {num_examples} test examples with {generator.__name__}...")
            examples = generator(num_examples)
            test_examples.extend(examples)
        random.shuffle(test_examples)
        
        # Write mixed test dataset to file
        with open(os.path.join(output_dir, "test.json"), "w") as f:
            json.dump(test_examples, f, indent=2)
        
        print(f"Generated dataset: {len(train_examples)} train, {len(test_examples)} test (mixed)")
    else:
        # Handle both single generator and list of generators
        if callable(test_generators):
            test_generators = [test_generators]
        
        # Generate separate test sets for each test generator
        for generator in test_generators:
            generator_name = generator.__name__.replace("generate_", "")
            print(f"Generating {total_test} test examples with {generator.__name__}...")
            test_examples = generator(total_test)
            
            # Write test dataset to file with specific name
            test_filename = f"test_{generator_name}.json"
            with open(os.path.join(output_dir, test_filename), "w") as f:
                json.dump(test_examples, f, indent=2)
            
            print(f"Generated test set: {len(test_examples)} examples in {test_filename}")
    
    print(f"Training set breakdown:")
    for generator, ratio in generator_ratios.items():
        print(f"  - {generator.__name__}: {ratio:.1%}")