"""
Example usage of the Expensive Calculator
"""

from expensive_calculator import ExpensiveCalculator

def example_local():
    """Example using local Ollama model (free but slower)"""
    print("\nExample 1: Using Local Model (Ollama)")
    print("=" * 60)
    
    calc = ExpensiveCalculator(
        model="llama3.2",
        base_url="http://localhost:11434/v1"
    )
    
    # Simple addition with 8 bits (33 API calls)
    result = calc.add(3, 4, bits=8)
    print(f"\n3 + 4 = {result}\n")


def example_openai_cheap():
    """Example using GPT-4o-mini (cheap and fast)"""
    print("\nExample 2: Using GPT-4o-mini (Cheap)")
    print("=" * 60)
    
    calc = ExpensiveCalculator(model="gpt-4o-mini")
    
    # Multiplication with 12 bits (49 API calls)
    result = calc.multiply(6, 7, bits=12)
    print(f"\n6 × 7 = {result}\n")
    
    # Division
    result = calc.divide(20, 4, bits=12)
    print(f"\n20 ÷ 4 = {result}\n")


def example_openai_expensive():
    """Example using GPT-4 (VERY expensive!)"""
    print("\nExample 3: Using GPT-4 (EXPENSIVE!)")
    print("=" * 60)
    print("WARNING: This will cost real money!")
    
    response = input("Continue? (yes/NO): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled. Your wallet is safe.")
        return
    
    calc = ExpensiveCalculator(model="gpt-4")
    
    # Complex calculation with 16 bits (65 API calls)
    result = calc.add(123, 456, bits=16)
    print(f"\n123 + 456 = {result}\n")


def example_comparison():
    """Compare different bit sizes"""
    print("\nExample 4: Bit Size Comparison")
    print("=" * 60)
    
    calc = ExpensiveCalculator(model="gpt-4o-mini")
    
    print("\nAdding 5 + 3 with different bit sizes:\n")
    
    for bits in [4, 8, 16, 32]:
        calc.reset_stats()
        result = calc.add(5, 3, bits=bits)
        print(f"{bits}-bit mode: {calc.api_calls_made} API calls, {calc.total_tokens_used} tokens\n")


def example_stress_test():
    """Stress test - calculate something 'complex'"""
    print("\nExample 5: Stress Test")
    print("=" * 60)
    print("WARNING: This will make many API calls!")
    
    response = input("Continue? (yes/NO): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    calc = ExpensiveCalculator(model="gpt-4o-mini")
    
    # Calculate (10 + 5) × 3
    print("\nComputing: (10 + 5) × 3")
    print("=" * 60)
    
    step1 = calc.add(10, 5, bits=8)
    print(f"Step 1: 10 + 5 = {step1}")
    
    calc.reset_stats()
    step2 = calc.multiply(step1, 3, bits=8)
    print(f"Step 2: {step1} × 3 = {step2}")
    
    print(f"\nFinal answer: (10 + 5) × 3 = {step2}")


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║               EXPENSIVE CALCULATOR EXAMPLES                ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    print("\nChoose an example to run:")
    print("1. Local Model (Ollama) - Free")
    print("2. OpenAI GPT-4o-mini - Cheap")
    print("3. OpenAI GPT-4 - Expensive!")
    print("4. Bit Size Comparison")
    print("5. Stress Test")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-5): ")
    
    examples = {
        '1': example_local,
        '2': example_openai_cheap,
        '3': example_openai_expensive,
        '4': example_comparison,
        '5': example_stress_test,
    }
    
    if choice in examples:
        try:
            examples[choice]()
        except Exception as e:
            print(f"\nError: {e}")
            print("Make sure you have the required API keys and models set up!")
    elif choice == '0':
        print("\nGoodbye! Thanks for not wasting too much money today!")
    else:
        print("\nInvalid choice!")
