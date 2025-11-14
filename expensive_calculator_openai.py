"""
The Most Expensive Calculator Ever Made™ - OpenAI Version
==========================================================
Where every bit costs an API call, and efficiency goes to die.
Uses OpenAI models (fast but EXPENSIVE!).

This calculator represents numbers in binary using individual LLM API calls.
Each bit is determined by asking an LLM a question. Maximum inefficiency achieved.
"""

import os
import sys
import time
from pathlib import Path
from typing import Literal
from openai import OpenAI


class ExpensiveCalculator:
    """A calculator so inefficient, it makes O(n!) look good."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the most wasteful calculator ever with OpenAI.
        
        Args:
            model: The OpenAI model to use. Default is gpt-4o-mini (cheapest)
                   Options: gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-4
        """
        self.model = model
        self.api_calls_made = 0
        self.total_tokens_used = 0
        
        # Load API key from .env file in same directory
        env_path = Path(__file__).parent / ".env"
        api_key = None
        
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("OPENAI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found!\n"
                f"Please add it to: {env_path}\n"
                "Format: OPENAI_API_KEY=sk-your-key-here"
            )
        
        # Setup OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        print(f"Expensive Calculator initialized with OpenAI model: {model}")
        print(f"Prepare your wallet...\n")
    
    def _waste_api_call_for_bit(self, bit_position: int, number: int, total_bits: int) -> str:
        """
        Waste an entire API call just to determine a single bit.
        This is the heart of the inefficiency.
        """
        # Ask the LLM to determine if this bit is 0 or 1
        # We make it do actual work to make it even more wasteful
        prompt = f"""You are a bit in a binary number representation system.
        
Current bit position: {bit_position} (where 0 is the least significant bit)
The number we're representing: {number}
Total bits we're using: {total_bits}

Calculate what bit value (0 or 1) should be at position {bit_position} for the number {number}.

Think through this step by step:
1. Convert {number} to binary
2. Find the bit at position {bit_position}
3. Respond with ONLY the single character: either "0" or "1"

Your response (0 or 1):"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a single bit in a binary number. Respond with only '0' or '1'."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0
            )
            
            self.api_calls_made += 1
            self.total_tokens_used += response.usage.total_tokens
            
            # Extract the bit value
            bit = response.choices[0].message.content.strip()
            # Clean up response to get just 0 or 1
            bit = '1' if '1' in bit else '0'
            
            return bit
            
        except Exception as e:
            print(f"Error wasting API call: {e}")
            return "0"
    
    def _number_to_expensive_binary(self, num: int, bits: int = 32) -> str:
        """
        Convert a number to binary using the most expensive method possible:
        One API call per bit!
        """
        if num < 0:
            raise ValueError("This calculator is too dumb for negative numbers")
        
        print(f"Converting {num} to binary using {bits} wasteful API calls...")
        
        binary = ""
        for bit_pos in range(bits):
            bit = self._waste_api_call_for_bit(bit_pos, num, bits)
            binary = bit + binary  # Prepend to build MSB first
            print(f"  Bit {bit_pos}: {bit} (API call #{self.api_calls_made})")
        
        print(f"Result: {binary}\n")
        return binary
    
    def _expensive_binary_to_number(self, binary: str) -> int:
        """
        Convert binary back to number.
        We could waste API calls here too, but let's show some mercy... 
        Just kidding! Let's waste more calls!
        """
        print(f"Converting binary {binary} back to decimal using MORE API calls...")
        
        result = 0
        for i, bit in enumerate(reversed(binary)):
            if bit == '1':
                # Waste an API call to calculate 2^i
                power = self._waste_api_call_for_power(i)
                result += power
                print(f"  Position {i}: bit={bit}, 2^{i}={power}")
        
        print(f"Result: {result}\n")
        return result
    
    def _waste_api_call_for_power(self, exponent: int) -> int:
        """Waste an API call to calculate 2^exponent. Why? Because we can!"""
        prompt = f"Calculate 2 raised to the power of {exponent}. Respond with ONLY the number, nothing else."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a calculator. Respond with only numbers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0
            )
            
            self.api_calls_made += 1
            self.total_tokens_used += response.usage.total_tokens
            
            result = int(response.choices[0].message.content.strip())
            return result
            
        except Exception as e:
            print(f"Error: {e}")
            return 2 ** exponent  # Fallback
    
    def _waste_api_call_for_operation(self, a: int, b: int, operation: str) -> int:
        """
        Waste an API call to perform a simple arithmetic operation.
        Because using Python's built-in operators would be too efficient.
        """
        prompt = f"Calculate: {a} {operation} {b}\nRespond with ONLY the numerical result, nothing else."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a calculator. Respond with only numbers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0
            )
            
            self.api_calls_made += 1
            self.total_tokens_used += response.usage.total_tokens
            
            result = response.choices[0].message.content.strip()
            # Handle potential decimal points or extra text
            result = ''.join(c for c in result if c.isdigit() or c == '-')
            return int(result)
            
        except Exception as e:
            print(f"Error: {e}")
            # Fallback to actually computing it (how embarrassing)
            if operation == '+':
                return a + b
            elif operation == '-':
                return a - b
            elif operation == '*':
                return a * b
            elif operation == '/':
                return a // b
    
    def add(self, a: int, b: int, bits: int = 16) -> int:
        """
        Add two numbers using the most expensive method possible.
        
        Process:
        1. Convert 'a' to binary (bits API calls)
        2. Convert 'b' to binary (bits API calls)
        3. Use an API call to add them
        4. Convert result to binary (bits API calls)
        5. Convert binary back to decimal (bits API calls)
        
        Total: 4*bits + 1 API calls for a simple addition!
        """
        print(f"\n{'='*60}")
        print(f"EXPENSIVE ADDITION: {a} + {b}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Step 1 & 2: Convert to binary (wastefully)
        binary_a = self._number_to_expensive_binary(a, bits)
        binary_b = self._number_to_expensive_binary(b, bits)
        
        # Step 3: Waste an API call to do the addition
        print(f"Using API call to add {a} + {b}...")
        result = self._waste_api_call_for_operation(a, b, '+')
        print(f"Addition result: {result}\n")
        
        # Step 4 & 5: Convert result to binary and back (maximum waste!)
        binary_result = self._number_to_expensive_binary(result, bits)
        final_result = self._expensive_binary_to_number(binary_result)
        
        elapsed = time.time() - start_time
        
        self._print_stats(elapsed)
        return final_result
    
    def subtract(self, a: int, b: int, bits: int = 16) -> int:
        """Subtraction with maximum inefficiency."""
        print(f"\n{'='*60}")
        print(f"EXPENSIVE SUBTRACTION: {a} - {b}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        binary_a = self._number_to_expensive_binary(a, bits)
        binary_b = self._number_to_expensive_binary(b, bits)
        
        print(f"Using API call to subtract {a} - {b}...")
        result = self._waste_api_call_for_operation(a, b, '-')
        print(f"Subtraction result: {result}\n")
        
        binary_result = self._number_to_expensive_binary(result, bits)
        final_result = self._expensive_binary_to_number(binary_result)
        
        elapsed = time.time() - start_time
        
        self._print_stats(elapsed)
        return final_result
    
    def multiply(self, a: int, b: int, bits: int = 16) -> int:
        """Multiplication with maximum inefficiency."""
        print(f"\n{'='*60}")
        print(f"EXPENSIVE MULTIPLICATION: {a} × {b}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        binary_a = self._number_to_expensive_binary(a, bits)
        binary_b = self._number_to_expensive_binary(b, bits)
        
        print(f"Using API call to multiply {a} × {b}...")
        result = self._waste_api_call_for_operation(a, b, '*')
        print(f"Multiplication result: {result}\n")
        
        binary_result = self._number_to_expensive_binary(result, bits)
        final_result = self._expensive_binary_to_number(binary_result)
        
        elapsed = time.time() - start_time
        
        self._print_stats(elapsed)
        return final_result
    
    def divide(self, a: int, b: int, bits: int = 16) -> int:
        """Division with maximum inefficiency."""
        print(f"\n{'='*60}")
        print(f"EXPENSIVE DIVISION: {a} ÷ {b}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        binary_a = self._number_to_expensive_binary(a, bits)
        binary_b = self._number_to_expensive_binary(b, bits)
        
        print(f"Using API call to divide {a} ÷ {b}...")
        result = self._waste_api_call_for_operation(a, b, '/')
        print(f"Division result: {result}\n")
        
        binary_result = self._number_to_expensive_binary(result, bits)
        final_result = self._expensive_binary_to_number(binary_result)
        
        elapsed = time.time() - start_time
        
        self._print_stats(elapsed)
        return final_result
    
    def _print_stats(self, elapsed_time: float):
        """Print the damage report."""
        print(f"\n{'='*60}")
        print(f"DAMAGE REPORT")
        print(f"{'='*60}")
        print(f"Time wasted: {elapsed_time:.2f} seconds")
        print(f"API calls made: {self.api_calls_made}")
        print(f"Total tokens used: {self.total_tokens_used}")
        print(f"Estimated cost: ${self._estimate_cost():.6f}")
        print(f"Efficiency: ABSOLUTELY TERRIBLE")
        print(f"{'='*60}\n")
    
    def _estimate_cost(self) -> float:
        """Estimate the cost based on the model."""
        if 'gpt-4o-mini' in self.model.lower():
            # GPT-4o-mini pricing: $0.150 per 1M input tokens, $0.600 per 1M output tokens
            # Rough estimate assuming 50/50 split
            return (self.total_tokens_used / 1_000_000) * 0.375
        elif 'gpt-4o' in self.model.lower():
            # GPT-4o pricing: $2.50 per 1M input tokens, $10.00 per 1M output tokens
            return (self.total_tokens_used / 1_000_000) * 6.25
        elif 'gpt-4-turbo' in self.model.lower():
            return (self.total_tokens_used / 1_000_000) * 15.00
        elif 'gpt-4' in self.model.lower():
            return (self.total_tokens_used / 1_000_000) * 45.00
        else:
            return 0.0
    
    def reset_stats(self):
        """Reset the statistics."""
        self.api_calls_made = 0
        self.total_tokens_used = 0


def main():
    """Demo the calculator in all its wasteful glory."""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║        THE MOST EXPENSIVE CALCULATOR EVER MADE™            ║
    ║                    OPENAI VERSION ($$$$)                   ║
    ║                                                            ║
    ║  Where every bit costs an API call                         ║
    ║  and efficiency is a distant memory                        ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    print("\nUsing OpenAI models - LET'S BURN SOME MONEY!")
    print("=" * 60)
    
    try:
        # Start with cheaper model
        print("\nPhase 1: Using gpt-4o-mini (relatively cheap)")
        print("-" * 60)
        calc_cheap = ExpensiveCalculator(model="gpt-4o-mini")
        result = calc_cheap.multiply(3, 4, bits=12)
        print(f"Final answer: 3 × 4 = {result}")
        
        # Ask if they want to go REALLY expensive
        print("\n" + "="*60)
        print("WARNING: The following will use GPT-4 and cost REAL MONEY!")
        print("="*60)
        response = input("\nSwitch to GPT-4 for MAXIMUM EXPENSE? (yes/NO): ")
        
        if response.lower() in ['yes', 'y']:
            print("\nPhase 2: ENGAGING MAXIMUM WASTE MODE!\n")
            calc_expensive = ExpensiveCalculator(model="gpt-4")
            result = calc_expensive.add(42, 69, bits=16)
            print(f"Final answer: 42 + 69 = {result}")
            print("\nYour wallet will remember this day...")
        else:
            print("\nWise choice. Your wallet is safe... for now.")
        
    except ValueError as e:
        print(f"\nError: {e}")
        return
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return
    
    print("\nDemo complete! Thank you for wasting money with us!")


if __name__ == "__main__":
    main()
