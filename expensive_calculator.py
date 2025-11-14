"""
The Most Expensive Calculator Ever Made™
=========================================
Where every bit costs an API call, and efficiency goes to die.

This calculator represents numbers in binary using individual LLM API calls.
Each bit is determined by asking an LLM a question. Maximum inefficiency achieved.
"""

import os
import time
import re
from typing import Literal
from openai import OpenAI


class ExpensiveCalculator:
    """A calculator so inefficient, it makes O(n!) look good."""
    
    def __init__(self, model: str = "gpt-4o-mini", base_url: str = None, api_key: str = None):
        """
        Initialize the most wasteful calculator ever.
        
        Args:
            model: The model to waste. Default is gpt-4o-mini (cheaper waste)
            base_url: Base URL for API (for local models like Ollama)
            api_key: API key (defaults to OPENAI_API_KEY env var)
        """
        self.model = model
        self.api_calls_made = 0
        self.total_tokens_used = 0
        
        # Setup OpenAI client
        if base_url:
            # For local models (Ollama, etc)
            self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        else:
            # For OpenAI
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        print(f"Expensive Calculator initialized with model: {model}")
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
            # Local model or unknown
            return 0.0
    
    def reset_stats(self):
        """Reset the statistics."""
        self.api_calls_made = 0
        self.total_tokens_used = 0


def main():
    """Interactive REPL so you can ask the expensive calculator to do your operations."""
    banner = (
        "THE MOST EXPENSIVE CALCULATOR EVER MADE\n"
        "Where every bit costs an API call and efficiency is a distant memory.\n"
    )
    print(banner)

    # Choose backend
    backend = input("Choose backend (ollama/openai) [ollama]: ").strip().lower() or "ollama"
    bits = 16

    def make_calc_for_backend(choice: str, model_name: str = None):
        if choice == "ollama":
            model = model_name or "llama3.2"
            return ExpensiveCalculator(model=model, base_url="http://localhost:11434/v1")
        else:
            model = model_name or "gpt-4o-mini"
            return ExpensiveCalculator(model=model)

    try:
        calc = make_calc_for_backend(backend)
    except Exception as e:
        print(f"Failed to initialize chosen backend ({backend}): {e}")
        # Try the other backend as a fallback
        other = "openai" if backend == "ollama" else "ollama"
        try:
            print(f"Trying fallback backend: {other}")
            calc = make_calc_for_backend(other)
            backend = other
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return

    print(f"Using backend: {backend}, model: {calc.model}, bits default: {bits}\n")

    print("Enter operations like: 7 + 5  or  10 * 3")
    print("Commands: bits <n>, model <name>, stats, reset, quit")

    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not cmd:
            continue
        if cmd.lower() in ("quit", "exit"):
            break
        if cmd.lower().startswith("bits "):
            try:
                bits = int(cmd.split()[1])
                print(f"Bits set to {bits}")
            except Exception:
                print("Invalid bits command. Usage: bits 16")
            continue
        if cmd.lower().startswith("model "):
            parts = cmd.split()
            if len(parts) >= 2:
                new_model = parts[1]
                try:
                    # recreate calculator with same backend but new model
                    calc = make_calc_for_backend(backend, model_name=new_model)
                    print(f"Switched to model {new_model}")
                except Exception as e:
                    print(f"Failed to switch model: {e}")
            else:
                print("Usage: model <model-name>")
            continue
        if cmd.lower() == "stats":
            calc._print_stats(0.0)
            continue
        if cmd.lower() == "reset":
            calc.reset_stats()
            print("Stats reset")
            continue

        # Try parsing a simple binary operation like: int op int
        m = re.match(r"^\s*([-+]?[0-9]+)\s*([+\-*/xX])\s*([-+]?[0-9]+)\s*$", cmd)
        if m:
            a = int(m.group(1))
            op = m.group(2)
            b = int(m.group(3))

            try:
                if op == "+":
                    res = calc.add(a, b, bits=bits)
                elif op == "-":
                    res = calc.subtract(a, b, bits=bits)
                elif op in ("*", "x", "X"):
                    res = calc.multiply(a, b, bits=bits)
                elif op == "/":
                    res = calc.divide(a, b, bits=bits)
                else:
                    print("Unsupported operator")
                    continue

                print(f"Result: {res}\n")
            except Exception as e:
                print(f"Error performing operation: {e}")
            continue

        print("Unrecognized command. Try: 7 + 5  or  bits 16  or model gpt-4o-mini")

    print("Goodbye.")


if __name__ == "__main__":
    main()
