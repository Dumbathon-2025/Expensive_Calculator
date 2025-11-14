"""
The ULTRA Expensive Calculator™ - Maximum Waste Edition
========================================================
Where EVERYTHING costs an API call, including the prompts themselves.

This version takes the original expensive calculator and makes it EVEN MORE WASTEFUL:
- Numbers are encoded as bits via API calls (like before)
- BUT NOW: The instruction prompts themselves are also encoded character-by-character as bits!
- Each character in the prompt → 8 bits → 8 API calls
- A 100-character prompt = 800 API calls just to send the instruction!

This is the pinnacle of inefficiency. The ultimate waste.
"""

import os
import time
import re
from pathlib import Path
from openai import OpenAI


class UltraExpensiveCalculator:
    """A calculator so wasteful, even the instructions cost hundreds of API calls."""
    
    def __init__(self, model: str = "gpt-4o-mini", base_url: str = None, api_key: str = None):
        """
        Initialize the ULTRA wasteful calculator.
        
        Args:
            model: The model to waste
            base_url: Base URL for API (for local models like Ollama)
            api_key: API key (defaults to OPENAI_API_KEY env var or .env file)
        """
        self.model = model
        self.api_calls_made = 0
        self.total_tokens_used = 0
        self.is_local = base_url is not None
        
        # Setup OpenAI client
        if base_url:
            # For local models (Ollama, etc)
            self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        else:
            # For OpenAI - try to load from .env
            if not api_key:
                env_path = Path(__file__).parent / ".env"
                if env_path.exists():
                    with open(env_path) as f:
                        for line in f:
                            if line.startswith("OPENAI_API_KEY="):
                                api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                                break
                if not api_key:
                    api_key = os.getenv("OPENAI_API_KEY")
            
            self.client = OpenAI(api_key=api_key)
        
        print(f"ULTRA Expensive Calculator initialized with model: {model}")
        print(f"WARNING: This version encodes EVERYTHING as bits!")
        print(f"Even the prompts cost hundreds of API calls!\n")
    
    def _waste_api_call_for_bit(self, bit_value: str, context: str = "bit") -> str:
        """
        The most basic waste unit: Ask the LLM to confirm a single bit value.
        We tell it what the bit should be, and ask it to repeat it back.
        Maximum waste achieved.
        """
        prompt = f"You are encoding a {context}. The bit value is: {bit_value}. Respond with ONLY that single character: {bit_value}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a bit encoder. Respond with only the bit value given."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5,
                temperature=0
            )
            
            self.api_calls_made += 1
            self.total_tokens_used += response.usage.total_tokens
            
            # Extract the bit value
            bit = response.choices[0].message.content.strip()
            bit = '1' if '1' in bit else '0'
            
            return bit
            
        except Exception as e:
            print(f"Error: {e}")
            return bit_value
    
    def _char_to_wasteful_bits(self, char: str) -> str:
        """
        Convert a single character to binary using 8 API calls.
        This is how we encode the instruction prompts!
        """
        ascii_val = ord(char)
        binary = format(ascii_val, '08b')
        
        # Waste an API call for each bit!
        wasteful_binary = ""
        for bit in binary:
            confirmed_bit = self._waste_api_call_for_bit(bit, f"character '{char}'")
            wasteful_binary += confirmed_bit
        
        return wasteful_binary
    
    def _string_to_ultra_wasteful_bits(self, text: str, show_progress: bool = True) -> str:
        """
        Convert an entire string to binary using API calls for each bit.
        This is ULTRA wasteful: len(text) * 8 API calls!
        """
        if show_progress:
            print(f"Encoding '{text[:50]}{'...' if len(text) > 50 else ''}' as bits...")
            print(f"This will cost {len(text) * 8} API calls!")
        
        full_binary = ""
        for i, char in enumerate(text):
            if show_progress and i % 10 == 0:
                print(f"  Character {i}/{len(text)}: '{char}' -> encoding...")
            
            char_binary = self._char_to_wasteful_bits(char)
            full_binary += char_binary
        
        if show_progress:
            print(f"Done encoding! Total bits: {len(full_binary)}\n")
        
        return full_binary
    
    def _wasteful_bits_to_string(self, binary: str) -> str:
        """
        Convert binary back to string.
        Each 8 bits = 1 character.
        """
        text = ""
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            if len(byte) == 8:
                ascii_val = int(byte, 2)
                text += chr(ascii_val)
        return text
    
    def _number_to_expensive_binary(self, num: int, bits: int = 16) -> str:
        """
        Convert a number to binary using API calls for each bit.
        This is the same as the original expensive calculator.
        """
        if num < 0:
            raise ValueError("This calculator is too dumb for negative numbers")
        
        print(f"Converting number {num} to binary using {bits} API calls...")
        
        binary = ""
        for bit_pos in range(bits):
            # Calculate what this bit should be
            bit_value = '1' if (num >> bit_pos) & 1 else '0'
            
            # Waste an API call to confirm it
            confirmed_bit = self._waste_api_call_for_bit(bit_value, f"number {num} bit {bit_pos}")
            binary = confirmed_bit + binary  # Prepend to build MSB first
            
            if bit_pos % 4 == 0:
                print(f"  Bit {bit_pos}: {confirmed_bit}")
        
        print(f"Result: {binary}\n")
        return binary
    
    def _ultra_wasteful_operation(self, a: int, b: int, operation: str, bits: int = 16) -> int:
        """
        Perform an operation with MAXIMUM waste:
        1. Encode the numbers as bits (2 * bits API calls)
        2. Encode the ENTIRE INSTRUCTION PROMPT as bits (hundreds of API calls!)
        3. Decode the instruction back
        4. Actually do the operation (1 API call)
        5. Encode the result as bits (bits API calls)
        
        Total: Could be 1000+ API calls for a simple addition!
        """
        print(f"\n{'='*70}")
        print(f"ULTRA EXPENSIVE {operation.upper()}: {a} {operation} {b}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Step 1: Encode the numbers
        print("STEP 1: Encoding first number as bits...")
        binary_a = self._number_to_expensive_binary(a, bits)
        
        print("STEP 2: Encoding second number as bits...")
        binary_b = self._number_to_expensive_binary(b, bits)
        
        # Step 2: Create and encode the instruction prompt itself!
        op_symbol = {'+': 'add', '-': 'subtract', '*': 'multiply', '/': 'divide'}.get(operation, operation)
        instruction = f"Calculate {a} {operation} {b} and respond with only the number"
        
        print(f"STEP 3: Encoding the instruction prompt as bits...")
        print(f"Instruction: '{instruction}'")
        encoded_instruction = self._string_to_ultra_wasteful_bits(instruction, show_progress=True)
        
        # Step 3: Decode the instruction (just to prove we can)
        print("STEP 4: Decoding the instruction from bits...")
        decoded_instruction = self._wasteful_bits_to_string(encoded_instruction)
        print(f"Decoded: '{decoded_instruction}'\n")
        
        # Step 4: Actually do the operation
        print(f"STEP 5: Finally doing the actual calculation...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a calculator. Respond with only numbers."},
                    {"role": "user", "content": decoded_instruction}
                ],
                max_tokens=50,
                temperature=0
            )
            
            self.api_calls_made += 1
            self.total_tokens_used += response.usage.total_tokens
            
            result_str = response.choices[0].message.content.strip()
            result = int(''.join(c for c in result_str if c.isdigit() or c == '-'))
            print(f"Result: {result}\n")
            
        except Exception as e:
            print(f"Error: {e}")
            # Fallback
            if operation == '+':
                result = a + b
            elif operation == '-':
                result = a - b
            elif operation == '*':
                result = a * b
            elif operation == '/':
                result = a // b
        
        # Step 5: Encode the result as bits
        print(f"STEP 6: Encoding result {result} as bits...")
        binary_result = self._number_to_expensive_binary(result, bits)
        
        elapsed = time.time() - start_time
        
        self._print_ultra_stats(elapsed)
        return result
    
    def add(self, a: int, b: int, bits: int = 16) -> int:
        """ULTRA wasteful addition."""
        return self._ultra_wasteful_operation(a, b, '+', bits)
    
    def subtract(self, a: int, b: int, bits: int = 16) -> int:
        """ULTRA wasteful subtraction."""
        return self._ultra_wasteful_operation(a, b, '-', bits)
    
    def multiply(self, a: int, b: int, bits: int = 16) -> int:
        """ULTRA wasteful multiplication."""
        return self._ultra_wasteful_operation(a, b, '*', bits)
    
    def divide(self, a: int, b: int, bits: int = 16) -> int:
        """ULTRA wasteful division."""
        return self._ultra_wasteful_operation(a, b, '/', bits)
    
    def _print_ultra_stats(self, elapsed_time: float):
        """Print the ULTRA damage report."""
        print(f"\n{'='*70}")
        print(f"ULTRA DAMAGE REPORT")
        print(f"{'='*70}")
        print(f"Time WASTED: {elapsed_time:.2f} seconds")
        print(f"API calls made: {self.api_calls_made}")
        print(f"Total tokens used: {self.total_tokens_used}")
        if not self.is_local:
            print(f"Estimated cost: ${self._estimate_cost():.6f}")
        else:
            print(f"Cost: $0.00 (FREE with local model)")
        print(f"Efficiency: BEYOND TERRIBLE")
        print(f"Waste level: MAXIMUM")
        print(f"{'='*70}\n")
    
    def _estimate_cost(self) -> float:
        """Estimate the cost based on the model."""
        if 'gpt-4o-mini' in self.model.lower():
            return (self.total_tokens_used / 1_000_000) * 0.375
        elif 'gpt-4o' in self.model.lower():
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
    """Interactive REPL for ULTRA wasteful calculations."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║          THE ULTRA EXPENSIVE CALCULATOR™                         ║
    ║              MAXIMUM WASTE EDITION                               ║
    ║                                                                  ║
    ║  Where even the instruction prompts are encoded as bits          ║
    ║  One character = 8 API calls                                     ║
    ║  One operation = 500-1000+ API calls                             ║
    ║                                                                  ║
    ║  This is the most wasteful calculator ever conceived.            ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Choose backend
    backend = input("Choose backend (ollama/openai) [ollama]: ").strip().lower() or "ollama"
    bits = 8  # Default to fewer bits because this is SO expensive
    
    def make_calc(choice: str, model_name: str = None):
        if choice == "ollama":
            model = model_name or "llama3.2"
            return UltraExpensiveCalculator(model=model, base_url="http://localhost:11434/v1")
        else:
            model = model_name or "gpt-4o-mini"
            return UltraExpensiveCalculator(model=model)
    
    try:
        calc = make_calc(backend)
    except Exception as e:
        print(f"Failed to initialize {backend}: {e}")
        other = "openai" if backend == "ollama" else "ollama"
        try:
            print(f"Trying {other}...")
            calc = make_calc(other)
            backend = other
        except Exception as e2:
            print(f"Both backends failed: {e2}")
            return
    
    print(f"\nUsing: {backend}, model: {calc.model}")
    print(f"Default bits: {bits} (fewer than normal because this is ULTRA expensive)\n")
    
    print("WARNING: Each operation will use 500-1000+ API calls!")
    print("The instruction prompts themselves are encoded as bits!\n")
    
    print("Enter operations like: 7 + 5")
    print("Commands: bits <n>, model <name>, stats, reset, quit\n")
    
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
                print("Invalid. Usage: bits 8")
            continue
        if cmd.lower().startswith("model "):
            parts = cmd.split()
            if len(parts) >= 2:
                try:
                    calc = make_calc(backend, model_name=parts[1])
                    print(f"Switched to {parts[1]}")
                except Exception as e:
                    print(f"Failed: {e}")
            continue
        if cmd.lower() == "stats":
            calc._print_ultra_stats(0.0)
            continue
        if cmd.lower() == "reset":
            calc.reset_stats()
            print("Stats reset")
            continue
        
        # Parse operation
        m = re.match(r"^\s*([-+]?[0-9]+)\s*([+\-*/])\s*([-+]?[0-9]+)\s*$", cmd)
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            
            print(f"\nPrepare yourself... this will use ~{bits*3 + len(f'Calculate {a} {op} {b} and respond with only the number')*8 + 1} API calls!")
            confirm = input("Continue? (yes/NO): ").strip().lower()
            if confirm not in ('yes', 'y'):
                print("Cancelled.\n")
                continue
            
            try:
                if op == '+':
                    res = calc.add(a, b, bits)
                elif op == '-':
                    res = calc.subtract(a, b, bits)
                elif op == '*':
                    res = calc.multiply(a, b, bits)
                elif op == '/':
                    res = calc.divide(a, b, bits)
                
                print(f"\nFINAL RESULT: {a} {op} {b} = {res}\n")
            except Exception as e:
                print(f"Error: {e}")
            continue
        
        print("Unrecognized. Try: 7 + 5")
    
    print("\nGoodbye. Thanks for wasting resources to the absolute maximum!")


if __name__ == "__main__":
    main()
