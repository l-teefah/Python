"""
Example usage of the DAX to SQL converter
"""

import os
from dax_to_sql_converter import DAXToSQLConverter

# Example DAX measures
example_measures = [
    "SUM(Sales[Amount])",
    "COUNTROWS(Sales)",
    "DISTINCTCOUNT(Sales[CustomerID])",
    "CALCULATE(SUM(Sales[Amount]), Sales[Year] = 2023)",
    "AVERAGE(Sales[Price])"
]

def main():
    # Check if API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set")
        print("Please set it using: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Initialize converter
    print("Initializing DAX to SQL Converter...")
    converter = DAXToSQLConverter(api_key=api_key, model='gpt-4')
    
    # Convert each example
    print("\n" + "="*80)
    print("Converting Example DAX Measures to SQL")
    print("="*80 + "\n")
    
    for i, dax in enumerate(example_measures, 1):
        print(f"\nExample {i}:")
        print(f"DAX: {dax}")
        print("-" * 80)
        
        try:
            result = converter.convert(dax, table_name="Sales")
            print(f"SQL:\n{result.sql_query}")
            print(f"\nExplanation: {result.explanation}")
            print(f"Confidence: {result.confidence:.2%}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("="*80)

if __name__ == "__main__":
    main()

