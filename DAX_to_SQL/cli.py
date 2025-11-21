"""
Command-line interface for DAX to SQL converter
"""

import argparse
import sys
import json
from pathlib import Path
from dax_to_sql_converter import DAXToSQLConverter, RuleBasedConverter


def print_result(result, verbose: bool = False):
    """Print conversion result in a formatted way"""
    print("\n" + "="*80)
    print("DAX MEASURE:")
    print("-"*80)
    print(result.dax_measure)
    print("\nSQL QUERY:")
    print("-"*80)
    print(result.sql_query)
    
    if verbose:
        print("\nEXPLANATION:")
        print("-"*80)
        print(result.explanation)
        print(f"\nConfidence: {result.confidence:.2%}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DAX measures to SQL queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single DAX measure
  python cli.py "SUM(Sales[Amount])"
  
  # Convert from file
  python cli.py -f measures.dax
  
  # Convert with table context
  python cli.py "SUM(Sales[Amount])" --table Sales
  
  # Verbose output
  python cli.py "SUM(Sales[Amount])" --verbose
  
  # Save to file
  python cli.py "SUM(Sales[Amount])" --output result.sql
        """
    )
    
    parser.add_argument(
        "dax",
        nargs="?",
        help="DAX measure to convert (or use -f for file input)"
    )
    
    parser.add_argument(
        "-f", "--file",
        help="Input file containing DAX measure(s), one per line"
    )
    
    parser.add_argument(
        "-t", "--table",
        help="Primary table name for context"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file to save SQL query"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed explanation and confidence"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4",
        help="OpenAI model to use (default: gpt-4)"
    )
    
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    # Get DAX measure(s)
    dax_measures = []
    
    if args.file:
        if not Path(args.file).exists():
            print(f"Error: File '{args.file}' not found", file=sys.stderr)
            sys.exit(1)
        with open(args.file, 'r', encoding='utf-8') as f:
            dax_measures = [line.strip() for line in f if line.strip()]
    elif args.dax:
        dax_measures = [args.dax]
    else:
        parser.print_help()
        sys.exit(1)
    
    if not dax_measures:
        print("Error: No DAX measures provided", file=sys.stderr)
        sys.exit(1)
    
    # Initialize converter
    try:
        converter = DAXToSQLConverter(api_key=args.api_key, model=args.model)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nPlease set OPENAI_API_KEY environment variable or use --api-key", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing converter: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Convert
    try:
        results = converter.convert_batch(dax_measures, table_name=args.table)
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Output results
    if args.json:
        output_data = [
            {
                "dax": r.dax_measure,
                "sql": r.sql_query,
                "explanation": r.explanation,
                "confidence": r.confidence
            }
            for r in results
        ]
        print(json.dumps(output_data, indent=2))
    else:
        for result in results:
            print_result(result, verbose=args.verbose)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"-- DAX: {result.dax_measure}\n")
                f.write(f"{result.sql_query}\n\n")
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

