# DAX to SQL Converter Bot

An intelligent bot/LLM that converts DAX (Data Analysis Expressions) measures to equivalent SQL queries.

## Features

- 🤖 **LLM-Powered Conversion**: Uses OpenAI's GPT models to easily convert DAX to SQL
- 📝 **Batch Processing**: Convert multiple DAX measures at once
- 🎯 **Context-Aware**: Supports table context and additional metadata
- 💬 **CLI Interface**: Easy-to-use command-line tool
- 📊 **Examples**: Includes common DAX patterns and their SQL equivalents

## Installation

1. Clone or download this repository

2. Install dependencies:
   
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
   
```bash
# Option 1: Environment variable (recommended)
export OPENAI_API_KEY="your-api-key-here"

# Option 2: Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Option 3: Use --api-key flag in CLI
```

## Usage

### Command Line Interface

#### Convert a single DAX measure:

```bash
python cli.py "SUM(Sales[Amount])"
```

#### Convert from a file:

```bash
python cli.py -f measures.dax
```

#### With table context:

```bash
python cli.py "SUM(Sales[Amount])" --table Sales
```

#### Verbose output (with explanation):

```bash
python cli.py "SUM(Sales[Amount])" --verbose
```

#### Save to file:

```bash
python cli.py "SUM(Sales[Amount])" --output result.sql
```

#### JSON output:

```bash
python cli.py "SUM(Sales[Amount])" --json
```

#### Use different model:

```bash
python cli.py "SUM(Sales[Amount])" --model gpt-3.5-turbo
```

### Python API

```python
from dax_to_sql_converter import DAXToSQLConverter

# Initialize converter
converter = DAXToSQLConverter(api_key="your-api-key", model="gpt-4")

# Convert a single measure
result = converter.convert("SUM(Sales[Amount])", table_name="Sales")
print(result.sql_query)
print(result.explanation)

# Convert multiple measures
results = converter.convert_batch([
    "SUM(Sales[Amount])",
    "COUNTROWS(Sales)",
    "AVERAGE(Sales[Price])"
])
```

## Examples

View example DAX measures and their SQL equivalents:

```bash
python examples.py
```

### Common DAX to SQL Conversions

| DAX | SQL Equivalent |
|-----|----------------|
| `SUM(Sales[Amount])` | `SELECT SUM(Amount) FROM Sales` |
| `COUNTROWS(Sales)` | `SELECT COUNT(*) FROM Sales` |
| `DISTINCTCOUNT(Sales[CustomerID])` | `SELECT COUNT(DISTINCT CustomerID) FROM Sales` |
| `CALCULATE(SUM(Sales[Amount]), Sales[Year] = 2023)` | `SELECT SUM(Amount) FROM Sales WHERE Year = 2023` |
| `AVERAGE(Sales[Price])` | `SELECT AVG(Price) FROM Sales` |

## Supported DAX Functions

The converter handles a wide range of DAX functions:

- **Aggregation**: SUM, AVERAGE, COUNT, COUNTROWS, DISTINCTCOUNT, MAX, MIN
- **Filtering**: CALCULATE, FILTER, ALL, ALLEXCEPT
- **Relationships**: RELATED, RELATEDTABLE
- **Time Intelligence**: DATEADD, DATESYTD, SAMEPERIODLASTYEAR, etc.
- **Ranking**: RANKX
- **Top N**: TOPN
- **Conditional**: IF, SWITCH
- **Table Functions**: VALUES, DISTINCT

## Web Interface

A simple web interface is also available:

```bash
python web_app.py
```

Then open http://localhost:5000 in your browser.

## File Structure

```
.
├── dax_to_sql_converter.py  # Main converter module
├── cli.py                    # Command-line interface
├── web_app.py                # Web interface (Flask)
├── examples.py               # Example DAX measures
├── example_usage.py          # Usage examples
├── test_measures.dax         # Sample DAX measures file
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Model Options

- `gpt-4`: Best quality, slower, more expensive
- `gpt-3.5-turbo`: Faster, cheaper, good quality
- `gpt-4-turbo`: Balanced option

## Limitations

- Complex DAX measures with multiple nested functions may require manual review
- Some DAX-specific concepts (like row context) may not translate perfectly
- The converter generates SQL that should be reviewed for optimization
- Requires an OpenAI API key and internet connection

## Contributing

Feel free to submit issues or pull requests to improve the converter!

## License

MIT License - feel free to use this project for your needs.

## Troubleshooting

### "OpenAI API key is required" error
- Make sure you've set the `OPENAI_API_KEY` environment variable
- Or use the `--api-key` flag when running the CLI

### Conversion errors
- Check that your DAX syntax is correct
- Try providing table context with `--table` flag
- For complex measures, try breaking them down into simpler parts

### API rate limits
- If you hit rate limits, wait a moment and try again
- Consider using `gpt-3.5-turbo` for faster/cheaper conversions

