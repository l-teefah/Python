# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Set Up API Key

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**Windows CMD:**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 3. Try It Out

### Command Line (Simple)
```bash
python cli.py "SUM(Sales[Amount])"
```

### Command Line (From File)
```bash
python cli.py -f test_measures.dax
```

### Command Line (Verbose)
```bash
python cli.py "SUM(Sales[Amount])" --verbose
```

### Web Interface
```bash
python web_app.py
```
Then open http://localhost:5000

### Python Script
```bash
python example_usage.py
```

## Example Output

```
================================================================================
DAX MEASURE:
--------------------------------------------------------------------------------
SUM(Sales[Amount])

SQL QUERY:
--------------------------------------------------------------------------------
SELECT SUM(Amount) FROM Sales
================================================================================
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [examples.py](examples.py) for more DAX patterns
- Customize the converter in `dax_to_sql_converter.py`

