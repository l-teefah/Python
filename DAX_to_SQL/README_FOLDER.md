# DAX_to_SQL

## Overview
This folder contains a comprehensive DAX-to-SQL conversion tool that translates DAX (Data Analysis Expressions) queries from Microsoft Power BI into equivalent SQL queries. The project includes both a command-line interface and a web application for easy conversion and testing.

## Contents

### Core Files
- **dax_to_sql_converter.py** - Main conversion engine that parses DAX and generates SQL
- **cli.py** - Command-line interface for batch processing and scripting
- **web_app.py** - Flask/web application for interactive conversion
- **example_usage.py** - Code examples demonstrating programmatic usage

### Documentation & Configuration
- **README.md** - Main project documentation
- **requirements.txt** - Python dependencies
- **instructions.txt** - Quick setup instructions
- **.gitignore** - Git ignore configuration
- **test_measures.dax** - Sample DAX expressions for testing

### Subdirectories
- **Documentation/** - Additional documentation and guides
  - QUICKSTART.md - Quick start guide for new users

## Purpose
This tool helps developers and analysts:
- Convert DAX expressions to SQL queries
- Port Power BI logic to SQL databases
- Automate DAX-to-SQL migration processes
- Validate conversion accuracy
- Generate equivalent SQL for data warehouses

## Technology Stack
- **Language**: Python
- **Framework**: Flask (for web app)
- **Parsing**: DAX parsing and SQL generation
- **Interfaces**: CLI and web-based UI

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
```bash
pip install -r requirements.txt
```

## Getting Started

### Command-Line Interface (CLI)
```bash
python cli.py --input "measure.dax" --output "query.sql"
```

### Web Application
```bash
python web_app.py
# Access at http://localhost:5000
```

### Programmatic Usage
```python
from dax_to_sql_converter import convert_dax_to_sql

dax_query = "EVALUATE SUMMARIZE(...)"
sql_query = convert_dax_to_sql(dax_query)
print(sql_query)
```

## Key Features
- **DAX Parsing** - Accurate parsing of DAX syntax
- **SQL Generation** - Generates optimized SQL queries
- **Multiple Interfaces** - CLI, web app, and Python API
- **Error Handling** - Clear error messages and validation
- **Testing Tools** - Sample DAX expressions included
- **Documentation** - Comprehensive guides and examples

## Project Structure
```
DAX_to_SQL/
├── dax_to_sql_converter.py   # Main converter
├── cli.py                     # Command-line interface
├── web_app.py                # Web application
├── example_usage.py          # Usage examples
├── requirements.txt          # Dependencies
├── Documentation/            # Additional docs
│   └── QUICKSTART.md        # Quick start guide
└── test_measures.dax        # Test data
```

## Documentation
For detailed information, see:
- **README.md** - Full project documentation
- **Documentation/QUICKSTART.md** - Quick start guide
- **example_usage.py** - Code examples

## Support
For issues and questions, refer to the documentation folder or test files for working examples.

---
*Generated on 2026-05-08*
