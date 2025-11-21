"""
Example DAX measures and their SQL equivalents
Useful for testing and reference
"""

EXAMPLES = {
    "Simple Sum": {
        "dax": "SUM(Sales[Amount])",
        "sql": "SELECT SUM(Amount) FROM Sales",
        "description": "Basic aggregation"
    },
    
    "Count Rows": {
        "dax": "COUNTROWS(Sales)",
        "sql": "SELECT COUNT(*) FROM Sales",
        "description": "Count all rows"
    },
    
    "Distinct Count": {
        "dax": "DISTINCTCOUNT(Sales[CustomerID])",
        "sql": "SELECT COUNT(DISTINCT CustomerID) FROM Sales",
        "description": "Count distinct values"
    },
    
    "Average": {
        "dax": "AVERAGE(Sales[Amount])",
        "sql": "SELECT AVG(Amount) FROM Sales",
        "description": "Calculate average"
    },
    
    "Filtered Sum": {
        "dax": "CALCULATE(SUM(Sales[Amount]), Sales[Year] = 2023)",
        "sql": "SELECT SUM(Amount) FROM Sales WHERE Year = 2023",
        "description": "Sum with filter condition"
    },
    
    "Related Table": {
        "dax": "SUMX(RELATEDTABLE(Sales), Sales[Amount])",
        "sql": """SELECT SUM(s.Amount) 
FROM Customers c
INNER JOIN Sales s ON c.CustomerID = s.CustomerID""",
        "description": "Sum from related table"
    },
    
    "Time Intelligence": {
        "dax": "CALCULATE(SUM(Sales[Amount]), SAMEPERIODLASTYEAR(Sales[Date]))",
        "sql": """SELECT SUM(Amount) 
FROM Sales 
WHERE Date >= DATEADD(YEAR, -1, DATEADD(YEAR, DATEDIFF(YEAR, 0, GETDATE()), 0))
  AND Date < DATEADD(YEAR, -1, DATEADD(YEAR, DATEDIFF(YEAR, 0, GETDATE()) + 1, 0))""",
        "description": "Year-over-year comparison"
    },
    
    "Top N": {
        "dax": "TOPN(10, Sales, Sales[Amount], DESC)",
        "sql": "SELECT TOP 10 * FROM Sales ORDER BY Amount DESC",
        "description": "Top N records"
    },
    
    "Conditional": {
        "dax": "IF(SUM(Sales[Amount]) > 1000, 'High', 'Low')",
        "sql": """SELECT 
    CASE 
        WHEN SUM(Amount) > 1000 THEN 'High' 
        ELSE 'Low' 
    END 
FROM Sales""",
        "description": "Conditional logic"
    },
    
    "Group By": {
        "dax": "SUMX(VALUES(Sales[Category]), CALCULATE(SUM(Sales[Amount])))",
        "sql": "SELECT Category, SUM(Amount) FROM Sales GROUP BY Category",
        "description": "Grouped aggregation"
    }
}


def print_examples():
    """Print all examples"""
    for name, example in EXAMPLES.items():
        print(f"\n{'='*80}")
        print(f"Example: {name}")
        print(f"Description: {example['description']}")
        print(f"\nDAX:")
        print(f"  {example['dax']}")
        print(f"\nSQL:")
        for line in example['sql'].split('\n'):
            print(f"  {line}")
        print("="*80)


if __name__ == "__main__":
    print_examples()

