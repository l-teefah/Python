"""
DAX to SQL Converter Bot
Converts DAX measures to SQL equivalent queries using LLM
"""

import os
import json
from typing import Optional, Dict, Any
from openai import OpenAI
from dataclasses import dataclass


@dataclass
class ConversionResult:
    """Result of DAX to SQL conversion"""
    dax_measure: str
    sql_query: str
    explanation: str
    confidence: float


class DAXToSQLConverter:
    """Converts DAX measures to SQL using LLM"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the converter
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use for conversion
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        # System prompt with DAX to SQL conversion guidelines
        self.system_prompt = """You are an expert in both DAX (Data Analysis Expressions) and SQL. 
Your task is to convert DAX measures to equivalent SQL queries.

Key conversion rules:
1. DAX functions map to SQL as follows:
   - SUM() -> SUM()
   - AVERAGE() -> AVG()
   - COUNT() -> COUNT()
   - COUNTROWS() -> COUNT(*)
   - DISTINCTCOUNT() -> COUNT(DISTINCT ...)
   - MAX() -> MAX()
   - MIN() -> MIN()
   - CALCULATE() -> WHERE clauses or subqueries
   - FILTER() -> WHERE clauses
   - RELATED() -> JOINs
   - RELATEDTABLE() -> JOINs
   - ALL() -> Remove filters or use subqueries
   - ALLEXCEPT() -> GROUP BY with specific columns
   - VALUES() -> DISTINCT
   - EARLIER() -> Correlated subqueries or window functions
   - RANKX() -> RANK() or ROW_NUMBER()
   - TOPN() -> ORDER BY ... LIMIT
   - IF() -> CASE WHEN
   - SWITCH() -> CASE WHEN with multiple conditions

2. Table references in DAX become FROM clauses in SQL
3. Column references use table.column format
4. Filter context in DAX becomes WHERE clauses in SQL
5. Row context in DAX becomes GROUP BY or window functions in SQL
6. Time intelligence functions (DATEADD, DATESYTD, etc.) map to SQL date functions

Always provide:
- A complete, executable SQL query
- Proper JOINs when multiple tables are referenced
- Appropriate WHERE clauses for filters
- GROUP BY when aggregations are used
- Comments explaining the conversion logic

Format your response as JSON with:
{
  "sql": "the SQL query",
  "explanation": "brief explanation of the conversion",
  "confidence": 0.0-1.0
}"""

    def convert(self, dax_measure: str, table_name: Optional[str] = None, 
                context: Optional[Dict[str, Any]] = None) -> ConversionResult:
        """
        Convert a DAX measure to SQL
        
        Args:
            dax_measure: The DAX measure to convert
            table_name: Optional primary table name for context
            context: Optional additional context (table schema, relationships, etc.)
        
        Returns:
            ConversionResult with SQL query and metadata
        """
        user_prompt = f"Convert this DAX measure to SQL:\n\n{dax_measure}"
        
        if table_name:
            user_prompt += f"\n\nPrimary table: {table_name}"
        
        if context:
            user_prompt += f"\n\nAdditional context: {json.dumps(context, indent=2)}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return ConversionResult(
                dax_measure=dax_measure,
                sql_query=result.get("sql", ""),
                explanation=result.get("explanation", ""),
                confidence=float(result.get("confidence", 0.5))
            )
            
        except Exception as e:
            raise Exception(f"Conversion failed: {str(e)}")
    
    def convert_batch(self, dax_measures: list[str], 
                     table_name: Optional[str] = None) -> list[ConversionResult]:
        """Convert multiple DAX measures to SQL"""
        results = []
        for measure in dax_measures:
            try:
                result = self.convert(measure, table_name)
                results.append(result)
            except Exception as e:
                results.append(ConversionResult(
                    dax_measure=measure,
                    sql_query="",
                    explanation=f"Error: {str(e)}",
                    confidence=0.0
                ))
        return results


class RuleBasedConverter:
    """Fallback rule-based converter for common DAX patterns"""
    
    @staticmethod
    def convert_simple_sum(dax: str) -> Optional[str]:
        """Convert simple SUM measures"""
        import re
        # Pattern: SUM('Table'[Column])
        pattern = r"SUM\s*\(\s*['\"]?(\w+)['\"]?\s*\[\s*(\w+)\s*\]\s*\)"
        match = re.search(pattern, dax, re.IGNORECASE)
        if match:
            table, column = match.groups()
            return f"SELECT SUM({column}) FROM {table}"
        return None
    
    @staticmethod
    def convert_simple_count(dax: str) -> Optional[str]:
        """Convert simple COUNT measures"""
        import re
        # Pattern: COUNT('Table'[Column]) or COUNTROWS('Table')
        if "COUNTROWS" in dax.upper():
            pattern = r"COUNTROWS\s*\(\s*['\"]?(\w+)['\"]?\s*\)"
            match = re.search(pattern, dax, re.IGNORECASE)
            if match:
                table = match.group(1)
                return f"SELECT COUNT(*) FROM {table}"
        return None

