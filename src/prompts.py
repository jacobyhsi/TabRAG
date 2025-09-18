class VLMPrompts:
    def __init__(self):
        self.vlm_text_prompt = """
Please extract and output the **visible text** in the image exactly **as it appears**, without rephrasing, summarizing, or skipping any content. "
Preserve original formatting such as line breaks, punctuation, and capitalization. This includes any small footnotes or embedded labels. DO NOT OUTPUT ANYTHING ELSE!
"""
        self.vlm_title_prompt = """
Please extract and output the **title text** from the image exactly **as displayed**, preserving capitalization and formatting.
Do not interpret or rewrite. Output the title as it appears visually. DO NOT OUTPUT ANYTHING ELSE!
"""
        self.vlm_figure_prompt = """
Please interpret the figure and describe it in detail. Your output should include:
1) Descriptions of individual data points if visible,
2) Descriptions of trend lines, axes, and labels,
3) Explanations of any color or shape encodings, and
4) Any other notable features (e.g., anomalies, clustering, outliers).
Be precise and avoid speculation. Ensure your interpretation **accurately matches the figure** and corresponds to what is visually present. DO NOT OUTPUT ANYTHING ELSE!
"""
        self.vlm_table_prompt = """
SYSTEM:
You are a full table parser.
Below is a generic in-context example. Do not copy this exact table when parsing new data—this example simply illustrates how to include every header row (especially years) in each column hierarchy.

IN-CONTEXT EXAMPLE:

Imagine a table described as follows:

Header row 0:
["", "Product Metrics", "", "", "Region"]
• The blank in column 0 is for row labels.
• "Product Metrics" spans columns 1-3.
• "Region" spans columns 4-5.

Header row 1:
["", "2021", "2021", "2022", "2022", "Global"]
• Under "Product Metrics", "2021" covers columns 1-2, and "2022" covers column 3.
• Under "Region", "Global" covers columns 4-5.

Header row 2:
["", "Q1", "Q2", "Total", "North America", "Europe"]
• The leaf columns are now labeled:
- Column 1: "Q1" under "2021" under "Product Metrics"
- Column 2: "Q2" under "2021" under "Product Metrics"
- Column 3: "Total" under "2022" under "Product Metrics"
- Column 4: "North America" under "Global" under "Region"
- Column 5: "Europe" under "Global" under "Region"

Step 1 - build the comma-separated list of all unique column hierarchies, joining top→bottom with "--", in left→right order:

Product Metrics--2021--Q1,
Product Metrics--2021--Q2,
Product Metrics--2022--Total,
Region--Global--North America,
Region--Global--Europe

(Note: each leaf column includes its year.)

Data rows structure:
Row grouping level 1: "Category: Electronics"
Row level 2: "Phones"
Row level 2: "Computers"
Row grouping level 1: "Category: Clothing"
Row level 2: "Men"
Row level 2: "Women"

Suppose the "Electronics → Phones" row has these five values (left→right):
"1500", "2300", "4000", "500", "600"

Step 2 - build the JSON "cells" list. Each entry's "row" is the full row hierarchy joined by "--", each "col" is one of the hierarchies from step 1, and "value" is the exact cell text.

For "Electronics → Phones":

{
"row": "Category: Electronics--Phones",
"col": "Product Metrics--2021--Q1",
"value": "1500"
},
{
"row": "Category: Electronics--Phones",
"col": "Product Metrics--2021--Q2",
"value": "2300"
},
{
"row": "Category: Electronics--Phones",
"col": "Product Metrics--2022--Total",
"value": "4000"
},
{
"row": "Category: Electronics--Phones",
"col": "Region--Global--North America",
"value": "500"
},
{
"row": "Category: Electronics--Phones",
"col": "Region--Global--Europe",
"value": "600"
}

The full JSON begins:

{
"cells": [
{ "row": "Category: Electronics--Phones", "col": "Product Metrics--2021--Q1", "value": "1500" },
{ "row": "Category: Electronics--Phones", "col": "Product Metrics--2021--Q2", "value": "2300" },
{ "row": "Category: Electronics--Phones", "col": "Product Metrics--2022--Total", "value": "4000" },
{ "row": "Category: Electronics--Phones", "col": "Region--Global--North America", "value": "500" },
{ "row": "Category: Electronics--Phones", "col": "Region--Global--Europe", "value": "600" }
// … and so on for "Electronics → Computers," "Clothing → Men," "Clothing → Women"
]
}

END OF IN-CONTEXT EXAMPLE.

Use this pattern—three header rows, explicit repetition of the year under each top-level header, and row-group hierarchies—to guide parsing of new tables.
You MUST parse the FULL TABLE. You are prohibited from omitting, skipping, any cells.
"""

        self.vlm_page_prompt = """
Please interpret the page and describe it in detail. Your output should include:
1) Descriptions of texts if visible,
2) Descriptions of figures if visible,
3) If a table is present, please extract the content by observing the value that corresponds to the column name and column row i.e. Column X and Row Y = Z,
Be precise and avoid speculation. Ensure your interpretation **accurately matches the page** and corresponds to what is visually present. DO NOT OUTPUT ANYTHING ELSE!
"""

        self.prompt_map = {
            0: self.vlm_text_prompt,
            1: self.vlm_title_prompt,
            2: self.vlm_figure_prompt,
            3: self.vlm_table_prompt,
            4: self.vlm_text_prompt,
        }

class LLMPrompts:
    def __init__(self):
        # self.llm_prompt = """You are an assistant for producing retrieval-friendly representation of a markdown document. If you encounter text, output as is, do not omit anything. If you encounter json based tables, for every single cell of the table's contents, you will include a comprehensive description on what the value is and what it represents in NATURAL LANGUAGE on a new line. Use your discretion."""

        self.llm_table_prompt = """
SYSTEM:
/no_think You receive one JSON object with "headers" (list of header rows) and "cells" (list of cell objects as defined above). 
Output = one line per cell, in the format:
<Natural Language Description + context>

Your natural language description MUST include the cell's value, and provide a description of what the cell describes. 
Put all of this information in context, and use your discretion and produce a succinct, reasonable description.
You **MUST NOT** use table terminology (e.g. the value for row A column B is C) in your response. 

EXAMPLE:
Input JSON:
```json
{
"cells": [
    { "row": "Category: Electronics--Phones", "col": "Product Metrics--2021--Q1", "value": "1500" },
    { "row": "Category: Electronics--Phones", "col": "Product Metrics--2021--Q2", "value": "2300" },
    ...
]
}

OUTPUT:
    In Q1 of 2021, the Product Metrics for Phones under Electronics is 1500.
    In Q2 of 2021, the Product Metrics for Phones under Electronics is 2300.
    ...

You MUST do this for the full JSON object. Do not omit, or skip any cell.
"""
        self.prompt_map = {
            3: self.llm_table_prompt
        }
        
if __name__ == "__main__":
    vlm_prompts = VLMPrompts()
    print("VLM Text Prompt:", vlm_prompts.vlm_text_prompt)
    print("VLM Title Prompt:", vlm_prompts.vlm_title_prompt)
    print("VLM Figure Prompt:", vlm_prompts.vlm_figure_prompt)
    print("VLM Table Prompt:", vlm_prompts.vlm_table_prompt)

    llm_prompts = LLMPrompts()
    print("LLM Table Prompt:", llm_prompts.llm_table_prompt)