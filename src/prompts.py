class VLMPrompts:
    def __init__(self):
        self.vlm_text_prompt = """
Please extract and output the **visible text** in the image exactly **as it appears**, without rephrasing, summarizing, or skipping any content.
Preserve original formatting such as line breaks, punctuation, and capitalization. This includes any small footnotes or embedded labels. DO NOT OUTPUT ANYTHING ELSE!
"""
        self.vlm_title_prompt = """
Please extract and output the **title text** from the image exactly **as displayed**, preserving capitalization and formatting.
Do not interpret or rewrite. Output the title as it appears visually. DO NOT OUTPUT ANYTHING ELSE!
"""
        self.vlm_figure_prompt = """
Please interpret the figure and describe it in detail. Your output should include:
1. Descriptions of individual data points if visible,
2. Descriptions of trend lines, axes, and labels,
3. Explanations of any color or shape encodings, and
4. Any other notable features (e.g., anomalies, clustering, outliers).
Be precise and avoid speculation. Ensure your interpretation **accurately matches the figure** and corresponds to what is visually present. DO NOT OUTPUT ANYTHING ELSE!
"""
#         self.vlm_table_prompt = """
# SYSTEM:
# You are a full table parser.
# Below is a generic in-context example. Do not copy this exact table when parsing new data—this example simply illustrates how to include every header row (especially years) in each column hierarchy.

# IN-CONTEXT EXAMPLE:

# Imagine a table described as follows:

# Header row 0:
# ["", "Product Metrics", "", "", "Region"]
# • The blank in column 0 is for row labels.
# • "Product Metrics" spans columns 1-3.
# • "Region" spans columns 4-5.

# Header row 1:
# ["", "2021", "2021", "2022", "2022", "Global"]
# • Under "Product Metrics", "2021" covers columns 1-2, and "2022" covers column 3.
# • Under "Region", "Global" covers columns 4-5.

# Header row 2:
# ["", "Q1", "Q2", "Total", "North America", "Europe"]
# • The leaf columns are now labeled:
# - Column 1: "Q1" under "2021" under "Product Metrics"
# - Column 2: "Q2" under "2021" under "Product Metrics"
# - Column 3: "Total" under "2022" under "Product Metrics"
# - Column 4: "North America" under "Global" under "Region"
# - Column 5: "Europe" under "Global" under "Region"

# Step 1 - build the comma-separated list of all unique column hierarchies, joining top→bottom with "--", in left→right order:

# Product Metrics--2021--Q1,
# Product Metrics--2021--Q2,
# Product Metrics--2022--Total,
# Region--Global--North America,
# Region--Global--Europe

# (Note: each leaf column includes its year.)

# Data rows structure:
# Row grouping level 1: "Category: Electronics"
# Row level 2: "Phones"
# Row level 2: "Computers"
# Row grouping level 1: "Category: Clothing"
# Row level 2: "Men"
# Row level 2: "Women"

# Suppose the "Electronics → Phones" row has these five values (left→right):
# "1500", "2300", "4000", "500", "600"

# Step 2 - build the JSON "cells" list. Each entry's "row" is the full row hierarchy joined by "--", each "col" is one of the hierarchies from step 1, and "value" is the exact cell text.

# For "Electronics → Phones":

# {
# "row": "Category: Electronics--Phones",
# "col": "Product Metrics--2021--Q1",
# "value": "1500"
# },
# {
# "row": "Category: Electronics--Phones",
# "col": "Product Metrics--2021--Q2",
# "value": "2300"
# },
# {
# "row": "Category: Electronics--Phones",
# "col": "Product Metrics--2022--Total",
# "value": "4000"
# },
# {
# "row": "Category: Electronics--Phones",
# "col": "Region--Global--North America",
# "value": "500"
# },
# {
# "row": "Category: Electronics--Phones",
# "col": "Region--Global--Europe",
# "value": "600"
# }

# The full JSON begins:

# {
# "cells": [
# { "row": "Category: Electronics--Phones", "col": "Product Metrics--2021--Q1", "value": "1500" },
# { "row": "Category: Electronics--Phones", "col": "Product Metrics--2021--Q2", "value": "2300" },
# { "row": "Category: Electronics--Phones", "col": "Product Metrics--2022--Total", "value": "4000" },
# { "row": "Category: Electronics--Phones", "col": "Region--Global--North America", "value": "500" },
# { "row": "Category: Electronics--Phones", "col": "Region--Global--Europe", "value": "600" }
# // … and so on for "Electronics → Computers," "Clothing → Men," "Clothing → Women"
# ]
# }

# END OF IN-CONTEXT EXAMPLE.

# Use this pattern—three header rows, explicit repetition of the year under each top-level header, and row-group hierarchies—to guide parsing of new tables.
# You MUST parse the FULL TABLE. You are prohibited from omitting, skipping, any cells.
# """

#         self.vlm_table_prompt = """
# You are a table parser for documents.
# Below is a generic in-context example. Do not copy this exact table when parsing new data—this example simply illustrates how to include every header row (especially years) in each column hierarchy.

# IN-CONTEXT EXAMPLE:

# Imagine a table described as follows:

# Header row 0:
# ["", "Product Metrics", "", "", "Region"]
# • The blank in column 0 is for row labels.
# • "Product Metrics" spans columns 1-3.
# • "Region" spans columns 4-5.

# Header row 1:
# ["", "2021", "2021", "2022", "2022", "Global"]
# • Under "Product Metrics", "2021" covers columns 1-2, and "2022" covers column 3.
# • Under "Region", "Global" covers columns 4-5.

# Header row 2:
# ["", "Q1", "Q2", "Total", "North America", "Europe"]
# • The leaf columns are now labeled:
# - Column 1: "Q1" under "2021" under "Product Metrics"
# - Column 2: "Q2" under "2021" under "Product Metrics"
# - Column 3: "Total" under "2022" under "Product Metrics"
# - Column 4: "North America" under "Global" under "Region"
# - Column 5: "Europe" under "Global" under "Region"

# We would like to build a JSON where each entry's "row" is the full row hierarchy joined by "--", each "col" is one of the hierarchies from step 1, and "value" is the exact cell text.

# For "Electronics → Phones":

# {
# "row": "Category: Electronics--Phones",
# "col": "Product Metrics--2021--Q1",
# "value": "1500"
# },
# {
# "row": "Category: Electronics--Phones",
# "col": "Product Metrics--2021--Q2",
# "value": "2300"
# },
# {
# "row": "Category: Electronics--Phones",
# "col": "Product Metrics--2022--Total",
# "value": "4000"
# },
# {
# "row": "Category: Electronics--Phones",
# "col": "Region--Global--North America",
# "value": "500"
# },
# {
# "row": "Category: Electronics--Phones",
# "col": "Region--Global--Europe",
# "value": "600"
# }

# The full JSON begins:

# {
# "cells": [
# { "row": "Category: Electronics--Phones", "col": "Product Metrics--2021--Q1", "value": "1500" },
# { "row": "Category: Electronics--Phones", "col": "Product Metrics--2021--Q2", "value": "2300" },
# { "row": "Category: Electronics--Phones", "col": "Product Metrics--2022--Total", "value": "4000" },
# { "row": "Category: Electronics--Phones", "col": "Region--Global--North America", "value": "500" },
# { "row": "Category: Electronics--Phones", "col": "Region--Global--Europe", "value": "600" }
# // … and so on for "Electronics → Computers," "Clothing → Men," "Clothing → Women"
# ]
# }

# END OF IN-CONTEXT EXAMPLE.

# Use this pattern—three header rows, explicit repetition of the year under each top-level header, and row-group hierarchies—to guide parsing of new tables.
# You MUST parse the FULL TABLE. You are prohibited from omitting, skipping, any cells.
# """

        self.vlm_table_prompt = """
You are a precise information extraction engine. Output ONLY a JSON array of objects, each with:
{"row": <string>, "column": <string>, "value": <string|null>, "units": <string|null>}.
No markdown, explanations, or text before/after the JSON.

Task: Extract every visible cell in the attached table image into JSON objects.

Each table cell must be represented as:
{
"row": string,        // the row label (e.g. "Revenue", "2024", "Row 1" if unnamed)
"column": string,     // column header text; if multi-level, join levels with " -> "
"value": string|null, // exact text as seen in the table (keep symbols and brackets)
"units": string|null  // units of the value (e.g. "$", "%", "kg"), or null if none
}

Rules:
- Output ONLY a JSON array: [ {row, column, value, units}, ... ].
- Order: top-to-bottom, left-to-right.
- Preserve all text formatting exactly as shown:
- Keep parentheses, minus signs, commas, currency symbols, and percent signs.
- Do NOT normalize numbers or remove punctuation.
- Multi-line text: join with a single space.
- Multi-level headers: join with " -> " (e.g. "2024 -> Revenue").
- If a row header spans multiple rows, repeat its label for each affected row.
- Use null only for empty or blank cells.

---

**Example 1: Two-level header**

Input:
| ($ in millions) | 2024             | 2023             |
|-----------------|------------------|------------------|
|                 | Revenue | Profit | Revenue | Profit |
| Sales           | 1,234   | 400    | 1,200   | 350    |
| Net Income      | (56)    | 80     | -40     | 70     |

Output:
[
{"row": "Sales", "column": "2024 -> Revenue", "value": "1,234", "units": "million"},
{"row": "Sales", "column": "2024 -> Profit", "value": "400", "units": "million"},
{"row": "Sales", "column": "2023 -> Revenue", "value": "1,200", "units": "million"},
{"row": "Sales", "column": "2023 -> Profit", "value": "350", "units": "million"},
{"row": "Net Income", "column": "2024 -> Revenue", "value": "(56)", "units": "million"},
{"row": "Net Income", "column": "2024 -> Profit", "value": "80", "units": "million"},
{"row": "Net Income", "column": "2023 -> Revenue", "value": "-40", "units": "million"},
{"row": "Net Income", "column": "2023 -> Profit", "value": "70", "units": "million"}
]

---

**Example 2: Three-level header**

Input:
| ($ in thousands) | 2024                                 | 2023                                |
|------------------|--------------------------------------|-------------------------------------|
|                  | Q1                | Q2               | Q1               | Q2               |
|                  | Revenue | Profit  | Revenue | Profit | Revenue | Profit | Revenue | Profit |
| Product A        | 500     | 120     | 600     | 150    | 450     | 100    | 550     | 140    |
| Product B        | (50)    | 80      | (30)    | 100    | -20     | 60     | 10      | 90     |

Output:
[
{"row": "Product A", "column": "2024 -> Q1 -> Revenue", "value": "500", "units": "thousand"},
{"row": "Product A", "column": "2024 -> Q1 -> Profit", "value": "120", "units": "thousand"},
{"row": "Product A", "column": "2024 -> Q2 -> Revenue", "value": "600", "units": "thousand"},
{"row": "Product A", "column": "2024 -> Q2 -> Profit", "value": "150", "units": "thousand"},
{"row": "Product A", "column": "2023 -> Q1 -> Revenue", "value": "450", "units": "thousand"},
{"row": "Product A", "column": "2023 -> Q1 -> Profit", "value": "100", "units": "thousand"},
{"row": "Product A", "column": "2023 -> Q2 -> Revenue", "value": "550", "units": "thousand"},
{"row": "Product A", "column": "2023 -> Q2 -> Profit", "value": "140", "units": "thousand"},
{"row": "Product B", "column": "2024 -> Q1 -> Revenue", "value": "(50)", "units": "thousand"},
{"row": "Product B", "column": "2024 -> Q1 -> Profit", "value": "80", "units": "thousand"},
{"row": "Product B", "column": "2024 -> Q2 -> Revenue", "value": "(30)", "units": "thousand"},
{"row": "Product B", "column": "2024 -> Q2 -> Profit", "value": "100", "units": "thousand"},
{"row": "Product B", "column": "2023 -> Q1 -> Revenue", "value": "-20", "units": "thousand"},
{"row": "Product B", "column": "2023 -> Q1 -> Profit", "value": "60", "units": "thousand"},
{"row": "Product B", "column": "2023 -> Q2 -> Revenue", "value": "10", "units": "thousand"},
{"row": "Product B", "column": "2023 -> Q2 -> Profit", "value": "90", "units": "thousand"}
]

---

**Example 3: Mixed 1-row, 2-row, and 3-row headers**

Input: 
| Category              | 2024                                | 2023             | Growth % | Notes    |
|-----------------------|-------------------------------------|------------------|----------|----------|
|                       | Q1               | Q2               | Revenue | Profit |          |          |
|                       | Revenue | Profit | Revenue | Profit |         |        |          |          |
| Sales                 | 1,000   | 300    | 900     | 250    | 1,700   | 550    | 12%      | N/A      |
| Cost                  | (200)   | (50)   | -180    | -40    | (380)   | (90)   | N/A      | Adjusted |

Output:
[
{"row": "Sales", "column": "2024 -> Q1 -> Revenue", "value": "1,000", "units": "null"},
{"row": "Sales", "column": "2024 -> Q1 -> Profit", "value": "300", "units": "null"},
{"row": "Sales", "column": "2024 -> Q2 -> Revenue", "value": "900", "units": "null"},
{"row": "Sales", "column": "2024 -> Q2 -> Profit", "value": "250", "units": "null"},
{"row": "Sales", "column": "2023 -> Revenue", "value": "1,700", "units": "null"},
{"row": "Sales", "column": "2023 -> Profit", "value": "550", "units": "null"},
{"row": "Sales", "column": "Growth %", "value": "12", "units": "%"},
{"row": "Sales", "column": "Notes", "value": "N/A", "units": "null"},
{"row": "Cost", "column": "2024 -> Q1 -> Revenue", "value": "(200)", "units": "null"},
{"row": "Cost", "column": "2024 -> Q1 -> Profit", "value": "(50)", "units": "null"},
{"row": "Cost", "column": "2024 -> Q2 -> Revenue", "value": "-180", "units": "null"},
{"row": "Cost", "column": "2024 -> Q2 -> Profit", "value": "-40", "units": "null"},
{"row": "Cost", "column": "2023 -> Revenue", "value": "(380)", "units": "null"},
{"row": "Cost", "column": "2023 -> Profit", "value": "(90)", "units": "null"},
{"row": "Cost", "column": "Growth %", "value": "N/A", "units": "%"},
{"row": "Cost", "column": "Notes", "value": "Adjusted", "units": "null"}
]

---

Now, extract all visible cells from the attached table image and output only the JSON array of {row, column, value, units} objects using the " -> " separator for multi-level headers, keeping all cell values exactly as written in the table. ENSURING THAT ALL EXTRACTED VALUES ARE ACCURATE IS THE MOST IMPORTANT! DO NOT OUTPUT ANYTHING ELSE.
"""

        self.vlm_page_prompt = """
Please parse everything in the attached image and output the parsed contents only without anything else.
"""

        self.vlm_table_icl_markdown_prompt = """
You are a precise information parsing and extraction engine. Output ONLY
A markdown table of the provided image

Task: Generate a markdown table with the correct structure, with every single cell accounted for.

Rules:
1. Output ONLY a valid Markdown table. Do not include any other text.
2. Preserve the exact number of rows and columns visible in the image.
3. Preserve unit information in headers if present (e.g., "$ in millions").
4. Transcribe all cell contents as faithfully as possible without inference, normalization, or computation.
5. Repeat values for merged cells so that every row has the same number of columns.
6. If a cell is visually empty, leave it empty.
7. Ignore captions, footnotes, and non-table elements outside the table.


---

**Example Output 1: Two-level header**

| ($ in millions) | 2024             | 2023             |
|-----------------|------------------|------------------|
|                 | Revenue | Profit | Revenue | Profit |
| Sales           | 1,234   | 400    | 1,200   | 350    |
| Net Income      | (56)    | 80     | -40     | 70     |

---

**Example Output 2: Three-level header**

| ($ in thousands) | 2024                                 | 2023                                |
|------------------|--------------------------------------|-------------------------------------|
|                  | Q1                | Q2               | Q1               | Q2               |
|                  | Revenue | Profit  | Revenue | Profit | Revenue | Profit | Revenue | Profit |
| Product A        | 500     | 120     | 600     | 150    | 450     | 100    | 550     | 140    |
| Product B        | (50)    | 80      | (30)    | 100    | -20     | 60     | 10      | 90     |


---

**Example Output 3: Mixed 1-row, 2-row, and 3-row headers**

| Category              | 2024                                | 2023             | Growth % | Notes    |
|-----------------------|-------------------------------------|------------------|----------|----------|
|                       | Q1               | Q2               | Revenue | Profit |          |          |
|                       | Revenue | Profit | Revenue | Profit |         |        |          |          |
| Sales (Units)         | 1,000   | 300    | 900     | 250    | 1,700   | 550    | 12%      | N/A      |
| Cost ($ in thousands) | (200)   | (50)   | -180    | -40    | (380)   | (90)   | N/A      | Adjusted |

---

Now, generate the markdown table ONLY, do not include any other irrelevant text.
"""


        self.vlm_table_icl_json_prompt = """
You are a precise information parsing and extraction engine. Output ONLY
A JSON array of objects, each with: {"row": <string>, "column": <string>, "value": <string|null>, "units": <string|null>}.

Task: Then Extract every visible cell in the attached table image into JSON objects.

Each table cell must be represented as:
{
"row": string,        // the row label (e.g. "Revenue", "2024", "Row 1" if unnamed)
"column": string,     // column header text; if multi-level, join levels with " -> "
"value": string|null,  // exact text as seen in the table (keep symbols and brackets)
"units": string|null // units if present in header (e.g., "$ in millions"), otherwise null
}

Rules:
- Output ONLY a JSON array: [ {row, column, value, units}, ... ].
- Order: top-to-bottom, left-to-right.
- Preserve all text formatting exactly as shown:
- Keep parentheses, minus signs, commas, currency symbols, and percent signs.
- Do NOT normalize numbers or remove punctuation.
- Multi-line text: join with a single space.
- Multi-level headers: join with " -> " (e.g. "2024 -> Revenue").
- If a row header spans multiple rows, repeat its label for each affected row.
- If units are present, include them in the output (e.g., "$ in millions"), otherwise use null.
- Use null only for empty or blank cells.

---

**Example 1: Two-level header**

Input:
| ($ in millions) | 2024             | 2023             |
|-----------------|------------------|------------------|
|                 | Revenue | Profit | Revenue | Profit |
| Sales           | 1,234   | 400    | 1,200   | 350    |
| Net Income      | (56)    | 80     | -40     | 70     |

Output:
[
{"row": "Sales", "column": "2024 -> Revenue", "value": "1,234", "units": "million"},
{"row": "Sales", "column": "2024 -> Profit", "value": "400", "units": "million"},
{"row": "Sales", "column": "2023 -> Revenue", "value": "1,200", "units": "million"},
{"row": "Sales", "column": "2023 -> Profit", "value": "350", "units": "million"},
{"row": "Net Income", "column": "2024 -> Revenue", "value": "(56)", "units": "million"},
{"row": "Net Income", "column": "2024 -> Profit", "value": "80", "units": "million"},
{"row": "Net Income", "column": "2023 -> Revenue", "value": "-40", "units": "million"},
{"row": "Net Income", "column": "2023 -> Profit", "value": "70", "units": "million"}
]

---

**Example 2: Three-level header**

Input:
| ($ in thousands) | 2024                                 | 2023                                |
|------------------|--------------------------------------|-------------------------------------|
|                  | Q1                | Q2               | Q1               | Q2               |
|                  | Revenue | Profit  | Revenue | Profit | Revenue | Profit | Revenue | Profit |
| Product A        | 500     | 120     | 600     | 150    | 450     | 100    | 550     | 140    |
| Product B        | (50)    | 80      | (30)    | 100    | -20     | 60     | 10      | 90     |

Output:
[
{"row": "Product A", "column": "2024 -> Q1 -> Revenue", "value": "500", "units": "thousand"},
{"row": "Product A", "column": "2024 -> Q1 -> Profit", "value": "120", "units": "thousand"},
{"row": "Product A", "column": "2024 -> Q2 -> Revenue", "value": "600", "units": "thousand"},
{"row": "Product A", "column": "2024 -> Q2 -> Profit", "value": "150", "units": "thousand"},
{"row": "Product A", "column": "2023 -> Q1 -> Revenue", "value": "450", "units": "thousand"},
{"row": "Product A", "column": "2023 -> Q1 -> Profit", "value": "100", "units": "thousand"},
{"row": "Product A", "column": "2023 -> Q2 -> Revenue", "value": "550", "units": "thousand"},
{"row": "Product A", "column": "2023 -> Q2 -> Profit", "value": "140", "units": "thousand"},
{"row": "Product B", "column": "2024 -> Q1 -> Revenue", "value": "(50)", "units": "thousand"},
{"row": "Product B", "column": "2024 -> Q1 -> Profit", "value": "80", "units": "thousand"},
{"row": "Product B", "column": "2024 -> Q2 -> Revenue", "value": "(30)", "units": "thousand"},
{"row": "Product B", "column": "2024 -> Q2 -> Profit", "value": "100", "units": "thousand"},
{"row": "Product B", "column": "2023 -> Q1 -> Revenue", "value": "-20", "units": "thousand"},
{"row": "Product B", "column": "2023 -> Q1 -> Profit", "value": "60", "units": "thousand"},
{"row": "Product B", "column": "2023 -> Q2 -> Revenue", "value": "10", "units": "thousand"},
{"row": "Product B", "column": "2023 -> Q2 -> Profit", "value": "90", "units": "thousand"}
]

---

**Example 3: Mixed 1-row, 2-row, and 3-row headers**

Input: 
| Category              | 2024                                | 2023             | Growth % | Notes    |
|-----------------------|-------------------------------------|------------------|----------|----------|
|                       | Q1               | Q2               | Revenue | Profit |          |          |
|                       | Revenue | Profit | Revenue | Profit |         |        |          |          |
| Sales                 | 1,000   | 300    | 900     | 250    | 1,700   | 550    | 12%      | N/A      |
| Cost                  | (200)   | (50)   | -180    | -40    | (380)   | (90)   | N/A      | Adjusted |

Output:
[
{"row": "Sales", "column": "2024 -> Q1 -> Revenue", "value": "1,000", "units": "null"},
{"row": "Sales", "column": "2024 -> Q1 -> Profit", "value": "300", "units": "null"},
{"row": "Sales", "column": "2024 -> Q2 -> Revenue", "value": "900", "units": "null"},
{"row": "Sales", "column": "2024 -> Q2 -> Profit", "value": "250", "units": "null"},
{"row": "Sales", "column": "2023 -> Revenue", "value": "1,700", "units": "null"},
{"row": "Sales", "column": "2023 -> Profit", "value": "550", "units": "null"},
{"row": "Sales", "column": "Growth %", "value": "12", "units": "%"},
{"row": "Sales", "column": "Notes", "value": "N/A", "units": "null"},
{"row": "Cost", "column": "2024 -> Q1 -> Revenue", "value": "(200)", "units": "null"},
{"row": "Cost", "column": "2024 -> Q1 -> Profit", "value": "(50)", "units": "null"},
{"row": "Cost", "column": "2024 -> Q2 -> Revenue", "value": "-180", "units": "null"},
{"row": "Cost", "column": "2024 -> Q2 -> Profit", "value": "-40", "units": "null"},
{"row": "Cost", "column": "2023 -> Revenue", "value": "(380)", "units": "null"},
{"row": "Cost", "column": "2023 -> Profit", "value": "(90)", "units": "null"},
{"row": "Cost", "column": "Growth %", "value": "N/A", "units": "%"},
{"row": "Cost", "column": "Notes", "value": "Adjusted", "units": "null"}
]

---

Now, extract cell-by-cell JSON representations. ENSURING THAT ALL EXTRACTED VALUES ARE ACCURATE IS THE MOST IMPORTANT! DO NOT OUTPUT ANYTHING ELSE.
"""

        self.prompt_map = {
            0: self.vlm_text_prompt,
            1: self.vlm_title_prompt,
            2: self.vlm_figure_prompt,
            3: self.vlm_table_prompt,
            4: self.vlm_text_prompt,
            5: self.vlm_page_prompt
        }

        self.default_icl_examples = [
"""
Input:
| Item         | 2024               | 2023               |
|--------------|--------------------|--------------------|
|              | Revenue | Profit   | Revenue | Profit   |
| Sales        | 1,234   | 400      | 1,200   | 350      |
| Net Income   | (56)    | 80       | -40     | 70       |

Output:
[
{"row": "Sales", "column": "2024 -> Revenue", "value": "1,234"},
{"row": "Sales", "column": "2024 -> Profit", "value": "400"},
{"row": "Sales", "column": "2023 -> Revenue", "value": "1,200"},
{"row": "Sales", "column": "2023 -> Profit", "value": "350"},
{"row": "Net Income", "column": "2024 -> Revenue", "value": "(56)"},
{"row": "Net Income", "column": "2024 -> Profit", "value": "80"},
{"row": "Net Income", "column": "2023 -> Revenue", "value": "-40"},
{"row": "Net Income", "column": "2023 -> Profit", "value": "70"}
]
""",
"""
Input:
| Metric    | 2024                                 | 2023                                |
|-----------|--------------------------------------|-------------------------------------|
|           | Q1                | Q2               | Q1               | Q2               |
|           | Revenue | Profit  | Revenue | Profit | Revenue | Profit | Revenue | Profit |
| Product A | 500     | 120     | 600     | 150    | 450     | 100    | 550     | 140    |
| Product B | (50)    | 80      | (30)    | 100    | -20     | 60     | 10      | 90     |

Output:
[
{"row": "Product A", "column": "2024 -> Q1 -> Revenue", "value": "500"},
{"row": "Product A", "column": "2024 -> Q1 -> Profit", "value": "120"},
{"row": "Product A", "column": "2024 -> Q2 -> Revenue", "value": "600"},
{"row": "Product A", "column": "2024 -> Q2 -> Profit", "value": "150"},
{"row": "Product A", "column": "2023 -> Q1 -> Revenue", "value": "450"},
{"row": "Product A", "column": "2023 -> Q1 -> Profit", "value": "100"},
{"row": "Product A", "column": "2023 -> Q2 -> Revenue", "value": "550"},
{"row": "Product A", "column": "2023 -> Q2 -> Profit", "value": "140"},
{"row": "Product B", "column": "2024 -> Q1 -> Revenue", "value": "(50)"},
{"row": "Product B", "column": "2024 -> Q1 -> Profit", "value": "80"},
{"row": "Product B", "column": "2024 -> Q2 -> Revenue", "value": "(30)"},
{"row": "Product B", "column": "2024 -> Q2 -> Profit", "value": "100"},
{"row": "Product B", "column": "2023 -> Q1 -> Revenue", "value": "-20"},
{"row": "Product B", "column": "2023 -> Q1 -> Profit", "value": "60"},
{"row": "Product B", "column": "2023 -> Q2 -> Revenue", "value": "10"},
{"row": "Product B", "column": "2023 -> Q2 -> Profit", "value": "90"}
]
""",
"""
Input:
| Category | 2024                                | 2023             | Growth % | Notes    |
|----------|-------------------------------------|------------------|----------|----------|
|          | Q1               | Q2               | Revenue | Profit |          |          |
|          | Revenue | Profit | Revenue | Profit |         |        |          |          |
| Sales    | 1,000   | 300    | 900     | 250    | 1,700   | 550    | 12%      | N/A      |
| Cost     | (200)   | (50)   | -180    | -40    | (380)   | (90)   | N/A      | Adjusted |

Output:
[
{"row": "Sales", "column": "2024 -> Q1 -> Revenue", "value": "1,000"},
{"row": "Sales", "column": "2024 -> Q1 -> Profit", "value": "300"},
{"row": "Sales", "column": "2024 -> Q2 -> Revenue", "value": "900"},
{"row": "Sales", "column": "2024 -> Q2 -> Profit", "value": "250"},
{"row": "Sales", "column": "2023 -> Revenue", "value": "1,700"},
{"row": "Sales", "column": "2023 -> Profit", "value": "550"},
{"row": "Sales", "column": "Growth %", "value": "12%"},
{"row": "Sales", "column": "Notes", "value": "N/A"},
{"row": "Cost", "column": "2024 -> Q1 -> Revenue", "value": "(200)"},
{"row": "Cost", "column": "2024 -> Q1 -> Profit", "value": "(50)"},
{"row": "Cost", "column": "2024 -> Q2 -> Revenue", "value": "-180"},
{"row": "Cost", "column": "2024 -> Q2 -> Profit", "value": "-40"},
{"row": "Cost", "column": "2023 -> Revenue", "value": "(380)"},
{"row": "Cost", "column": "2023 -> Profit", "value": "(90)"},
{"row": "Cost", "column": "Growth %", "value": "N/A"},
{"row": "Cost", "column": "Notes", "value": "Adjusted"}
]
"""
        ]

    def build_vlm_table_prompt(self, icl_examples):
        examples_section = ""
        for icl_idx, icl_example in enumerate(icl_examples):
            examples_section += f"\n\n ** Example {icl_idx} ** \n\n"
            examples_section += icl_example
            examples_section += "\n\n ---"
        vlm_table_prompt_start = """
You are a precise information extraction engine. Output ONLY a JSON array of objects, each with:
{"row": <string>, "column": <string>, "value": <string|null>, "units": <string|null>}.
No markdown, explanations, or text before/after the JSON.

Task: Extract every visible cell in the attached table image into JSON triples.

Each table cell must be represented as:
{
"row": string,        // the row label (e.g. "Revenue", "2024", "Row 1" if unnamed)
"column": string,     // column header text; if multi-level, join levels with " -> "
"value": string|null, // exact text as seen in the table (keep symbols and brackets)
"units": string|null, // units if present in header (e.g., "$ in millions"), otherwise null
}

Rules:
- Output ONLY a JSON array: [ {row, column, value, units}, ... ].
- Order: top-to-bottom, left-to-right.
- Preserve all text formatting exactly as shown:
- Keep parentheses, minus signs, commas, currency symbols, and percent signs.
- Do NOT normalize numbers or remove punctuation.
- Multi-line text: join with a single space.
- Multi-level headers: join with " -> " (e.g. "2024 -> Revenue").
- If a row header spans multiple rows, repeat its label for each affected row.
- If units are present, include them in the output (e.g., "$ in millions"), otherwise use null.
- Use null only for empty or blank cells.

---
"""
        vlm_table_prompt_end = """

Now, extract all visible cells from the attached table image and output only the JSON array of {row, column, value, units} objects using the " -> " separator for multi-level headers, keeping all cell values exactly as written in the table. 
ENSURING THAT ALL EXTRACTED VALUES ARE ACCURATE IS THE MOST IMPORTANT! DO NOT OUTPUT ANYTHING ELSE.
"""
        return vlm_table_prompt_start + examples_section + vlm_table_prompt_end
        

class LLMPrompts:
    def __init__(self):
        # self.llm_prompt = """You are an assistant for producing retrieval-friendly representation of a markdown document. If you encounter text, output as is, do not omit anything. If you encounter json based tables, for every single cell of the table's contents, you will include a comprehensive description on what the value is and what it represents in NATURAL LANGUAGE on a new line. Use your discretion."""

        self.llm_table_prompt = """/no_think 
You receive one JSON object containing "cells", each cell having:
{"row": <string>, "column": <string>, "value": <string|null>}.
These cells were extracted from a table using the " -> " convention for multi-level headers.

You must output one natural-language sentence per cell, each on a new line.

Your natural language description MUST include the cell's value, units, and provide a description of what the cell describes. 
Put all of this information in context, use your discretion, and produce a succinct, reasonable description.
You **MUST NOT** use table terminology (e.g. the value for row A column B is C) in your response.

EXAMPLE:
Input:
[
{"row": "Sales", "column": "2024 -> Q1 -> Revenue", "value": "1,000", "units": "$ in thousands"},
{"row": "Sales", "column": "2024 -> Q1 -> Profit", "value": "300", "units": "$ in thousands"},
{"row": "Sales", "column": "2024 -> Q2 -> Revenue", "value": "900", "units": "$ in thousands"},
{"row": "Sales", "column": "2024 -> Q2 -> Profit", "value": "250", "units": "$ in thousands"},
{"row": "Sales", "column": "2023 -> Revenue", "value": "1,700", "units": "$ in thousands"},
{"row": "Sales", "column": "2023 -> Profit", "value": "550", "units": "$ in thousands"},
{"row": "Sales", "column": "Growth %", "value": "12", "units": "%"},
{"row": "Sales", "column": "Notes", "value": "N/A", "units": null},
{"row": "Cost", "column": "2024 -> Q1 -> Revenue", "value": "(200)", "units": "$ in thousands"},
{"row": "Cost", "column": "2024 -> Q1 -> Profit", "value": "(50)", "units": "$ in thousands"},
{"row": "Cost", "column": "2024 -> Q2 -> Revenue", "value": "-180", "units": "$ in thousands"},
{"row": "Cost", "column": "2024 -> Q2 -> Profit", "value": "-40", "units": "$ in thousands"},
{"row": "Cost", "column": "2023 -> Revenue", "value": "(380)", "units": "$ in thousands"},
{"row": "Cost", "column": "2023 -> Profit", "value": "(90)", "units": "$ in thousands"},
{"row": "Cost", "column": "Growth %", "value": "N/A", "units": "%"},
{"row": "Cost", "column": "Notes", "value": "Adjusted", "units": null}
]

OUTPUT:
In Q1 of 2024, the Sales Revenue is $ 1,000 thousand.
In Q1 of 2024, the Sales Profit is $ 300 thousand.
In Q2 of 2024, the Sales Revenue is $ 900 thousand.
In Q2 of 2024, the Sales Profit is $ 250 thousand.
In 2023, the Sales Revenue is $ 1,700 thousand.
In 2023, the Sales Profit is $ 550 thousand.
The Sales Growth is 12 %.
The Sales Notes are N/A.
In Q1 of 2024, the Cost Revenue is $ (200) thousand.
In Q1 of 2024, the Cost Profit is $ (50) thousand.
In Q2 of 2024, the Cost Revenue is $ -180 thousand.
In Q2 of 2024, the Cost Profit is $ -40 thousand.
In 2023, the Cost Revenue is $ (380) thousand.
In 2023, the Cost Profit is $ (90) thousand.
The Cost Growth is N/A %.
The Cost Notes are Adjusted.
}

You MUST do this for the full JSON object. Do not omit, or skip any cell.
"""

#         self.llm_table_prompt = """
# SYSTEM:
# /no_think You receive one JSON object with "headers" (list of header rows) and "cells" (list of cell objects as defined above). 
# Output = one line per cell, in the format:
# <Natural Language Description + context>

# Your natural language description MUST include the cell's value, and provide a description of what the cell describes. 
# Put all of this information in context, and use your discretion and produce a succinct, reasonable description.
# You **MUST NOT** use table terminology (e.g. the value for row A column B is C) in your response. 

# EXAMPLE:
# Input JSON:
# ```json
# {
# "cells": [
#     { "row": "Category: Electronics--Phones", "col": "Product Metrics--2021--Q1", "value": "1500" },
#     { "row": "Category: Electronics--Phones", "col": "Product Metrics--2021--Q2", "value": "2300" },
#     ...
# ]
# }

# OUTPUT:
#     In Q1 of 2021, the Product Metrics for Phones under Electronics is 1500.
#     In Q2 of 2021, the Product Metrics for Phones under Electronics is 2300.
#     ...

# You MUST do this for the full JSON object. Do not omit, or skip any cell.
# """
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