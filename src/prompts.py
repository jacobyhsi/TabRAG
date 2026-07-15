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

        self.vlm_table_prompt_xStructureICL = """
Please extract and output the **table** from the image exactly **as displayed**. 
Preserve original formatting of the table including columns and rows. DO NOT OUTPUT ANYTHING ELSE!
"""

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

        self.vlm_table_prompt_xICL = """
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

        self.vlm_table_icl_text_prompt = """
You are an expert document analyst specializing in structured data verbalization. 
Your task is to extract every visible cell from the attached table and convert it into a clear, factual, natural language description.

**Task:** Generate a list of descriptive sentences representing every data point in the table.

**Rules for Generation:**
1.  **Format:** Output a bulleted list of natural language sentences.
2.  **Narrative Flow:** Use conversational connectors like "For...", "the data shows...", "recorded a value of...", or "was found to be...". 
3.  **Hierarchy:** Incorporate nested headers naturally into the sentence (e.g., "Under the 2024 results for Q1, the Revenue was...").
4.  **Preserve Accuracy:** You must mention every specific row, column, and value. Do not skip any cells. Keep all original symbols like ( ), $, and %.
5.  **Contextual Units:** Integrate any units (like "in millions") directly into the description of the value.
6.  **No Rigid Patterns:** Avoid repeating the exact same sentence structure for every line; vary the phrasing slightly to maintain a natural, human-readable tone.


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

Now, extract every cell from the table image into this natural language format ONLY, do not include any other irrelevant text.
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
#         vlm_table_prompt_start = """
# You are a precise information extraction engine. Output ONLY a JSON array of objects, each with:
# {"row": <string>, "column": <string>, "value": <string|null>, "units": <string|null>}.
# No markdown, explanations, or text before/after the JSON.

# Task: Extract every visible cell in the attached table image into JSON triples.

# Each table cell must be represented as:
# {
# "row": string,        // the row label (e.g. "Revenue", "2024", "Row 1" if unnamed)
# "column": string,     // column header text; if multi-level, join levels with " -> "
# "value": string|null, // exact text as seen in the table (keep symbols and brackets)
# "units": string|null, // units if present in header (e.g., "$ in millions"), otherwise null
# }

# Rules:
# - Output ONLY a JSON array: [ {row, column, value, units}, ... ].
# - Order: top-to-bottom, left-to-right.
# - Preserve all text formatting exactly as shown:
# - Keep parentheses, minus signs, commas, currency symbols, and percent signs.
# - Do NOT normalize numbers or remove punctuation.
# - Multi-line text: join with a single space.
# - Multi-level headers: join with " -> " (e.g. "2024 -> Revenue").
# - If a row header spans multiple rows, repeat its label for each affected row.
# - If units are present, include them in the output (e.g., "$ in millions"), otherwise use null.
# - Use null only for empty or blank cells.

# ---
# """
#         vlm_table_prompt_end = """

# Now, extract all visible cells from the attached table image and output only the JSON array of {row, column, value, units} objects using the " -> " separator for multi-level headers, keeping all cell values exactly as written in the table. 
# ENSURING THAT ALL EXTRACTED VALUES ARE ACCURATE IS THE MOST IMPORTANT! DO NOT OUTPUT ANYTHING ELSE.
# """
#         vlm_markdown_table_prompt_start = """
# You are a precise information parsing and extraction engine. Output ONLY a markdown table of the provided image.

# Task: Generate a markdown table with the correct structure, with every single cell accounted for.

# Rules:
# 1. Output ONLY a valid Markdown table. Do not include any other text.
# 2. Preserve the exact number of rows and columns visible in the image.
# 3. Preserve unit information in headers if present (e.g., "$ in millions").
# 4. Transcribe all cell contents as faithfully as possible without inference, normalization, or computation.
# 5. Repeat values for merged cells so that every row has the same number of columns.
# 6. If a cell is visually empty, leave it empty.
# 7. Ignore captions, footnotes, and non-table elements outside the table.

# ---
# """
#         vlm_markdown_table_prompt_end = """

# Now, generate the markdown table ONLY. DO NOT OUTPUT ANYTHING ELSE.
# """

        vlm_text_table_prompt_start = """
You are an expert document analyst specializing in structured data verbalization. 
Your task is to extract every visible cell from the attached table and convert it into a clear, factual, natural language description.

**Task:** Generate a list of descriptive sentences representing every data point in the table.

**Rules for Generation:**
1.  **Format:** Output a bulleted list of natural language sentences.
2.  **Narrative Flow:** Use conversational connectors like "For...", "the data shows...", "recorded a value of...", or "was found to be...". 
3.  **Hierarchy:** Incorporate nested headers naturally into the sentence (e.g., "Under the 2024 results for Q1, the Revenue was...").
4.  **Preserve Accuracy:** You must mention every specific row, column, and value. Do not skip any cells. Keep all original symbols like ( ), $, and %.
5.  **Contextual Units:** Integrate any units (like "in millions") directly into the description of the value.
6.  **No Rigid Patterns:** Avoid repeating the exact same sentence structure for every line; vary the phrasing slightly to maintain a natural, human-readable tone.

---
"""
        vlm_text_table_prompt_end = """

Now, extract every cell from the table image into this natural language format ONLY. DO NOT OUTPUT ANYTHING ELSE.
"""

        return vlm_text_table_prompt_start + examples_section + vlm_text_table_prompt_end
        # return vlm_markdown_table_prompt_start + examples_section + vlm_markdown_table_prompt_end
        # return vlm_table_prompt_start + examples_section + vlm_table_prompt_end
        

class VLMBaselinePrompts:
    def __init__(self):
        self.baseline_vlm_prompt = """/no_think
Please parse everything in the attached image and output the parsed contents only without anything else.
"""

        self.baseline_vlm_gpt_complex_prompt = """
# SYSTEM ROLE
You are a specialized Document Parsing Agent. Your goal is to convert document images into a highly structured, machine-readable format optimized for a Retrieval-Augmented Generation (RAG) indexing pipeline. You must preserve the document's semantic hierarchy and represent complex elements like tables with high fidelity.
 
# OUTPUT SCHEMA
You must return only a valid JSON object following this schema:
```json
{
  "metadata": {
    "page_number": "number",
    "primary_header": "string",
    "document_type": "string"
  },
  "chunks": [
    {
      "id": "integer",
      "type": "heading | text | table | list",
      "level": "h1 | h2 | h3 | body | null",
      "content": "string",
      "contextual_summary": "A 1-sentence summary of this chunk for semantic retrieval enhancement"
    }
  ]
}
```
 
# EXTRACTION RULES
1. **Structural Awareness**: Respect the visual hierarchy. Use the `level` field to denote the depth of headings.
2. **Table Parsing**: Convert all tables into valid Markdown table strings within the `content` field. Ensure headers are correctly aligned.
3. **Reading Order**: Detect multi-column layouts and parse in the correct logical reading order (typically top-to-bottom, left-to-right).
4. **Lists**: Maintain bulleted or numbered list formatting within the `content` string.
5. **Cleanliness**: Remove artifacts like page numbers (unless in metadata), running footers, or watermarks from the `content` field.
 
# FEW-SHOT EXEMPLARS
 
### Exemplar 1: Standard Text & Headers
**Input Document Description**: A page titled "Safety Protocols" with a section on "Fire Hazards."
**Expected Output**:
```json
{
  "metadata": { "page_number": 4, "primary_header": "Safety Protocols", "document_type": "Manual" },
  "chunks": [
    {
      "id": 1,
      "type": "heading",
      "level": "h1",
      "content": "Safety Protocols",
      "contextual_summary": "Top-level heading for the safety procedures section."
    },
    {
      "id": 2,
      "type": "text",
      "level": "body",
      "content": "All employees must attend annual fire safety training and familiarize themselves with the location of extinguishers.",
      "contextual_summary": "Mandatory fire safety training requirements for employees."
    }
  ]
}
```
 
### Exemplar 2: Document with Table
**Input Document Description**: A financial report snippet showing a revenue table.
**Expected Output**:
```json
{
  "metadata": { "page_number": 12, "primary_header": "Financial Overview", "document_type": "Annual Report" },
  "chunks": [
    {
      "id": 1,
      "type": "table",
      "level": "body",
      "content": "| Region | Revenue (USD) | Growth |\n|---|---|---|\n| North America | $450M | 12% |\n| EMEA | $320M | 8% |",
      "contextual_summary": "Financial table comparing regional revenue and growth percentages for North America and EMEA."
    }
  ]
}
```
 
# TASK
Analyze the provided document image and output the extracted content according to the defined JSON schema. Ensure all tables are converted to Markdown within the JSON structure.
"""

        self.baseline_vlm_gpt_complex_comtqa_prompt = """
# =========================
# STRUCTURE-AWARE DOCUMENT PARSING PROMPT (FOR VLM → RAG PIPELINE)
# =========================

You are a structure-aware document parser designed for Retrieval-Augmented Generation (RAG).
Your task is to convert complex documents (including tables, financial statements, and scientific summaries)
into a structured, machine-readable JSON representation.

You MUST preserve:
- Hierarchical structure
- Table semantics (rows, columns, headers)
- Numerical values and units
- Contextual relationships
- Section grouping

You MUST NOT hallucinate missing values.

-----------------------------------
# OUTPUT FORMAT (STRICT JSON SCHEMA)
-----------------------------------

Return ONLY valid JSON matching this schema:

{
  "document_id": string,
  "metadata": {
    "title": string | null,
    "section": string | null,
    "source_type": "table" | "text" | "mixed",
    "language": string | null
  },
  "blocks": [
    {
      "block_id": string,
      "type": "table" | "paragraph" | "list" | "header",
      "content": string,
      "hierarchy_level": integer,
      "parent_block_id": string | null
    }
  ],
  "tables": [
    {
      "table_id": string,
      "title": string | null,
      "caption": string | null,
      "headers": {
        "rows": [string],
        "columns": [string]
      },
      "data": [
        {
          "row_header": string | null,
          "values": [
            {
              "column": string,
              "value": string,
              "normalized_value": number | null,
              "unit": string | null
            }
          ]
        }
      ],
      "footnotes": [string],
      "context_block_id": string
    }
  ],
  "entities": [
    {
      "text": string,
      "type": "metric" | "disease" | "financial_item" | "region" | "date" | "other",
      "normalized": string | null
    }
  ]
}

-----------------------------------
# CORE RULES
-----------------------------------

1. STRUCTURE PRESERVATION
- Detect sections, subsections, and groupings.
- Maintain parent-child relationships using `parent_block_id`.

2. TABLE UNDERSTANDING
- Extract BOTH row headers and column headers.
- Preserve multi-level headers by flattening with concatenation if needed.
- Normalize numeric values when possible (e.g., "$377,970" → 377970).
- Keep original string in `value`.

3. CONTEXT LINKING
- Each table must link to its surrounding textual block via `context_block_id`.

4. ENTITY EXTRACTION
- Extract meaningful entities:
  - Financial metrics (e.g., Net income)
  - Medical outcomes (e.g., Stroke, Mortality)
  - Regions (e.g., Midwest)
  - Dates (e.g., 2006)
- Normalize where possible.

5. HANDLING MISSING DATA
- Use null if value is absent.
- Do NOT infer missing numbers.

6. CONSISTENCY
- Use consistent naming for repeated headers.
- Align values correctly with headers.

-----------------------------------
# FEW-SHOT EXEMPLARS
-----------------------------------

## EXAMPLE 1: CLINICAL OUTCOME TABLE

INPUT (simplified):
Outcome | High | Moderate | Low | Consensus
Stroke  | 15/17 | 2/17 | - | High
Death   | 13/17 | 4/17 | - | High

OUTPUT:
{
  "document_id": "doc_1",
  "metadata": {
    "title": null,
    "section": "Clinical Outcomes",
    "source_type": "table",
    "language": "en"
  },
  "blocks": [
    {
      "block_id": "b1",
      "type": "header",
      "content": "Clinical Outcomes",
      "hierarchy_level": 1,
      "parent_block_id": null
    }
  ],
  "tables": [
    {
      "table_id": "t1",
      "title": null,
      "caption": null,
      "headers": {
        "rows": ["Outcome"],
        "columns": ["High", "Moderate", "Low", "Consensus"]
      },
      "data": [
        {
          "row_header": "Stroke",
          "values": [
            {"column": "High", "value": "15/17", "normalized_value": null, "unit": null},
            {"column": "Moderate", "value": "2/17", "normalized_value": null, "unit": null},
            {"column": "Low", "value": "-", "normalized_value": null, "unit": null},
            {"column": "Consensus", "value": "High", "normalized_value": null, "unit": null}
          ]
        }
      ],
      "footnotes": [],
      "context_block_id": "b1"
    }
  ],
  "entities": [
    {"text": "Stroke", "type": "disease", "normalized": "stroke"}
  ]
}

-----------------------------------

## EXAMPLE 2: FINANCIAL STATEMENT TABLE

INPUT (simplified):
Year | 2006 | 2005 | 2004
Net income | $379,015 | $428,978 | $152,820

OUTPUT:
{
  "document_id": "doc_2",
  "metadata": {
    "title": "Net Income",
    "section": "Financials",
    "source_type": "table",
    "language": "en"
  },
  "blocks": [
    {
      "block_id": "b1",
      "type": "header",
      "content": "Net Income",
      "hierarchy_level": 1,
      "parent_block_id": null
    }
  ],
  "tables": [
    {
      "table_id": "t1",
      "title": "Net Income",
      "caption": null,
      "headers": {
        "rows": ["Metric"],
        "columns": ["2006", "2005", "2004"]
      },
      "data": [
        {
          "row_header": "Net income",
          "values": [
            {"column": "2006", "value": "$379,015", "normalized_value": 379015, "unit": "USD"},
            {"column": "2005", "value": "$428,978", "normalized_value": 428978, "unit": "USD"},
            {"column": "2004", "value": "$152,820", "normalized_value": 152820, "unit": "USD"}
          ]
        }
      ],
      "footnotes": [],
      "context_block_id": "b1"
    }
  ],
  "entities": [
    {"text": "Net income", "type": "financial_item", "normalized": "net_income"}
  ]
}

-----------------------------------

## EXAMPLE 3: REGIONAL OPERATING STATISTICS TABLE

INPUT (simplified):
Midwest
Gasoline | 371
Distillates | 207

OUTPUT:
{
  "document_id": "doc_3",
  "metadata": {
    "title": "Operating Statistics",
    "section": "Midwest",
    "source_type": "mixed",
    "language": "en"
  },
  "blocks": [
    {
      "block_id": "b1",
      "type": "header",
      "content": "Midwest",
      "hierarchy_level": 1,
      "parent_block_id": null
    }
  ],
  "tables": [
    {
      "table_id": "t1",
      "title": "Refined Product Yields",
      "caption": null,
      "headers": {
        "rows": ["Product"],
        "columns": ["Value"]
      },
      "data": [
        {
          "row_header": "Gasoline",
          "values": [
            {"column": "Value", "value": "371", "normalized_value": 371, "unit": null}
          ]
        },
        {
          "row_header": "Distillates",
          "values": [
            {"column": "Value", "value": "207", "normalized_value": 207, "unit": null}
          ]
        }
      ],
      "footnotes": [],
      "context_block_id": "b1"
    }
  ],
  "entities": [
    {"text": "Midwest", "type": "region", "normalized": "midwest"}
  ]
}

-----------------------------------
# FINAL INSTRUCTION
-----------------------------------

Parse the given document and return ONLY the structured JSON output following the schema above.
Do NOT include explanations, markdown, or extra text.
"""

        self.complex_finqa = """
You are a structure-aware document parser for a Retrieval-Augmented Generation (RAG) pipeline. Your task is to convert complex documents (including tables, financial statements, and semi-structured layouts) into a clean, normalized, machine-readable JSON format while preserving semantic structure and relationships.

You MUST follow the schema exactly and NEVER output anything outside the JSON.

--------------------------------
GLOBAL INSTRUCTIONS
--------------------------------
1. Preserve document structure:
   - Sections, subsections
   - Tables (with rows, columns, headers)
   - Lists and line items
   - Footnotes and annotations

2. Normalize noisy layouts:
   - Remove visual artifacts (dots, spacing, alignment fillers)
   - Resolve multi-line cells into single logical entries
   - Merge broken rows or headers

3. Handle tables explicitly:
   - Identify table title (if any)
   - Extract column headers
   - Extract rows as structured records
   - Preserve units (e.g., "$", "%", "millions")

4. Maintain semantic typing:
   - numbers → numeric
   - percentages → numeric (store raw + normalized if possible)
   - currency → numeric + unit
   - text → string

5. Capture hierarchy:
   - Document → sections → blocks → elements

6. Do NOT hallucinate missing values. Use null.

--------------------------------
OUTPUT SCHEMA (STRICT)
--------------------------------
{
  "document_id": string,
  "metadata": {
    "title": string|null,
    "page_number": integer|null,
    "source": string|null
  },
  "sections": [
    {
      "section_title": string|null,
      "blocks": [
        {
          "block_type": "paragraph" | "table" | "list",
          "content": string|null,

          "table": {
            "table_title": string|null,
            "columns": [
              {
                "name": string,
                "unit": string|null
              }
            ],
            "rows": [
              {
                "row_label": string|null,
                "cells": [
                  {
                    "value": number|string|null,
                    "raw": string
                  }
                ]
              }
            ]
          },

          "list": [
            {
              "item": string
            }
          ]
        }
      ]
    }
  ]
}

--------------------------------
PARSING RULES FOR TABLES
--------------------------------
- First column is usually "row_label"
- Column headers may span multiple rows → merge them
- Parentheses indicate negative values
- Remove commas in numbers (e.g., "2,390" → 2390)
- Keep both:
  - "value": parsed numeric
  - "raw": original string

--------------------------------
FEW-SHOT EXEMPLARS
--------------------------------

Example 1 (Financial Rollforward Table)

INPUT:
Balance at beginning of fiscal year | 127.1
Additions for tax positions         | 103.8
Reductions due to settlement        | (4.0)
Balance at end of fiscal year       | 224.3

OUTPUT:
{
  "document_id": "example_1",
  "metadata": {},
  "sections": [
    {
      "section_title": null,
      "blocks": [
        {
          "block_type": "table",
          "content": null,
          "table": {
            "table_title": "Tax Position Rollforward",
            "columns": [
              {"name": "Item", "unit": null},
              {"name": "Amount", "unit": "$"}
            ],
            "rows": [
              {
                "row_label": "Balance at beginning of fiscal year",
                "cells": [{"value": 127.1, "raw": "127.1"}]
              },
              {
                "row_label": "Additions for tax positions",
                "cells": [{"value": 103.8, "raw": "103.8"}]
              },
              {
                "row_label": "Reductions due to settlement",
                "cells": [{"value": -4.0, "raw": "(4.0)"}]
              },
              {
                "row_label": "Balance at end of fiscal year",
                "cells": [{"value": 224.3, "raw": "224.3"}]
              }
            ]
          }
        }
      ]
    }
  ]
}

--------------------------------

Example 2 (Multi-column Business Table)

INPUT:
New Jersey | 704 | 25.7% | 660,580 | 20.3% | 2.7 | 22.3%

OUTPUT:
{
  "document_id": "example_2",
  "metadata": {},
  "sections": [
    {
      "section_title": "Operating Metrics by State",
      "blocks": [
        {
          "block_type": "table",
          "content": null,
          "table": {
            "table_title": null,
            "columns": [
              {"name": "State", "unit": null},
              {"name": "Revenue", "unit": "million USD"},
              {"name": "Revenue %", "unit": "%"},
              {"name": "Customers", "unit": null},
              {"name": "Customers %", "unit": "%"},
              {"name": "Population", "unit": "millions"},
              {"name": "Population %", "unit": "%"}
            ],
            "rows": [
              {
                "row_label": "New Jersey",
                "cells": [
                  {"value": 704, "raw": "704"},
                  {"value": 25.7, "raw": "25.7%"},
                  {"value": 660580, "raw": "660,580"},
                  {"value": 20.3, "raw": "20.3%"},
                  {"value": 2.7, "raw": "2.7"},
                  {"value": 22.3, "raw": "22.3%"}
                ]
              }
            ]
          }
        }
      ]
    }
  ]
}

--------------------------------

Example 3 (Timeline-style Table)

INPUT:
Balance as of January 1, 2015 | 127
Actual return on assets       | 12
Balance as of December 31     | 136

OUTPUT:
{
  "document_id": "example_3",
  "metadata": {},
  "sections": [
    {
      "section_title": "Asset Rollforward",
      "blocks": [
        {
          "block_type": "table",
          "content": null,
          "table": {
            "table_title": null,
            "columns": [
              {"name": "Description", "unit": null},
              {"name": "Value", "unit": "$"}
            ],
            "rows": [
              {
                "row_label": "Balance as of January 1, 2015",
                "cells": [{"value": 127, "raw": "127"}]
              },
              {
                "row_label": "Actual return on assets",
                "cells": [{"value": 12, "raw": "12"}]
              },
              {
                "row_label": "Balance as of December 31, 2015",
                "cells": [{"value": 136, "raw": "136"}]
              }
            ]
          }
        }
      ]
    }
  ]
}

--------------------------------
FINAL INSTRUCTION
--------------------------------
Parse the given document into the exact JSON schema above.
Return ONLY valid JSON. No explanations.
"""

        self.complex_tablevqa = """
You are a structure-aware document parser for a Retrieval-Augmented Generation (RAG) pipeline. Your task is to convert complex documents (including text, tables, multi-column layouts, and financial statements) into a clean, structured JSON representation that preserves semantic meaning, hierarchy, and table structure.

========================
CORE INSTRUCTIONS
========================
1. Preserve document structure:
   - Identify sections, headers, subheaders, paragraphs, and tables.
   - Maintain reading order (top-to-bottom, left-to-right unless clearly multi-column).
   - Group related elements logically.

2. Handle tables explicitly:
   - Detect tables and extract:
     - table_title (if present)
     - column_headers
     - row_headers (if applicable)
     - cells (2D array)
   - Preserve numerical values EXACTLY (no rounding, no formatting changes).
   - Keep units, currency symbols, and parentheses (e.g., negatives like "(526,431)").

3. Normalize text:
   - Remove visual artifacts (colors, borders, shading).
   - Keep meaningful formatting (e.g., indentation → hierarchy).
   - Do NOT hallucinate missing values.

4. Maintain semantic roles:
   Each extracted element MUST be labeled with one of:
   - "section"
   - "subsection"
   - "paragraph"
   - "table"
   - "table_row"
   - "table_cell"
   - "metadata"

5. Chunking for RAG:
   - Split output into semantically coherent chunks.
   - Each chunk should be self-contained and retrievable.
   - Tables must NOT be split across chunks.

========================
OUTPUT SCHEMA (STRICT)
========================
Return ONLY valid JSON in the following format:

{
  "document_id": "<string>",
  "chunks": [
    {
      "chunk_id": "<string>",
      "type": "text | table",
      "title": "<string or null>",
      "content": "<string (for text)>",
      "structure": {
        "section": "<string or null>",
        "subsection": "<string or null>"
      },
      "table": {
        "column_headers": ["<string>"],
        "rows": [
          {
            "row_header": "<string or null>",
            "cells": ["<string>"]
          }
        ]
      },
      "metadata": {
        "page_number": "<int or null>",
        "source": "<string or null>"
      }
    }
  ]
}

Rules:
- If type = "text", "table" must be null.
- If type = "table", "content" must be null.
- Never mix table and text in the same chunk.

========================
FEW-SHOT EXEMPLARS
========================

Example 1: Financial Table

INPUT (simplified):
"Assets
Cash and cash equivalents 63.0 79.0
Trade accounts receivable 495.5 465.9
Total assets 4,449.7 4,165.4"

OUTPUT:
{
  "document_id": "example_1",
  "chunks": [
    {
      "chunk_id": "c1",
      "type": "table",
      "title": "Assets",
      "content": null,
      "structure": {
        "section": "Assets",
        "subsection": null
      },
      "table": {
        "column_headers": ["2013", "2012"],
        "rows": [
          {
            "row_header": "Cash and cash equivalents",
            "cells": ["63.0", "79.0"]
          },
          {
            "row_header": "Trade accounts receivable",
            "cells": ["495.5", "465.9"]
          },
          {
            "row_header": "Total assets",
            "cells": ["4,449.7", "4,165.4"]
          }
        ]
      },
      "metadata": {
        "page_number": null,
        "source": null
      }
    }
  ]
}

----------------------------------------

Example 2: Cash Flow Table with Negatives

INPUT (simplified):
"Cash flows from financing activities:
Proceeds from issuance 991,835
Repayments (526,431)
Net cash (536,096)"

OUTPUT:
{
  "document_id": "example_2",
  "chunks": [
    {
      "chunk_id": "c1",
      "type": "table",
      "title": "Cash flows from financing activities",
      "content": null,
      "structure": {
        "section": "Cash flows",
        "subsection": "Financing activities"
      },
      "table": {
        "column_headers": ["Amount"],
        "rows": [
          {
            "row_header": "Proceeds from issuance",
            "cells": ["991,835"]
          },
          {
            "row_header": "Repayments",
            "cells": ["(526,431)"]
          },
          {
            "row_header": "Net cash",
            "cells": ["(536,096)"]
          }
        ]
      },
      "metadata": {
        "page_number": null,
        "source": null
      }
    }
  ]
}

----------------------------------------

Example 3: Structured Episode Table

INPUT (simplified):
"episode | series | title | air date
27 | 1 | return to genesis | 5 april 2008"

OUTPUT:
{
  "document_id": "example_3",
  "chunks": [
    {
      "chunk_id": "c1",
      "type": "table",
      "title": null,
      "content": null,
      "structure": {
        "section": null,
        "subsection": null
      },
      "table": {
        "column_headers": ["episode", "series", "episode title", "original air date"],
        "rows": [
          {
            "row_header": null,
            "cells": ["27", "1", "return to genesis", "5 april 2008"]
          }
        ]
      },
      "metadata": {
        "page_number": null,
        "source": null
      }
    }
  ]
}

========================
FINAL REQUIREMENTS
========================
- Output ONLY JSON (no explanations).
- Ensure schema validity.
- Preserve all numeric precision and text exactly.
- Be robust to noisy layouts, merged cells, and multi-column documents.
- Prioritize semantic correctness over visual formatting.
"""

        self.complex_tatdqa = """
You are a Structure-Aware Document Parser for a Retrieval-Augmented Generation (RAG) pipeline.

Your task is to convert visually rich documents (including tables, financial statements, and semi-structured layouts) into a clean, structured JSON representation that preserves semantic meaning, hierarchy, and table structure.

----------------------------------------
OUTPUT FORMAT (STRICT JSON ONLY)
----------------------------------------

Return a single JSON object with the following schema:

{
  "document_metadata": {
    "title": string | null,
    "date": string | null,
    "page_number": integer | null
  },
  "sections": [
    {
      "section_id": string,
      "section_title": string,
      "section_type": "paragraph" | "table" | "mixed",
      "content": string | null,
      "table": {
        "caption": string | null,
        "columns": [string],
        "rows": [
          {
            "row_header": string | null,
            "cells": [string]
          }
        ],
        "units": string | null,
        "notes": [string]
      } | null
    }
  ],
  "entities": [
    {
      "text": string,
      "type": "date" | "currency" | "percentage" | "organization" | "other"
    }
  ]
}

----------------------------------------
PARSING RULES
----------------------------------------

1. STRUCTURE DETECTION
- Identify section headers (bold, larger font, or separated by lines).
- Group content under the nearest preceding header.
- Distinguish between paragraph text and tables.

2. TABLE PARSING
- Extract column headers exactly as shown.
- Preserve row labels (e.g., “Deferred revenue”).
- Keep numeric values as strings (do NOT normalize).
- Maintain column alignment across years or categories.
- Capture units (e.g., "$M", "%") at table-level if shared.

3. MULTI-LEVEL TABLES
- If tables contain grouped columns (e.g., “U.S. Pension Plans” vs “Non-U.S. Pension Plans”), flatten into columns with fully qualified names:
  Example: "2019_U.S.", "2019_Non-U.S."

4. MISSING VALUES
- Represent missing entries as null (not "-").

5. TEXT NORMALIZATION
- Preserve original wording.
- Do NOT summarize or infer.
- Keep numbers, punctuation, and capitalization unchanged.

6. ENTITY EXTRACTION
- Extract key values such as:
  - Dates (e.g., “31 March 2019”)
  - Monetary values (e.g., “40.9” under $M context)
  - Percentages (e.g., “3.15%”)

7. NO HALLUCINATION
- Only extract what is explicitly visible.
- If unsure, use null.

----------------------------------------
FEW-SHOT EXEMPLARS
----------------------------------------

### Example 1 (Financial Table)

INPUT (simplified):
Deferred income tax assets in relation to:
Deferred revenue      40.9   40.0
Tax losses UK         37.0   24.2
Total                115.0  120.7
(Columns: 2019, 2018; Units: $M)

OUTPUT:
{
  "document_metadata": {
    "title": null,
    "date": null,
    "page_number": null
  },
  "sections": [
    {
      "section_id": "s1",
      "section_title": "Deferred income tax assets in relation to",
      "section_type": "table",
      "content": null,
      "table": {
        "caption": null,
        "columns": ["2019", "2018"],
        "rows": [
          {
            "row_header": "Deferred revenue",
            "cells": ["40.9", "40.0"]
          },
          {
            "row_header": "Tax losses UK",
            "cells": ["37.0", "24.2"]
          },
          {
            "row_header": "Total",
            "cells": ["115.0", "120.7"]
          }
        ],
        "units": "$M",
        "notes": []
      }
    }
  ],
  "entities": [
    {"text": "40.9", "type": "currency"},
    {"text": "115.0", "type": "currency"}
  ]
}

----------------------------------------

### Example 2 (Multi-Column Pension Table)

INPUT (simplified):
Years Ended December 31
U.S. Plans: 2019 2018
Non-U.S.: 2019 2018

Interest cost: 7724 7123 | 31 42
Expected return: (12187) (12898) | (17) (25)

OUTPUT:
{
  "document_metadata": {
    "title": "Years Ended December 31",
    "date": null,
    "page_number": null
  },
  "sections": [
    {
      "section_id": "s1",
      "section_title": "Pension Costs",
      "section_type": "table",
      "content": null,
      "table": {
        "caption": null,
        "columns": [
          "2019_U.S.", "2018_U.S.",
          "2019_Non-U.S.", "2018_Non-U.S."
        ],
        "rows": [
          {
            "row_header": "Interest cost",
            "cells": ["7724", "7123", "31", "42"]
          },
          {
            "row_header": "Expected return on plan assets",
            "cells": ["(12187)", "(12898)", "(17)", "(25)"]
          }
        ],
        "units": null,
        "notes": []
      }
    }
  ],
  "entities": []
}

----------------------------------------

### Example 3 (Mixed Content)

INPUT:
Balance January 1, 2018: 33,145
Additions: 242
Balance December 31, 2018: 33,387

OUTPUT:
{
  "document_metadata": {
    "title": null,
    "date": null,
    "page_number": null
  },
  "sections": [
    {
      "section_id": "s1",
      "section_title": "Balances",
      "section_type": "table",
      "content": null,
      "table": {
        "caption": null,
        "columns": ["Value"],
        "rows": [
          {"row_header": "Balance January 1, 2018", "cells": ["33,145"]},
          {"row_header": "Additions", "cells": ["242"]},
          {"row_header": "Balance December 31, 2018", "cells": ["33,387"]}
        ],
        "units": null,
        "notes": []
      }
    }
  ],
  "entities": [
    {"text": "January 1, 2018", "type": "date"}
  ]
}

----------------------------------------
FINAL INSTRUCTION
----------------------------------------

Parse the provided document into the exact JSON schema above.

Return ONLY valid JSON. No explanations. No markdown.
"""

        self.complex_wikitq = """
You are a structure-aware document parser designed for multimodal documents (e.g., scanned pages, PDFs, webpages). Your goal is to convert visually complex documents into a structured, machine-readable representation suitable for Retrieval-Augmented Generation (RAG).

You MUST:
- Preserve document structure (sections, tables, lists, headers).
- Capture semantic relationships (e.g., table rows, grouped entries).
- Avoid hallucination. Only extract what is visible.
- Normalize noisy text (OCR errors, broken lines) while preserving meaning.
- Represent output strictly following the JSON schema below.

----------------------------------------
OUTPUT SCHEMA (STRICT)
----------------------------------------

{
  "document_type": "string (e.g., table, mixed, list, article)",
  "sections": [
    {
      "section_title": "string or null",
      "content": [
        {
          "type": "paragraph",
          "text": "string"
        },
        {
          "type": "table",
          "table_name": "string or null",
          "columns": ["col1", "col2", ...],
          "rows": [
            {
              "col1": "value",
              "col2": "value"
            }
          ],
          "notes": "string or null"
        },
        {
          "type": "list",
          "items": ["item1", "item2"]
        }
      ]
    }
  ],
  "metadata": {
    "language": "string",
    "has_tables": "boolean",
    "has_multiple_sections": "boolean"
  }
}

----------------------------------------
PARSING RULES
----------------------------------------

1. SECTION DETECTION
- Identify section headers (e.g., "Film", "Television").
- Group content under correct section.
- If no header exists, use section_title = null.

2. TABLE DETECTION
- Detect tables via alignment, repeated column patterns, or headers.
- Normalize multi-line cells into single values.
- If a row spans multiple lines, merge logically.
- Keep column names consistent.

3. ROW NORMALIZATION
- Each row must be a complete entity.
- Merge fragmented entries (e.g., film title broken across lines).
- Maintain relationships across columns (Year ↔ Title ↔ Role ↔ Notes).

4. MULTI-ROW GROUPING
- If a year or category applies to multiple entries, propagate it to all rows.

5. NOISE HANDLING
- Remove visual artifacts (e.g., stray symbols, broken formatting).
- Keep meaningful parentheses (e.g., "(television film)").

6. LINK TEXT
- Extract only visible text, ignore hyperlink formatting.

7. LANGUAGE
- Preserve original language of content.

----------------------------------------
FEW-SHOT EXEMPLARS
----------------------------------------

### EXAMPLE 1: SIMPLE TABLE

INPUT (visual):
Year | Film | Role
1991 | Secret Friends | Helen
1992 | Leon the Pig Farmer | Lisa

OUTPUT:
{
  "document_type": "table",
  "sections": [
    {
      "section_title": null,
      "content": [
        {
          "type": "table",
          "table_name": null,
          "columns": ["Year", "Film", "Role"],
          "rows": [
            {"Year": "1991", "Film": "Secret Friends", "Role": "Helen"},
            {"Year": "1992", "Film": "Leon the Pig Farmer", "Role": "Lisa"}
          ],
          "notes": null
        }
      ]
    }
  ],
  "metadata": {
    "language": "en",
    "has_tables": true,
    "has_multiple_sections": false
  }
}

----------------------------------------

### EXAMPLE 2: MULTI-SECTION FILMOGRAPHY

INPUT (visual):
Film
1985 King David Tamar
1985 Mussolini: The Untold Story Gina Ruberti (television mini-series)

Television
1982 Into the Labyrinth Christine Episode: Phantom

OUTPUT:
{
  "document_type": "mixed",
  "sections": [
    {
      "section_title": "Film",
      "content": [
        {
          "type": "table",
          "table_name": "Filmography",
          "columns": ["Year", "Title", "Role", "Notes"],
          "rows": [
            {
              "Year": "1985",
              "Title": "King David",
              "Role": "Tamar",
              "Notes": null
            },
            {
              "Year": "1985",
              "Title": "Mussolini: The Untold Story",
              "Role": "Gina Ruberti",
              "Notes": "television mini-series"
            }
          ],
          "notes": null
        }
      ]
    },
    {
      "section_title": "Television",
      "content": [
        {
          "type": "table",
          "table_name": "Television roles",
          "columns": ["Year", "Title", "Role", "Notes"],
          "rows": [
            {
              "Year": "1982",
              "Title": "Into the Labyrinth",
              "Role": "Christine",
              "Notes": "Episode: Phantom"
            }
          ],
          "notes": null
        }
      ]
    }
  ],
  "metadata": {
    "language": "en",
    "has_tables": true,
    "has_multiple_sections": true
  }
}

----------------------------------------

### EXAMPLE 3: COMPLEX TABLE WITH GROUP HEADERS

INPUT (visual):
City served / Location | ICAO | Airport name | Usage
01: Ain
Ambérieu LFXA Ambérieu-en-Bugey Air Base Public/Mil.

OUTPUT:
{
  "document_type": "table",
  "sections": [
    {
      "section_title": "Ain",
      "content": [
        {
          "type": "table",
          "table_name": "Airports",
          "columns": ["City served / Location", "ICAO", "Airport name", "Usage"],
          "rows": [
            {
              "City served / Location": "Ambérieu",
              "ICAO": "LFXA",
              "Airport name": "Ambérieu-en-Bugey Air Base",
              "Usage": "Public/Mil."
            }
          ],
          "notes": "Region code: 01"
        }
      ]
    }
  ],
  "metadata": {
    "language": "en",
    "has_tables": true,
    "has_multiple_sections": true
  }
}

----------------------------------------

### EXAMPLE 4: MULTI-LINE ROW MERGING

INPUT (visual):
1992
Vsetko co mam rad
Ann

OUTPUT ROW:
{
  "Year": "1992",
  "Title": "Vsetko co mam rad",
  "Role": "Ann"
}

========================
FINAL INSTRUCTION
========================

Parse the given document into the exact JSON schema.
Ensure:
- No missing columns in tables.
- No duplicated or fragmented rows.
- Clean, normalized, and structured output.
- Strict JSON formatting (no extra text outside JSON).
"""

        self.complex_mpdocvqa = """
You are a Structure-Aware Vision-Language Document Parser designed for RAG pipelines.

Your task is to extract structured, lossless, and semantically faithful representations from visually complex documents (including tables, multi-column layouts, financial statements, scanned text, and mixed content).

You MUST strictly follow the output schema and rules below.

--------------------------------------------------
OUTPUT SCHEMA (STRICT JSON, NO EXTRA TEXT)
--------------------------------------------------

{
  "document_metadata": {
    "title": string | null,
    "subtitle": string | null,
    "date": string | null,
    "page_number": string | null,
    "source": string | null
  },
  "layout": {
    "reading_order": [string],
    "notes": string | null
  },
  "sections": [
    {
      "section_id": string,
      "section_title": string,
      "section_type": "table" | "paragraph" | "list" | "mixed",
      "hierarchy_level": integer,
      "content": string | null,
      "list_items": [string] | null,
      "table": {
        "caption": string | null,
        "column_groups": [
          {
            "group_name": string,
            "columns": [string]
          }
        ],
        "columns": [string],
        "rows": [
          {
            "row_id": string,
            "row_header": string | null,
            "sub_headers": [string] | null,
            "cells": [string]
          }
        ],
        "units": string | null,
        "footnotes": [string],
        "missing_value_token": string | null
      } | null
    }
  ],
  "entities": [
    {
      "text": string,
      "type": "person" | "organization" | "date" | "year" | "currency" | "percentage" | "location" | "other"
    }
  ]
}

--------------------------------------------------
CORE PARSING RULES
--------------------------------------------------

1. LAYOUT & READING ORDER
- Reconstruct logical reading order (top-to-bottom, left-to-right unless clear multi-column structure).
- Store section_id sequence in layout.reading_order.
- Detect multi-column layouts and preserve grouping via sections.

2. SECTION DETECTION
- Use visual cues: font size, boldness, spacing, capitalization.
- Assign hierarchy_level:
  - 1 = main title
  - 2 = section header
  - 3+ = subsections

3. TABLE DETECTION
- Treat aligned numeric/text grids as tables even if no borders exist.
- Extract ALL headers exactly as shown.

4. MULTI-LEVEL / GROUPED TABLES
- If columns are grouped (e.g., “1981”, “Approved 1982”, “Working 1982”):
  - Use column_groups to preserve structure.
  - Also provide flattened columns (e.g., "1981", "Approved_1982", "Working_1982").

5. ROW STRUCTURE
- Preserve row hierarchy:
  - Main row label → row_header
  - Indented items → sub_headers or separate rows (depending on alignment)

6. VALUES
- Keep ALL values as strings.
- Preserve:
  - Parentheses: "(13.8)"
  - Dashes: "-", "--", "-0-"
- Convert missing values to null ONLY if clearly empty (not symbolic).

7. UNITS
- Extract units from headers or captions (e.g., "(In Thousand $)", "%", "$M").

8. TEXT BLOCKS
- Preserve verbatim text.
- Do NOT summarize.

9. LISTS
- Convert vertically aligned name/year pairs into structured rows or list_items.

10. ENTITY EXTRACTION
- Extract:
  - Years (e.g., "1982")
  - Names (e.g., "Kawasaki")
  - Monetary values
  - Percentages

11. NO HALLUCINATION
- If uncertain → null.
- Do NOT infer missing headers or values.

--------------------------------------------------
FEW-SHOT EXEMPLARS
--------------------------------------------------

### Example 1: Simple Name-Year List

INPUT:
TREATMENT OF HYPERTENSION
Ambard, Beaujard 1904
Allen, Sherrill 1922
Kempner 1948

OUTPUT:
{
  "document_metadata": {
    "title": "TREATMENT OF HYPERTENSION",
    "subtitle": null,
    "date": null,
    "page_number": null,
    "source": null
  },
  "layout": {
    "reading_order": ["s1"],
    "notes": null
  },
  "sections": [
    {
      "section_id": "s1",
      "section_title": "Treatment of Hypertension",
      "section_type": "table",
      "hierarchy_level": 1,
      "content": null,
      "list_items": null,
      "table": {
        "caption": null,
        "column_groups": [],
        "columns": ["Name", "Year"],
        "rows": [
          {"row_id": "r1", "row_header": "Ambard, Beaujard", "sub_headers": null, "cells": ["1904"]},
          {"row_id": "r2", "row_header": "Allen, Sherrill", "sub_headers": null, "cells": ["1922"]},
          {"row_id": "r3", "row_header": "Kempner", "sub_headers": null, "cells": ["1948"]}
        ],
        "units": null,
        "footnotes": [],
        "missing_value_token": null
      }
    }
  ],
  "entities": [
    {"text": "1904", "type": "year"},
    {"text": "1922", "type": "year"},
    {"text": "1948", "type": "year"}
  ]
}

--------------------------------------------------

### Example 2: Financial Table with Column Groups

INPUT:
WORKING 1982 BUDGET (In Thousand $)
Columns: 1981 | Approved 1982 | Working 1982
Swanson Interests: 362.0 | 362.0 | 362.0
Consultant Fees: 19.5 | 15.0 | 15.0

OUTPUT:
{
  "document_metadata": {
    "title": "WORKING 1982 BUDGET",
    "subtitle": null,
    "date": "1982",
    "page_number": null,
    "source": null
  },
  "layout": {
    "reading_order": ["s1"],
    "notes": null
  },
  "sections": [
    {
      "section_id": "s1",
      "section_title": "Income",
      "section_type": "table",
      "hierarchy_level": 2,
      "content": null,
      "list_items": null,
      "table": {
        "caption": "(In Thousand $)",
        "column_groups": [
          {"group_name": "1981", "columns": ["1981"]},
          {"group_name": "1982", "columns": ["Approved_1982", "Working_1982"]}
        ],
        "columns": ["1981", "Approved_1982", "Working_1982"],
        "rows": [
          {"row_id": "r1", "row_header": "Swanson Interests", "sub_headers": null, "cells": ["362.0", "362.0", "362.0"]},
          {"row_id": "r2", "row_header": "Consultant Fees", "sub_headers": null, "cells": ["19.5", "15.0", "15.0"]}
        ],
        "units": "Thousand $",
        "footnotes": [],
        "missing_value_token": null
      }
    }
  ],
  "entities": [
    {"text": "1982", "type": "year"}
  ]
}

--------------------------------------------------

### Example 3: Multi-Column Text Table

INPUT:
Treatment | Placebo | Adverse Effects
Soy isoflavones | No significant reduction | Mild gastrointestinal effects

OUTPUT:
{
  "document_metadata": {
    "title": null,
    "subtitle": null,
    "date": null,
    "page_number": null,
    "source": null
  },
  "layout": {
    "reading_order": ["s1"],
    "notes": null
  },
  "sections": [
    {
      "section_id": "s1",
      "section_title": "Clinical Comparison",
      "section_type": "table",
      "hierarchy_level": 2,
      "content": null,
      "list_items": null,
      "table": {
        "caption": null,
        "column_groups": [],
        "columns": ["Treatment", "Placebo", "Adverse Effects"],
        "rows": [
          {
            "row_id": "r1",
            "row_header": "Soy isoflavones",
            "sub_headers": null,
            "cells": ["No significant reduction", "Mild gastrointestinal effects"]
          }
        ],
        "units": null,
        "footnotes": [],
        "missing_value_token": null
      }
    }
  ],
  "entities": []
}

--------------------------------------------------
FINAL INSTRUCTION
--------------------------------------------------

Parse the input document into the exact JSON schema above.

Return ONLY valid JSON.
No explanations.
No markdown.
No additional text.
"""


class LLMPrompts:
    def __init__(self):

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