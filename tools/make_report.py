"""Generate docs/results_chart.png and docs/annual_report_2025.pdf (run once: python tools/make_report.py)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

DOCS = Path(__file__).parent.parent / "docs"

# Quarterly client growth (fictional) and package mix
quarters = ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"]
clients = [14, 17, 21, 26]
mix = {"Starter": 9, "Growth": 11, "Premier": 6}

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
a.bar(quarters, clients, color="#c2410c")
a.set_title("Active clients per quarter, 2025")
for i, v in enumerate(clients):
    a.text(i, v + 0.4, str(v), ha="center")
b.pie(mix.values(), labels=[f"{k} ({v})" for k, v in mix.items()], colors=["#fdba74", "#ea580c", "#7c2d12"], startangle=90)
b.set_title("Clients by package, end of 2025")
fig.tight_layout()
fig.savefig(DOCS / "results_chart.png")

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", "B", 20)
pdf.cell(0, 12, "Bright Studio - Annual Report 2025", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 11)
body = (
    "Bright Studio grew from 14 active clients in Q1 2025 to 26 active clients in Q4 2025, an 86% increase. "
    "Revenue for 2025 was $1.42 million, up 38% from 2024 ($1.03 million).\n\n"
    "Clients by package at the end of 2025: 9 on the Starter package, 11 on the Growth package, "
    "and 6 on the Premier package. The Growth package remained the most popular option.\n\n"
    "Client retention: 91% of clients renewed after their minimum contract term. "
    "The average client stays 14 months.\n\n"
    "Team: we hired Aisha Rahman as Account Manager in March 2025, bringing the team to six people. "
    "Plans for 2026 include a seventh hire (a video specialist) and a possible second office in Dallas.\n\n"
    "Customer satisfaction: our average client satisfaction score in 2025 was 4.7 out of 5, "
    "based on quarterly surveys with a 78% response rate."
)
pdf.multi_cell(0, 6, body)
pdf.ln(4)
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "Chart: active clients per quarter and package mix", new_x="LMARGIN", new_y="NEXT")
pdf.image(str(DOCS / "results_chart.png"), w=180)
pdf.set_font("Helvetica", "", 10)
pdf.multi_cell(0, 5, "Chart data - Active clients: Q1 14, Q2 17, Q3 21, Q4 26. Package mix: Starter 9, Growth 11, Premier 6.")
pdf.output(str(DOCS / "annual_report_2025.pdf"))
print("Wrote chart and PDF")
