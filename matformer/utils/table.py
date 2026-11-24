# matformer/utils/table.py
"""Table: the framework's structured command result.

Command interpreters (MDat's facade, the future task/registry consoles, the server's RPCs) return
a Table instead of pre-formatted text when they have rows to show. Every consumer renders it its
own way, so the data is produced once:

    TUI / console   -> str(table): aligned, monospace, wandb-style
    server / GUI    -> table.asdict(): plain JSON over the wire, drawn as a real table widget
"""


class Table:
    def __init__(self, columns, rows):
        """columns: [str]; rows: list of dicts (keys picked from columns) or of lists."""
        self.columns = list(columns)
        self.rows = [[r.get(c, "") for c in self.columns] if isinstance(r, dict) else list(r)
                     for r in rows]

    def __len__(self):
        return len(self.rows)

    def __str__(self):
        return self.text()

    def __repr__(self):
        return f"Table({self.columns}, {len(self.rows)} rows)"

    def text(self) -> str:
        """Aligned text rendering for terminals; empty tables say so instead of printing nothing."""
        if not self.rows:
            return "(empty)"
        cells = [[_fmt(v) for v in row] for row in self.rows]
        widths = [max(len(self.columns[i]), *(len(row[i]) for row in cells))
                  for i in range(len(self.columns))]
        head = "  ".join(c.upper().ljust(w) for c, w in zip(self.columns, widths))
        return "\n".join([head, "-" * len(head)]
                         + ["  ".join(v.rjust(w) if isinstance(value, (int, float)) and not isinstance(value, bool)
                                      else v.ljust(w)
                                      for v, w, value in zip(row, widths, cells_r))
                            for row, cells_r in zip(cells, self.rows)])

    def asdict(self):
        """JSON form; the __table__ marker tells the GUI this is rows, not prose."""
        return {"__table__": True, "columns": self.columns,
                "rows": [dict(zip(self.columns, row)) for row in self.rows]}


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
