"""House style for the results tabs of the Google Sheet, generated from the payloads so it always lines up.

Reads every *_sheet_payload.json for the styled tabs and writes outputs/sheet_style_requests.json: a list of
spreadsheets.batchUpdate requests (formats only, never values). Needs the tabs' sheet IDs:

    python -m src.sheet_style '{"Overview": 123, "Catalogue": 456, ...}'

Style: Arial; 16pt bold title; long notes merged across the table and wrapped; 12pt bold section labels;
navy header rows with white bold text; number formats chosen from each column's header; hairline row rules.
The Phase 1-2 tabs (Forecast summary, Scores, Significance, Coefficients, Daily forecasts, Run info) keep their
existing hand-made formatting and charts and are not touched.
"""

import json
import math
import sys

from src.data import ROOT

OUT = ROOT / "outputs"
NAVY = {"red": 0.118, "green": 0.220, "blue": 0.388}
WHITE = {"red": 1, "green": 1, "blue": 1}
RULE = {"red": 0.882, "green": 0.878, "blue": 0.851}
GREEN_BG, RED_BG, AMBER_BG = ({"red": 0.85, "green": 0.94, "blue": 0.85}, {"red": 0.98, "green": 0.87, "blue": 0.86},
                              {"red": 0.99, "green": 0.94, "blue": 0.82})
PCT, NUM2, NUM3, NUM4, SCI, USD = "0.0%", "0.00", "0.000", "0.0000", "0.00E+00", "$#,##0"
PX_PER_CHAR, LINE_PX = 6.6, 17

PAYLOADS = {"Slope model": "slope", "Vol-managed SPY": "volmanaged", "Higher returns": "higher_return",
            "Multi-asset": "multiasset", "Robustness": "robustness", "International": "international"}
WIDTHS = {  # pixels per column; anything unlisted gets DEFAULT_WIDTHS
    "Overview": [300, 380, 300, 170, 120],
    "Catalogue": [60, 250, 300, 340, 190, 130],
    "Methods": [230, 720],
    "International": [70, 130, 110, 130] + [105] * 12,
}
DEFAULT_WIDTHS = [290] + [115] * 15


def fmt_for(header: str, table_kind: str) -> str | None:
    h = header.lower()
    if table_kind == "returns":
        return PCT
    if "final value" in h:
        return USD
    if any(k in h for k in ["cagr", "return", "volatility", "drawdown", "max dd", "worst day", "confidence",
                            "hit rate", "buy & hold", "managed"]) and "sharpe" not in h:
        return PCT
    if "qlike" in h or "mean loss" in h:
        return NUM4
    if h == "mse":
        return SCI
    if "p-value" in h:
        return "0.0E+00"
    if any(k in h for k in ["sharpe", "low 95", "high 95", "difference", "corr"]):
        return NUM3
    if any(k in h for k in ["calmar", "exposure", "turnover", "dm stat"]):
        return NUM2
    return None


def _grid(sid, r0, r1, c0, c1):
    return {"sheetId": sid, "startRowIndex": r0, "endRowIndex": r1, "startColumnIndex": c0, "endColumnIndex": c1}


def _cell_fmt(sid, r0, r1, c0, c1, fmt, fields):
    return {"repeatCell": {"range": _grid(sid, r0, r1, c0, c1), "cell": {"userEnteredFormat": fmt}, "fields": fields}}


def classify(rows):
    """Label each row: title, note, label, header, data, special, blank."""
    kinds = []
    for i, r in enumerate(rows):
        cells = [c for c in r if c not in ("", None)]
        if i == 0:
            kinds.append("title")
        elif not cells:
            kinds.append("blank")
        elif isinstance(r[0], str) and r[0].startswith("Decision:"):
            kinds.append("special")
        elif len(r) == 1 or len(cells) == 1:
            nxt = rows[i + 1] if i + 1 < len(rows) else []
            text = str(r[0])
            kinds.append("label" if len(text) < 110 and (len(nxt) >= 2 or not text.rstrip().endswith(".")) else "note")
        elif kinds and kinds[-1] in ("label", "blank", "note"):
            kinds.append("header")
        else:
            kinds.append("data")
    return kinds


def tables(kinds):
    """(header row, first data row, end row) for each header followed by data rows."""
    out = []
    for i, k in enumerate(kinds):
        if k == "header":
            j = i + 1
            while j < len(kinds) and kinds[j] == "data":
                j += 1
            if j > i + 1:
                out.append((i, i + 1, j))
    return out


def requests_for(name: str, rows: list, sid: int) -> list:
    ncol = max(len(r) for r in rows)
    widths = (WIDTHS.get(name) or DEFAULT_WIDTHS)[:ncol]
    widths += [DEFAULT_WIDTHS[-1]] * (ncol - len(widths))
    total_px = sum(widths)
    nrow = len(rows) + 5
    req = [{"unmergeCells": {"range": _grid(sid, 0, 1000, 0, 26)}},
           _cell_fmt(sid, 0, 1000, 0, 26, {"textFormat": {"fontFamily": "Arial", "fontSize": 10},
                                           "verticalAlignment": "TOP", "wrapStrategy": "WRAP"}, "userEnteredFormat"),
           {"updateSheetProperties": {"properties": {"sheetId": sid, "gridProperties": {"hideGridlines": True}},
                                      "fields": "gridProperties.hideGridlines"}},
           {"updateDimensionProperties": {"range": {"sheetId": sid, "dimension": "ROWS", "startIndex": 0, "endIndex": nrow},
                                          "properties": {"pixelSize": 21}, "fields": "pixelSize"}}]
    for c, w in enumerate(widths):
        req.append({"updateDimensionProperties": {"range": {"sheetId": sid, "dimension": "COLUMNS", "startIndex": c,
                                                            "endIndex": c + 1},
                                                  "properties": {"pixelSize": w}, "fields": "pixelSize"}})
    kinds = classify(rows)
    header = None
    for i, (r, k) in enumerate(zip(rows, kinds)):
        if k == "title":
            req.append(_cell_fmt(sid, i, i + 1, 0, 1, {"textFormat": {"fontFamily": "Arial", "bold": True, "fontSize": 16},
                                                       "wrapStrategy": "OVERFLOW_CELL"}, "userEnteredFormat"))
            req.append({"updateDimensionProperties": {"range": {"sheetId": sid, "dimension": "ROWS", "startIndex": i,
                                                                "endIndex": i + 1},
                                                      "properties": {"pixelSize": 32}, "fields": "pixelSize"}})
        elif k == "note":
            lines = max(1, math.ceil(len(str(r[0])) * PX_PER_CHAR / max(total_px - 20, 200)))
            req.append({"mergeCells": {"range": _grid(sid, i, i + 1, 0, ncol), "mergeType": "MERGE_ALL"}})
            req.append(_cell_fmt(sid, i, i + 1, 0, ncol, {"textFormat": {"fontFamily": "Arial",
                                                                         "foregroundColor": {"red": .32, "green": .32, "blue": .31}},
                                                          "wrapStrategy": "WRAP", "verticalAlignment": "TOP"},
                                 "userEnteredFormat"))
            req.append({"updateDimensionProperties": {"range": {"sheetId": sid, "dimension": "ROWS", "startIndex": i,
                                                                "endIndex": i + 1},
                                                      "properties": {"pixelSize": lines * LINE_PX + 8}, "fields": "pixelSize"}})
        elif k == "label":
            req.append(_cell_fmt(sid, i, i + 1, 0, 1, {"textFormat": {"fontFamily": "Arial", "bold": True, "fontSize": 12},
                                                       "wrapStrategy": "OVERFLOW_CELL", "verticalAlignment": "BOTTOM"},
                                 "userEnteredFormat"))
        elif k == "special":
            req.append(_cell_fmt(sid, i, i + 1, 0, len(r), {"textFormat": {"fontFamily": "Arial", "bold": True},
                                                            "backgroundColor": AMBER_BG, "wrapStrategy": "WRAP"},
                                 "userEnteredFormat"))
        elif k == "header":
            header = [str(c) for c in r]
            req.append(_cell_fmt(sid, i, i + 1, 0, len(r), {"textFormat": {"fontFamily": "Arial", "bold": True,
                                                                           "foregroundColor": WHITE},
                                                            "backgroundColor": NAVY, "wrapStrategy": "WRAP",
                                                            "verticalAlignment": "MIDDLE"}, "userEnteredFormat"))
        elif k == "data" and header:
            # longest text in a data row drives its height when it wraps
            texts = [(c, v) for c, v in enumerate(r) if isinstance(v, str)]
            lines = max((math.ceil(len(v) * PX_PER_CHAR / max(widths[c] - 8, 40)) for c, v in texts), default=1)
            if lines > 1:
                req.append({"updateDimensionProperties": {"range": {"sheetId": sid, "dimension": "ROWS", "startIndex": i,
                                                                    "endIndex": i + 1},
                                                          "properties": {"pixelSize": lines * LINE_PX + 6},
                                                          "fields": "pixelSize"}})
    # one border and one number format per table column, over each block of data rows
    for h0, d0, d1 in tables(kinds):
        hdr = [str(c) for c in rows[h0]]
        kind = "returns" if hdr[0] == "Year" else "normal"
        req.append({"updateBorders": {"range": _grid(sid, d0, d1, 0, len(hdr)),
                                      "innerHorizontal": {"style": "SOLID", "width": 1, "colorStyle": {"rgbColor": RULE}},
                                      "bottom": {"style": "SOLID", "width": 1, "colorStyle": {"rgbColor": RULE}}}})
        fmts = [fmt_for(h, kind) if c > 0 and any(c < len(rows[i]) and isinstance(rows[i][c], (int, float))
                                                  for i in range(d0, d1)) else None for c, h in enumerate(hdr)]
        c = 0
        while c < len(fmts):  # one request per run of adjacent columns sharing a format
            e = c + 1
            while e < len(fmts) and fmts[e] == fmts[c]:
                e += 1
            if fmts[c]:
                req.append(_cell_fmt(sid, d0, d1, c, e, {"numberFormat": {"type": "NUMBER", "pattern": fmts[c]},
                                                         "horizontalAlignment": "RIGHT"},
                                     "userEnteredFormat.numberFormat,userEnteredFormat.horizontalAlignment"))
            c = e
    # verdict / pass colouring
    for i, (r, k) in enumerate(zip(rows, kinds)):
        if k != "data":
            continue
        for c, v in enumerate(r):
            if not isinstance(v, str):
                continue
            bg = (GREEN_BG if v == "PASS" or v.startswith("Kept") else
                  RED_BG if v.startswith("Rejected") or v == "fail" else
                  AMBER_BG if v in ("Crash insurance only", "Risk reduction only")
                  or v.startswith("More return for more risk") else None)
            if bg and (name in ("Catalogue", "International")):
                req.append(_cell_fmt(sid, i, i + 1, c, c + 1, {"backgroundColor": bg},
                                     "userEnteredFormat.backgroundColor"))
    return req


def collapse(reqs: list) -> list:
    """Merge back-to-back dimension-size requests that set the same size on adjacent ranges."""
    out = []
    for r in reqs:
        prev = out[-1] if out else None
        if prev and "updateDimensionProperties" in r and "updateDimensionProperties" in prev:
            a, b = prev["updateDimensionProperties"], r["updateDimensionProperties"]
            if (a["range"]["sheetId"] == b["range"]["sheetId"] and a["range"]["dimension"] == b["range"]["dimension"]
                    and a["properties"] == b["properties"] and a["range"]["endIndex"] == b["range"]["startIndex"]):
                a["range"]["endIndex"] = b["range"]["endIndex"]
                continue
        out.append(r)
    return out


def main(sheet_ids: dict) -> None:
    tabs = {}
    for t in json.load(open(OUT / "overview_sheet_payload.json"))["tabs"]:
        tabs[t["range"].split("'")[1]] = t["data"]
    for name, stem in PAYLOADS.items():
        tabs[name] = json.load(open(OUT / f"{stem}_sheet_payload.json"))["data"]
    reqs = []
    for name, rows in tabs.items():
        if name not in sheet_ids:
            raise KeyError(f"no sheet id for tab {name!r}")
        reqs += collapse(requests_for(name, rows, int(sheet_ids[name])))
    with open(OUT / "sheet_style_requests.json", "w") as f:
        json.dump(reqs, f, separators=(",", ":"))
    by_tab = {}
    for name, sid in sheet_ids.items():  # per-tab files, small enough for one batchUpdate call each
        part = [r for r in reqs if f'"sheetId":{int(sid)},' in json.dumps(r, separators=(",", ":"))
                or f'"sheetId":{int(sid)}}}' in json.dumps(r, separators=(",", ":"))]
        by_tab[name] = part
    with open(OUT / "sheet_style_requests_by_tab.json", "w") as f:
        json.dump(by_tab, f, separators=(",", ":"))
    print(f"{len(reqs)} requests for {len(tabs)} tabs -> outputs/sheet_style_requests.json")


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
