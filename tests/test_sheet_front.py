from src.sheet_overview import catalogue, load, overview
from src.sheet_style import classify, collapse, fmt_for, tables


def test_fmt_for_picks_formats_from_headers():
    assert fmt_for("CAGR", "normal") == "0.0%"
    assert fmt_for("Max drawdown", "normal") == "0.0%"
    assert fmt_for("Sharpe (excess of cash)", "normal") == "0.000"
    assert fmt_for("QLIKE (all)", "normal") == "0.0000"
    assert fmt_for("Final value of $10,000", "normal") == "$#,##0"
    assert fmt_for("Anything", "returns") == "0.0%"
    assert fmt_for("Fund", "normal") is None


def test_classify_and_tables():
    rows = [["Title"], ["A long note sentence that ends with a period."], [""], ["Section"],
            ["Strategy", "CAGR"], ["SPY", 0.1], ["EW", 0.08], [""], ["Bottom line"], ["Text."]]
    kinds = classify(rows)
    assert kinds == ["title", "note", "blank", "label", "header", "data", "data", "blank", "label", "note"]
    assert tables(kinds) == [(4, 5, 7)]


def test_collapse_merges_adjacent_sizes_only():
    def dim(a, b, px):
        return {"updateDimensionProperties": {"range": {"sheetId": 1, "dimension": "ROWS", "startIndex": a,
                                                        "endIndex": b}, "properties": {"pixelSize": px},
                                              "fields": "pixelSize"}}
    out = collapse([dim(0, 1, 40), dim(1, 2, 40), dim(3, 4, 40), dim(4, 5, 21)])
    assert len(out) == 3 and out[0]["updateDimensionProperties"]["range"]["endIndex"] == 2


def test_front_tabs_build_from_outputs():
    n = load()
    ov, cat = overview(n), catalogue(n)
    assert ov[0][0].startswith("Forecasting market volatility")
    assert len(cat) == 18 and cat[3][0] == "Phase"
    assert all(len(r) == 6 for r in cat[4:16])
