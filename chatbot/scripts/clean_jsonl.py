import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _loads_first_json_obj(line: str) -> Optional[Dict[str, Any]]:
    s = (line or "").strip()
    if not s:
        return None
    start = s.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[start:])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def main() -> int:
    p = argparse.ArgumentParser(description="Clean JSONL by removing error records and invalid lines")
    p.add_argument("--in", dest="in_path", required=True, help="Input JSONL path")
    p.add_argument("--out", dest="out_path", required=True, help="Output JSONL path (data-only)")
    p.add_argument(
        "--errors-out",
        dest="errors_out",
        default=None,
        help="Optional path to write error records",
    )
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    err_path = Path(args.errors_out) if args.errors_out else None
    if err_path:
        err_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped_errors = 0
    dropped_invalid = 0

    with in_path.open("r", encoding="utf-8") as rf, out_path.open("w", encoding="utf-8") as wf:
        ef = err_path.open("w", encoding="utf-8") if err_path else None
        try:
            for line in rf:
                rec = _loads_first_json_obj(line)
                if not rec:
                    dropped_invalid += 1
                    continue
                if "data" in rec:
                    wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    kept += 1
                    continue
                if "error" in rec:
                    dropped_errors += 1
                    if ef is not None:
                        ef.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    continue
                dropped_invalid += 1
        finally:
            if ef is not None:
                ef.close()

    print(
        json.dumps(
            {
                "input": str(in_path),
                "output": str(out_path),
                "kept_data_records": kept,
                "dropped_error_records": dropped_errors,
                "dropped_invalid_lines": dropped_invalid,
                "errors_output": str(err_path) if err_path else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
