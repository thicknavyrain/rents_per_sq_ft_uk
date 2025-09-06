import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    #!/usr/bin/env python3
    import os
    import re
    import csv
    import json
    from collections import defaultdict
    from typing import Dict, List, Tuple, Optional

    from fuzzywuzzy import fuzz, process

    # --------- USER SETTINGS ----------
    EPC_PARENT_DIR = "EPC_data"  # folder containing all 'domestic-<code>-<name>/' dirs
    AREA_CSV = "LA_bedroom_properties/flat_maisonette_bedrooms.csv"  # input CSV with an 'Area Name' column

    # Output files
    OUT_JSON = "area_to_epc_dir.json"
    OUT_CSV_REPORT = "area_to_epc_dir_report.csv"

    # Optional: force specific mappings here if you already know the exact folder name.
    # Keys are EXACT 'Area Name' strings from the CSV; values are directory names (as they appear under EPC_PARENT_DIR).
    MANUAL_OVERRIDES: Dict[str, str] = {
        # Example:
        # "Kingston upon Hull, City of": "domestic-E06000010-Kingston-upon-Hull-City-of",
        # "Bristol UA": "domestic-E06000023-Bristol,-City-of",
    }

    # Matching confidence threshold (0–100).
    # Anything below this will be flagged in the report under 'needs_review'.
    CONFIDENCE_THRESHOLD = 88
    # ----------------------------------


    DIR_PREFIX_RE = re.compile(r"^domestic-[A-Z0-9]+-", flags=re.IGNORECASE)

    def clean_text(s: str) -> str:
        """
        Normalise an area or directory 'name' for robust matching.
        - lower, strip
        - remove codes/prefixes
        - replace hyphens/underscores with spaces
        - normalise ampersands and 'and'
        - drop brackets text like '(met county)'
        - remove ' ua', ' district', ' city of', trailing commas
        - collapse whitespace
        - normalise punctuation variants of apostrophes/commas
        """
        if s is None:
            return ""
        s0 = s.strip()

        # Remove the 'domestic-<code>-' bit if present (dir names)
        s1 = DIR_PREFIX_RE.sub("", s0)

        # Normalise punctuation
        s1 = s1.replace("’", "'").replace("‘", "'").replace("`", "'")
        s1 = s1.replace("–", "-").replace("—", "-").replace("-", "-")

        # Remove bracketed qualifiers like '(Met County)'
        s1 = re.sub(r"\([^)]*\)", " ", s1)

        # Replace separators with spaces
        s1 = s1.replace("-", " ").replace("_", " ").replace("/", " ")
        s1 = s1.replace(",", " ")

        # Normalise “&” and the word “and”
        s1 = re.sub(r"\b&\b", " and ", s1, flags=re.IGNORECASE)

        # Lowercase for comparison
        s1 = s1.lower()

        # Canonicalise some common phrases
        replacements = [
            (r"\bua\b", " "),
            (r"\bdistrict\b", " "),
            (r"\bcity council\b", " "),
            (r"\bcounty council\b", " "),
            (r"\bmetropolitan borough\b", " "),
            (r"\bmetropolitan district\b", " "),
            (r"\broyal borough\b", " "),
            (r"\blondon borough of\b", " "),
            (r"\bborough council\b", " "),
            (r"\bcity of\b", " "),
            (r"\bupon\b", " upon "),  # preserve “upon” but space it consistently
            (r"\s+", " "),  # collapse spaces
        ]
        for pat, repl in replacements:
            s1 = re.sub(pat, repl, s1, flags=re.IGNORECASE).strip()

        # Special canonicalisations that help token-based matching
        s1 = s1.replace(" on tees", " on tees")
        s1 = s1.replace(" upon tyne", " upon tyne")

        # Final trim and single spaces
        s1 = re.sub(r"\s+", " ", s1).strip()
        return s1


    def read_area_names(csv_path: str) -> List[str]:
        areas = []
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if "Area Name" not in reader.fieldnames:
                raise ValueError("CSV must have an 'Area Name' column.")
            for row in reader:
                name = (row.get("Area Name") or "").strip()
                if name:
                    areas.append(name)
        return areas


    def list_epc_dirs(parent: str) -> List[str]:
        if not os.path.isdir(parent):
            raise FileNotFoundError(f"EPC parent directory not found: {parent}")
        entries = []
        for d in os.listdir(parent):
            full = os.path.join(parent, d)
            if os.path.isdir(full) and d.lower().startswith("domestic-"):
                entries.append(d)
        if not entries:
            raise RuntimeError(f"No 'domestic-*' directories found in: {parent}")
        return sorted(entries)


    def build_clean_dir_index(epc_dirs: List[str]) -> Dict[str, str]:
        """
        Map cleaned name -> original directory name.
        If collisions occur (two dirs clean to the same text), keep the longer directory name
        (heuristic that often reflects more specific naming).
        """
        idx: Dict[str, str] = {}
        for d in epc_dirs:
            cleaned = clean_text(d)
            prev = idx.get(cleaned)
            if prev is None or len(d) > len(prev):
                idx[cleaned] = d
        return idx


    def best_match(
        query_clean: str, clean_dir_index: Dict[str, str]
    ) -> Tuple[str, int, str]:
        """
        Return (matched_dirname, score, matched_clean_key) for the best match.
        """
        choices = list(clean_dir_index.keys())
        # token_set_ratio is robust to extra/missing tokens; token_sort_ratio helps ordering differences.
        # We'll combine two scorers by averaging.
        def scorer(a: str, b: str) -> int:
            return int((fuzz.token_set_ratio(a, b) + fuzz.token_sort_ratio(a, b)) / 2)

        match_key, score = process.extractOne(query_clean, choices, scorer=scorer)
        return clean_dir_index[match_key], score, match_key


    def main():
        # Load inputs
        area_names = read_area_names(AREA_CSV)
        epc_dirs = list_epc_dirs(EPC_PARENT_DIR)
        clean_dir_index = build_clean_dir_index(epc_dirs)

        # Collect all claims (overrides + fuzzy) BEFORE deciding winners/conflicts
        # Each claim: (area, dir_name, score, clean_area, matched_key, is_override)
        claims: List[Tuple[str, str, int, str, str, bool]] = []

        for area in area_names:
            if area in MANUAL_OVERRIDES:
                dir_name = MANUAL_OVERRIDES[area]
                if dir_name not in epc_dirs:
                    raise ValueError(f"Manual override points to a non-existent directory: {dir_name}")
                claims.append((area, dir_name, 999, clean_text(area), clean_text(dir_name), True))
            else:
                clean_area = clean_text(area)
                matched_dir, score, matched_key = best_match(clean_area, clean_dir_index)
                claims.append((area, matched_dir, score, clean_area, matched_key, False))

        # Group claims by directory to detect conflicts
        claims_by_dir: Dict[str, List[Tuple[str, int, str, str, bool]]] = defaultdict(list)
        for area, dir_name, score, clean_area, matched_key, is_override in claims:
            claims_by_dir[dir_name].append((area, score, clean_area, matched_key, is_override))

        # Determine winners per directory (highest score; overrides score as 999)
        winners_by_dir: Dict[str, str] = {}
        for dir_name, claim_list in claims_by_dir.items():
            best = max(claim_list, key=lambda t: t[1])  # by score
            winners_by_dir[dir_name] = best[0]  # area name

        # Any directory with >1 claimant is a conflict; EXCLUDE ALL its claimants from final JSON
        conflicted_dirs = {d for d, claim_list in claims_by_dir.items() if len(claim_list) > 1}

        # Build mapping and report
        mapping: Dict[str, Optional[str]] = {}
        report_rows: List[Dict[str, str]] = []

        for area, dir_name, score, clean_area, matched_key, is_override in claims:
            is_conflict = dir_name in conflicted_dirs
            winner_area = winners_by_dir.get(dir_name)
            # Keep low-confidence if NOT a conflict; exclude all conflicts (even the winner)
            if not is_conflict:
                mapping[area] = dir_name
            else:
                mapping[area] = None

            needs_review = "yes" if (is_conflict or (not is_override and score < CONFIDENCE_THRESHOLD)) else "no"
            if is_conflict:
                matched_display = f"(conflict -> winner: {winner_area})"
                note = "conflict"
            else:
                matched_display = dir_name
                note = "manual_override" if is_override else ("low_confidence" if score < CONFIDENCE_THRESHOLD else "")

            report_rows.append({
                "area_name": area,
                "matched_directory": matched_display,
                "score": "override" if is_override else str(score),
                "clean_area": clean_area,
                "clean_directory_key": matched_key,
                "needs_review": needs_review,
                "note": note
            })

        # Save JSON mapping (exclude conflicts and nulls)
        final_mapping = {k: v for k, v in mapping.items() if v is not None}
        with open(OUT_JSON, "w", encoding="utf-8") as jf:
            json.dump(final_mapping, jf, ensure_ascii=False, indent=2)

        # Save CSV report
        fieldnames = ["area_name", "matched_directory", "score", "clean_area", "clean_directory_key", "needs_review", "note"]
        with open(OUT_CSV_REPORT, "w", newline="", encoding="utf-8") as rf:
            writer = csv.DictWriter(rf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_rows)

        # Quick summary to stdout
        total = len(area_names)
        matched = sum(1 for v in mapping.values() if v is not None)
        needs_review_count = sum(1 for r in report_rows if r["needs_review"] == "yes")
        conflicts = sum(1 for r in report_rows if r["note"] == "conflict")

        print(f"Total areas: {total}")
        print(f"Mapped (non-null, conflicts excluded): {matched}")
        print(f"Flagged for review (low confidence or conflict): {needs_review_count}")
        print(f"Conflicts detected (area-claims): {conflicts}")
        print(f"Saved mapping -> {OUT_JSON}")
        print(f"Saved audit report -> {OUT_CSV_REPORT}")


    if __name__ == "__main__":
        main()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
