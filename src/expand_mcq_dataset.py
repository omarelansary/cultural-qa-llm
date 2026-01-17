#!/usr/bin/env python3
"""
MCQ Expansion Utility (UK/US/Iran/China only)

This script expands MCQ rows across exactly four target countries: UK, US, Iran, China.

Core rules (as requested):
1) Fingerprint is (ID, country). We must not create duplicates for the same (ID, country).
2) A row can be expanded to a target country C2 only if C2 appears in `choice_countries` values.
3) Expansion only targets the four countries {UK, US, Iran, China}.
4) Prompt text replacement:
   - US -> "the US"
   - UK -> "the UK"
   - Iran -> "Iran"
   - China -> "China"
   Replacement happens by swapping the base country phrase with the target country phrase.
5) When expanding to C2:
   - MCQID is suffixed deterministically (default: "__EXP_<C2>")
   - `country` is set to C2
   - `answer_idx` becomes the option letter whose `choice_countries[letter] == C2`
6) We do NOT "expand" to the same country as the base row (no self-expansion).

Input columns expected:
MCQID, ID, country, prompt, choices, choice_countries, answer_idx

Example behavior:
- If base row is (ID=Na-ko-30, country=US) and choice_countries contains "China" at "B",
  then we create (ID=Na-ko-30, country=China) with answer_idx="B" and prompt "... in China? ...".
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd


TARGET_COUNTRIES: Set[str] = {"UK", "US", "Iran", "China"}

# Exact phrases expected inside the prompt text.
PROMPT_PHRASE: Dict[str, str] = {
    "US": "the US",
    "UK": "the UK",
    "Iran": "Iran",
    "China": "China",
}


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    p = argparse.ArgumentParser(description="Expand MCQ dataset across UK/US/Iran/China without duplicates.")
    p.add_argument("--in_csv", required=True, help="Path to input MCQ CSV.")
    p.add_argument("--out_csv", required=True, help="Path to output MCQ CSV (expanded).")
    p.add_argument(
        "--mcqid_suffix_fmt",
        default="__EXP_{country}",
        help='Suffix format appended to MCQID when expanding (default: "__EXP_{country}").',
    )
    p.add_argument(
        "--keep_only_target_base_rows",
        action="store_true",
        help="If set, drop base rows whose country is not in {UK,US,Iran,China} before expansion.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, do not write output; only print summary.",
    )
    return p.parse_args()


def _normalize_csv_jsonish(s: str) -> str:
    """
    Normalize a CSV-escaped JSON string.

    Your CSV often contains JSON-like strings with doubled quotes:
      {""A"": ""US"", ...}
    This converts them to valid JSON:
      {"A": "US", ...}

    Args:
        s (str): Input string.

    Returns:
        str: Normalized JSON string.
    """
    s = str(s).strip()
    if '""' in s:
        s = s.replace('""', '"')
    return s


def parse_choice_countries(value) -> Dict[str, str]:
    """
    Parse the `choice_countries` cell into a dict mapping letters -> country labels.

    Args:
        value: The raw cell value (string or dict-like).

    Returns:
        Dict[str, str]: Parsed mapping, e.g. {"A": "US", "B": "China", ...}.
                        Returns {} if parsing fails.
    """
    if isinstance(value, dict):
        return {str(k).strip(): str(v).strip() for k, v in value.items()}

    if value is None:
        return {}

    s = str(value).strip()
    if not s:
        return {}

    try:
        s = _normalize_csv_jsonish(s)
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return {}
        return {str(k).strip(): str(v).strip() for k, v in obj.items()}
    except Exception:
        return {}


def find_answer_letter_for_country(choice_countries: Dict[str, str], country_code: str) -> str:
    """
    Find the option letter whose mapped country equals `country_code`.

    Args:
        choice_countries (Dict[str, str]): Mapping like {"A":"US","B":"China",...}
        country_code (str): One of {"UK","US","Iran","China"}.

    Returns:
        str: The option letter (e.g., "B") if found, otherwise "".
    """
    for letter, cc in choice_countries.items():
        if str(cc).strip() == country_code:
            return str(letter).strip().upper()
    return ""


def replace_prompt_country(prompt: str, from_country: str, to_country: str) -> str:
    """
    Replace the country phrase in the prompt text from `from_country` to `to_country`.

    Exact mapping:
      US   -> "the US"
      UK   -> "the UK"
      Iran -> "Iran"
      China-> "China"

    This assumes the base prompt contains the `from_country` phrase at least once.
    Only the first occurrence is replaced to avoid unintended modifications.

    Args:
        prompt (str): The original prompt text.
        from_country (str): Country code in the row's country column.
        to_country (str): Target country code.

    Returns:
        str: Updated prompt text.
    """
    s = str(prompt)
    frm = PROMPT_PHRASE[from_country]
    to = PROMPT_PHRASE[to_country]

    # Word boundary replacement; safe for "Iran"/"China" and ok for "the US"/"the UK".
    pattern = r"\b" + re.escape(frm) + r"\b"
    return re.sub(pattern, to, s, count=1)


def expand_mcq_rows(
    df: pd.DataFrame,
    target_countries: Set[str] = TARGET_COUNTRIES,
    mcqid_suffix_fmt: str = "__EXP_{country}",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expand MCQ rows across the target countries using `choice_countries`, avoiding duplicates.

    Expansion rule:
      For each base row r:
        - Let base_fp = (r["ID"], r["country"])
        - Parse r["choice_countries"] into a mapping letter->country_name
        - Candidate expansions are: (values ∩ target_countries) minus {base_country}
        - For each candidate C2:
            if (ID, C2) not already present in dataset:
              - Duplicate row
              - MCQID = f'{base_MCQID}{suffix}' (suffix from mcqid_suffix_fmt)
              - country = C2
              - prompt: replace base country phrase with C2 phrase
              - answer_idx: letter where choice_countries[letter] == C2

    Args:
        df (pd.DataFrame): Input MCQ dataframe.
        target_countries (Set[str]): Exactly the countries to consider for expansion.
        mcqid_suffix_fmt (str): Format for suffix appended to MCQID; must include "{country}".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
          - expanded_df: original rows + newly created rows
          - added_rows_df: only the newly created rows
    """
    required_cols = {"MCQID", "ID", "country", "prompt", "choice_countries", "answer_idx"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    base_df = df.copy()

    # Track existing fingerprints (ID, country) to prevent duplicates.
    existing: Set[Tuple[str, str]] = set(
        zip(base_df["ID"].astype(str), base_df["country"].astype(str))
    )

    new_rows: List[pd.Series] = []

    for _, r in base_df.iterrows():
        base_id = str(r["ID"])
        base_country = str(r["country"])

        # Only expand from rows that are already one of the four target countries.
        if base_country not in target_countries:
            continue

        cc_map = parse_choice_countries(r.get("choice_countries", ""))

        # Countries present in choices (restricted to target set), excluding self.
        present_targets = {v for v in cc_map.values() if v in target_countries}
        candidates = present_targets - {base_country}

        for c2 in sorted(candidates):
            fp = (base_id, c2)
            if fp in existing:
                continue  # already have this variant

            # Determine correct answer letter for c2
            ans_letter = find_answer_letter_for_country(cc_map, c2)
            if not ans_letter:
                continue  # cannot create supervised label

            # Duplicate row with required changes
            new_r = r.copy()
            new_r["MCQID"] = f"{str(r['MCQID'])}{mcqid_suffix_fmt.format(country=c2)}"
            new_r["country"] = c2
            new_r["answer_idx"] = ans_letter

            # Prompt rewrite
            new_r["prompt"] = replace_prompt_country(str(r["prompt"]), base_country, c2)

            new_rows.append(new_r)
            existing.add(fp)

    added_df = pd.DataFrame(new_rows) if new_rows else base_df.iloc[0:0].copy()
    expanded_df = pd.concat([base_df, added_df], ignore_index=True) if len(added_df) else base_df

    return expanded_df, added_df


def main() -> None:
    """
    Main entry point: load input CSV, expand rows, write output CSV, print summary.
    """
    args = parse_args()

    df = pd.read_csv(args.in_csv)

    if args.keep_only_target_base_rows:
        df = df[df["country"].isin(TARGET_COUNTRIES)].copy()

    expanded, added = expand_mcq_rows(
        df=df,
        target_countries=TARGET_COUNTRIES,
        mcqid_suffix_fmt=args.mcqid_suffix_fmt,
    )

    print(f"Input rows: {len(df)}")
    print(f"Added rows: {len(added)}")
    print(f"Output rows: {len(expanded)}")

    # Optional quick sanity checks:
    # 1) No duplicate (ID, country)
    fp = list(zip(expanded["ID"].astype(str), expanded["country"].astype(str)))
    if len(fp) != len(set(fp)):
        raise RuntimeError("Duplicate (ID, country) found after expansion. Logic bug or input has duplicates.")

    # 2) Added rows are only for target countries
    if len(added) and not set(added["country"].astype(str)).issubset(TARGET_COUNTRIES):
        raise RuntimeError("Added rows include non-target countries, which should be impossible.")

    if not args.dry_run:
        expanded.to_csv(args.out_csv, index=False)
        print(f"Wrote expanded CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()
