import csv
import ast
from pathlib import Path

INPUT_CSV = "/home/huch135g/cultural-qa-llm/data/train_dataset_saq.csv"
OUTPUT_CSV = "/home/huch135g/cultural-qa-llm/data/train_dataset_saq_expanded.csv"

def expand_annotations_en_repeat_count(row):
    """
    For each annotation dict:
      - for each en_answer in en_answers:
          repeat the row exactly `count` times
      - keep same columns; replace annotations with {"en_answer": <single answer>}
    """
    annotations = ast.literal_eval(row["annotations"])
    expanded_rows = []
    row_counter = 0

    for ann in annotations:
        en_answers = ann.get("en_answers", []) or []
        count = int(ann.get("count", 0) or 0)

        if count <= 0 or not en_answers:
            continue

        for en_answer in en_answers:
            for _ in range(count):
                row_counter += 1
                new_row = dict(row)
                new_row["ID"] = f"{row['ID']}-{row_counter}"
                new_row["annotations"] = str({"en_answer": en_answer})
                expanded_rows.append(new_row)

    return expanded_rows


def main():
    input_path = Path(INPUT_CSV)
    output_path = Path(OUTPUT_CSV)

    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        all_rows = []
        for row in reader:
            all_rows.extend(expand_annotations_en_repeat_count(row))

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Generated {len(all_rows)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()