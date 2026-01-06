import argparse
import shutil
import zipfile
from pathlib import Path

from src.utils import validate_saq_submission, validate_mcq_submission

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="submission/latest", help="Directory that contains inference outputs")
    parser.add_argument("--out_zip", default="submission/latest/submission.zip", help="Path to output zip")
    parser.add_argument("--mcq", default="submission_mcq.tsv", help="MCQ inference output filename inside in_dir")
    parser.add_argument("--saq", default="submission_saq.tsv", help="SAQ inference output filename inside in_dir")
    parser.add_argument("--require_both", action="store_true",
                        help="Fail if either MCQ or SAQ file is missing (recommended for final submissions).")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_zip = Path(args.out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    mcq_src = in_dir / args.mcq
    saq_src = in_dir / args.saq

    have_mcq = mcq_src.exists()
    have_saq = saq_src.exists()

    if args.require_both and (not have_mcq or not have_saq):
        missing = []
        if not have_mcq: missing.append(str(mcq_src))
        if not have_saq: missing.append(str(saq_src))
        raise FileNotFoundError(f"Missing required submission files: {missing}")

    staging = in_dir / "_staging_codabench"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    # Validate + copy/rename into Codabench-required names
    if have_mcq:
        validate_mcq_submission(str(mcq_src), expected_id_col="MCQID")
        mcq_dst = staging / "mcq_prediction.tsv"
        shutil.copyfile(mcq_src, mcq_dst)

    if have_saq:
        validate_saq_submission(str(saq_src))
        saq_dst = staging / "saq_prediction.tsv"
        shutil.copyfile(saq_src, saq_dst)

    # Build zip
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if have_mcq:
            z.write(staging / "mcq_prediction.tsv", arcname="mcq_prediction.tsv")
        if have_saq:
            z.write(staging / "saq_prediction.tsv", arcname="saq_prediction.tsv")

    print(f"✅ Built Codabench submission zip: {out_zip}")
    print("Contents:")
    with zipfile.ZipFile(out_zip, "r") as z:
        for name in z.namelist():
            print(" -", name)

    # Optional cleanup
    shutil.rmtree(staging, ignore_errors=True)

if __name__ == "__main__":
    main()

# usage: 
# python -m submission.make_submission --require_both