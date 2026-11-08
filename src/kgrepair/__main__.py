import argparse
import sys

from .config import load_config, rpath


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kgrepair", description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-schema", help="derive allowed type pairs from TRAIN and freeze schema/schema.json")
    sub.add_parser("freeze-test", help="write a checksum manifest locking the final TEST configuration")
    e = sub.add_parser("extract", help="run the extractor LLM once per document (resumable)")
    e.add_argument("--split", default="train", choices=["train", "dev", "test"])
    e.add_argument("--n", default="10", help="number of documents or 'all'")
    e.add_argument("--start", type=int, default=0)
    e.add_argument("--out", required=True)
    pr = sub.add_parser("pilot-report", help="format/quote/raw-metric statistics for a raw file")
    pr.add_argument("--raw", required=True)
    pr.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ev = sub.add_parser("evaluate", help="run C/S/P combinations and automatic metrics")
    ev.add_argument("--raw", required=True)
    ev.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ev.add_argument("--configs", default="all", choices=["all", "main"])
    ev.add_argument("--no-judge", action="store_true", help="run P with quote validation only")
    ev.add_argument("--out-dir", default=None, help="override the results directory")
    sa = sub.add_parser("sample-annotation", help="draw the paired manual-annotation sample")
    sa.add_argument("--raw", required=True)
    sa.add_argument("--split", default="dev", choices=["train", "dev"])
    sa.add_argument("--n", type=int, default=None, help="override annotation sample size")
    sa.add_argument("--overlap-frac", type=float, default=None,
                    help="override the share double-labelled by annotator 2 (0..1)")
    sa.add_argument("--no-judge", action="store_true", help="run P with quote validation only")
    sa.add_argument("--out-dir", default=None, help="override annotation output directory")
    sa.add_argument("--exclude-key", default=None, help="exclude raw IDs sampled in another key.json")
    ps = sub.add_parser("sample-proposal-annotation",
                        help="sample proposal-aligned unique accepted edges from five-seed test runs")
    ps.add_argument("--raw", required=True)
    ps.add_argument("--out-dir", default="annotation/proposal_test")
    ps.add_argument("--n-per-stage", type=int, default=None)
    ps.add_argument("--overlap-frac", type=float, default=None)
    ps.add_argument("--no-judge", action="store_true", help="use quote-rule P only")
    sc = sub.add_parser("score-annotation", help="manual precision, retention, kappa from labelled CSVs")
    sc.add_argument("--labels", default=None)
    sc.add_argument("--labels2", default=None)
    sc.add_argument("--labels3", default=None)
    sp = sub.add_parser("score-proposal-annotation",
                        help="score proposal's stage-balanced two-annotator test sample")
    sp.add_argument("--labels", default="annotation/proposal_test/annotator1.csv")
    sp.add_argument("--labels2", default="annotation/proposal_test/annotator2.csv")
    sp.add_argument("--sample-dir", default="annotation/proposal_test")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)

    if a.cmd == "build-schema":
        from .data import load_split
        from .schema import build_schema, save_schema
        schema = build_schema(cfg, load_split(rpath(cfg, cfg["paths"]["train"])))
        save_schema(schema, rpath(cfg, cfg["paths"]["schema"]))
        for rid, r in schema["relations"].items():
            print(f"{rid:6} {r['name']:28} train={r['train_count']:6d} allowed={r['allowed_pairs']}")
    elif a.cmd == "freeze-test":
        from .freeze import freeze_test_configuration
        print(f"Frozen TEST setup: {freeze_test_configuration(cfg)}")
    elif a.cmd == "extract":
        from .extract import run_extract
        run_extract(cfg, a.split, None if a.n == "all" else int(a.n), a.start, rpath(cfg, a.out))
    elif a.cmd == "pilot-report":
        from .report import pilot_report
        pilot_report(cfg, rpath(cfg, a.raw), a.split)
    elif a.cmd == "evaluate":
        from .evaluate import evaluate_all, print_primary
        out_dir = rpath(cfg, a.out_dir) if a.out_dir else None
        rows, _ = evaluate_all(cfg, rpath(cfg, a.raw), a.split, a.configs,
                               use_judge=not a.no_judge, results_dir=out_dir)
        print_primary(cfg, rows)
    elif a.cmd == "sample-annotation":
        from .annotate import sample_annotation
        sample_annotation(cfg, rpath(cfg, a.raw), a.split, n_sample=a.n, overlap_frac=a.overlap_frac,
                          use_judge=not a.no_judge,
                          out_dir=rpath(cfg, a.out_dir) if a.out_dir else None,
                          exclude_key=rpath(cfg, a.exclude_key) if a.exclude_key else None)
    elif a.cmd == "sample-proposal-annotation":
        from .annotate import sample_proposal_annotation
        sample_proposal_annotation(cfg, rpath(cfg, a.raw), split="test",
                                   n_per_stage=a.n_per_stage, overlap_frac=a.overlap_frac,
                                   use_judge=not a.no_judge, out_dir=rpath(cfg, a.out_dir))
    elif a.cmd == "score-annotation":
        from .annotate import score_annotation
        d = rpath(cfg, cfg["paths"]["annotation_dir"])
        score_annotation(cfg, a.labels or d / "annotator1.csv", a.labels2, a.labels3)
    elif a.cmd == "score-proposal-annotation":
        from .annotate import score_proposal_annotation
        score_proposal_annotation(cfg, rpath(cfg, a.sample_dir), rpath(cfg, a.labels),
                                  rpath(cfg, a.labels2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
