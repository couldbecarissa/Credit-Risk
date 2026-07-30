"""Single-command, end-to-end demo: trains the PD model on the bundled
50k-row sample, runs the Benford's Law screen, evaluates the model, and
saves plots + a run report to reports/. No server needed.

Usage:
    python run_demo.py
"""
import train

if __name__ == "__main__":
    print("=" * 60)
    print("Credit Risk PD Model -- End-to-End Demo")
    print("=" * 60)
    train.main(data_path="data/sample_loans.csv")
    print("\nDemo complete. See reports/ for plots and run_report.json.")
    print("To serve predictions: uvicorn service:app --reload")
