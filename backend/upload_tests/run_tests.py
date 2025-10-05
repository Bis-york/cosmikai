import sys
import os
from pathlib import Path

def main():
    """Run the test suite."""
    print("Combined Exoplanet Analyzer Test Runner")
    print("=" * 40)

    # Check if test files exist
    test_files = ["test_combined_exoplanet_analyzer.py", "test_lightcurve_data.csv", "test_exoplanet_parameters.csv"]

    missing_files = []
    for file in test_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"Missing files: {missing_files}")
        print("Please ensure all test files are in the current directory.")
        return 1

    # Try to import and run tests
    try:
        from test_combined_exoplanet_analyzer import run_manual_test
        run_manual_test()
        return 0
    except Exception as e:
        print(f"Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
