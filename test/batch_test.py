# test_program.py
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test program for HTCondor.")
    parser.add_argument('--learning-rate', type=float, required=True, help="Learning rate")
    parser.add_argument('--batch-size', type=int, required=True, help="Batch size")
    args = parser.parse_args()

    # Simulate some processing
    print(f"Running with learning rate: {args.learning_rate} and batch size: {args.batch_size}")

if __name__ == "__main__":
    main()
