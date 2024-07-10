import argparse

def main():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('name', type=str, help='Name of the user')
    parser.add_argument('--age', type=int, help='Age of the user', default=0)

    args = parser.parse_args()

    print(f"Hello, {args.name}!")
    if args.age:
        print(f"You are {args.age} years old.")

if __name__ == "__main__":
    main()