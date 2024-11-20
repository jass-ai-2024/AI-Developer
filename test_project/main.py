import argparse
from program import Program


def process_file(input_file, output_file):
    program = Program(file_path=input_file)
    data_json = program.export_declarations_to_json(output_file)
    return data_json


def main():
    parser = argparse.ArgumentParser(description="Process Kotlin files to extract declarations.")

    parser.add_argument("input_file", type=str, help="Path to the Kotlin file to be processed.")
    parser.add_argument("output_file", type=str, help="Path where the JSON output will be saved.")

    args = parser.parse_args()

    result = process_file(args.input_file, args.output_file)

    print(f"Output saved to {args.output_file}. JSON content: {result}")


if __name__ == "__main__":
    main()
