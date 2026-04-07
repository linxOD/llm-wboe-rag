import json
import glob
import os
from typing import Generator as Gen


INPUT = "output"
ENCODING = "utf-8"

def load_files() -> list[str]:
    files = glob.glob(f"{INPUT}/*.json")
    return files


def parse_json_file(file_path) -> dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def save_parsed_data(data, output_file) -> None:
    with open(output_file, 'w', encoding=ENCODING) as file:
        json.dump(data, file, indent=4)
    print(f"Parsed data saved to {output_file}")
    print(f"Total items parsed: {len(data)}")


def parse_llm_response() -> Gen[str, dict, bool]:
    files = load_files()
    valid = False

    for file in files:
        file_name = os.path.basename(file)
        if "conversation_history" in file_name:
            continue  # Skip chat history file

        fn = os.path.join(f"{INPUT}_parsed", file_name)
        if not os.path.exists(f"{INPUT}_parsed"):
            os.makedirs(f"{INPUT}_parsed")

        data = parse_json_file(file)
        if isinstance(data, dict):
            # response = data.get("choices", {})[0].get("message", {}
            #                                           ).get("content", "")
            response = data.get("content", "")
            # If response is a string containing JSON data and you want
            # to parse it:
            try:
                response = json.loads(response)
                print(f"Parsing file: {file_name}")
                valid = True
            except (json.JSONDecodeError, TypeError):
                print(f"Invalid JSON in file: {file_name}, storing as string.")
                # If it's not valid JSON, just store the string
                response = response

            # save_parsed_data(response, fn)

            yield fn, response, valid

        else:
            response = {"error": "Invalid JSON format"}
            yield fn, response, valid


def verify_parsed_data(parsed_data) -> str:
    """
    Recursively verifies and formats parsed data into a string representation.
    Handles dictionaries, lists, and strings appropriately.

    Args:
        parsed_data: The data structure to process (dict, list, or str)

    Returns:
        str: Formatted string representation of the data
    """
    result = ""

    if isinstance(parsed_data, dict):
        for key, value in parsed_data.items():
            if isinstance(value, str):
                result += f"{key}: {value}\n"
            else:
                result += f"{key}:\n"
                result += verify_parsed_data(value)

    elif isinstance(parsed_data, list):
        for item in parsed_data:
            if isinstance(item, str):
                result += f"- {item}\n"
            else:
                nested_result = verify_parsed_data(item)

                indented = "\n".join(
                    f"  {line}" for line in nested_result.splitlines())
                result += f"- \n{indented}\n"

    elif isinstance(parsed_data, str):
        result += f"{parsed_data}\n"

    else:
        result += f"{str(parsed_data)}\n"

    return result


def create_text_from_parsed_data(fn, parsed_data) -> str:
    parse_json = False
    if isinstance(parsed_data, str):
        text = parsed_data
    else:
        text = parsed_data
        parse_json = True

    if not parse_json:
        text = text.strip()
        fn = fn.replace(".json", ".md")
        with open(fn, "w", encoding=ENCODING) as text_file:
            text_file.write(text)
    else:
        with open(fn, "w", encoding=ENCODING) as text_file:
            json.dump(text, text_file, indent=4)
    return text


if __name__ == "__main__":
    for file_name, parsed_data, valid in parse_llm_response():
        if not valid:
            print(f"Skipping invalid file: {file_name}")
            continue
        text = create_text_from_parsed_data(file_name, parsed_data)
    # print(f"Total items parsed: {len(parsed_data)}")
