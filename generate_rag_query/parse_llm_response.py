import json
import glob
import os


def load_files():
    files = glob.glob("output/*.json")
    return files


def parse_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def save_parsed_data(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Parsed data saved to {output_file}")
    print(f"Total items parsed: {len(data)}")


def parse_llm_response():
    files = load_files()
    all_data = {}

    for file in files:
        file_name = os.path.basename(file)
        if file_name == "chat_history.json":
            continue  # Skip chat history file

        data = parse_json_file(file)
        if isinstance(data, dict):
            response = data.get("choices", {})[0].get("message", {}
                                                      ).get("content", "")
            # If response is a string containing JSON data and you want
            # to parse it:
            try:
                all_data[file_name] = json.loads(response)
            except json.JSONDecodeError:
                # If it's not valid JSON, just store the string
                all_data[file_name] = response

    save_parsed_data(all_data, "parsed_llm_response.json")

    return all_data


def verify_parsed_data(parsed_data):
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

                indented = "\n".join(f"  {line}" for line in nested_result.splitlines())
                result += f"- \n{indented}\n"

    elif isinstance(parsed_data, str):
        result += f"{parsed_data}\n"

    else:
        result += f"{str(parsed_data)}\n"

    return result


def create_text_from_parsed_data(parsed_data):
    text = ""
    for file_name, content in parsed_data.items():
        text += f"File: {file_name}\n"
        for key, value in content.items():
            text += f"{key}:\n"
            text += verify_parsed_data(value)
            text += "\n"
        text += "\n\n"

    text = text.strip()
    with open("parsed_text.txt", "w") as text_file:
        text_file.write(text)
    return text


if __name__ == "__main__":
    parsed_data = parse_llm_response()
    create_text_from_parsed_data(parsed_data)
    print(f"Total items parsed: {len(parsed_data)}")
