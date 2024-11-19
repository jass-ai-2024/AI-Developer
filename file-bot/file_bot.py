import os
import subprocess
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from key import OPENAI_API_KEY

# Initialize the OpenAI LLM
llm = OpenAI(api_key=OPENAI_API_KEY)

# Prompt template for interpreting user commands
action_prompt = PromptTemplate(
    input_variables=["request"],
    template="""
You are a helpful assistant for a file management bot. Interpret the user's request and map it to one of the following actions:
1) List directory structure: 'list_directory(directory_path)'
2) Create a file: 'create_file(file_name, contents)'
3) Modify file contents: 'modify_file(file_name, diff_description)'
4) Read a file: 'read_file(file_name)'
5) Commit changes: 'commit_changes(commit_message)'

If the user's input doesn't match any action, reply 'invalid_command'.

Request: {request}

Output the action ID and arguments in JSON format. For example:
{{"action_id": 1, "arguments": {{"directory_path": "/path/to/dir"}}}}
""",
)

# Prompt template for the Diff Bot
diff_prompt = PromptTemplate(
    input_variables=["original", "diff_description"],
    template="""
You are a helpful assistant for modifying file contents. Given the original file contents and a description of the changes, provide the updated file contents.

Original contents:
{original}

Diff description:
{diff_description}

Updated contents:
""",
)

action_chain = LLMChain(llm=llm, prompt=action_prompt)
diff_chain = LLMChain(llm=llm, prompt=diff_prompt)


def run_command_with_confirmation(command):
    """Run a command after user confirmation."""
    print(f"Suggested command:\n{command}")
    approval = input("Do you want to execute this command? (yes/no): ")
    if approval.lower() == "yes":
        try:
            result = subprocess.check_output(
                command, shell=True, stderr=subprocess.STDOUT)
            print(result.decode())
        except subprocess.CalledProcessError as e:
            print(f"Error executing command:\n{e.output.decode()}")
    else:
        print("Command not executed.")


def list_directory(directory_path):
    """List the directory structure of a given directory."""
    for root, dirs, files in os.walk(directory_path):
        print(f"Root: {root}")
        print(f"Directories: {dirs}")
        print(f"Files: {files}")


def create_file(file_name, contents):
    """Create a file with the given name and contents."""
    with open(file_name, "w") as file:
        file.write(contents)
    print(f"File '{file_name}' created.")


def modify_file(file_name, diff_description):
    """Modify file contents by launching the Diff Bot."""
    try:
        with open(file_name, "r") as file:
            original_contents = file.read()
        print("Launching Diff Bot...")
        updated_contents = diff_chain.run(
            {"original": original_contents, "diff_description": diff_description})
        print(f"Updated contents:\n{updated_contents}")

        approval = input(
            "Do you want to save these changes to the file? (yes/no): ")
        if approval.lower() == "yes":
            with open(file_name, "w") as file:
                file.write(updated_contents)
            print(f"File '{file_name}' updated successfully.")
        else:
            print("Changes discarded.")
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")


def read_file(file_name):
    """Read and print the contents of a file."""
    try:
        with open(file_name, "r") as file:
            contents = file.read()
        print(contents)
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")


def commit_changes(commit_message):
    """Stage and commit changes using git."""
    run_command_with_confirmation("git add .")
    run_command_with_confirmation(f"git commit -m '{commit_message}'")

# Main bot logic


def bot():
    print("Welcome to the Natural Language File Management Bot!")
    print("Describe what you want to do in natural language, or type 'exit' to quit.")

    while True:
        request = input("Your request: ")
        if request.lower() == "exit":
            print("Goodbye!")
            break

        try:
            # Interpret the request using ChatGPT
            response = action_chain.run({"request": request})
            print(f"Interpreted response: {response}")

            # Parse the response to identify the action and arguments
            try:
                # Convert JSON string to dictionary
                response_dict = eval(response.strip())
                action_id = response_dict.get("action_id")
                arguments = response_dict.get("arguments", {})

                if action_id is None:
                    raise ValueError("Response is missing 'action_id'.")

            except Exception as e:
                print(f"Error parsing response: {e}")
                print(
                    "The response was invalid. Please rephrase your request or try again.")
                continue

            # Call the appropriate action based on the action ID
            if action_id == 1:
                list_directory(**arguments)
            elif action_id == 2:
                create_file(**arguments)
            elif action_id == 3:
                modify_file(**arguments)
            elif action_id == 4:
                read_file(**arguments)
            elif action_id == 5:
                commit_changes(**arguments)
            else:
                print("Invalid action ID. Please rephrase your request or try again.")

        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            print("Please try again.")


if __name__ == "__main__":
    bot()
