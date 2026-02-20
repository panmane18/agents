import os
from langchain_core.tools import tool

@tool
def create_file(filename: str, content: str) -> str:
    """
    Create a file with given filename and content.
    """
    directory = "agent_files"

    if not os.path.exists(directory):
        os.makedirs(directory)

    path = os.path.join(directory, filename)

    with open(path, "w") as f:
        f.write(content)

    return "File created successfully at " + path