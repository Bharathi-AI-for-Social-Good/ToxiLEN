import yaml
from jinja2 import Template
from pathlib import Path

def load_prompt_template(file_path: str, variables: dict) -> list:
    """
    Load a YAML prompt file and render it with the provided variables.
    Args:
        file_path (str): Path to the YAML file.
        variables (dict): Variables to render the template.
        Returns:
            list: A list of rendered messages.
    """
    template_path = Path(file_path)
    assert template_path.exists(), f"Template file {file_path} does not exist."
    
    with open(template_path, "r", encoding="utf-8") as file:
        messages = yaml.safe_load(file)
    
    
    for msg in messages:
        template = Template(msg["content"])
        msg["content"] = template.render(**variables)
    
    return messages

def load_prompt_by_name(prompt_name: str, variables: dict, prompt_dir="app/prompts") -> list:
    """
    Load a prompt by its name and render it with the provided variables.
    Args:
        prompt_name (str): Name of the prompt file without extension.
        variables (dict): Variables to render the template.
        Returns:
            list: A list of rendered messages.
    """
    path= Path(prompt_dir) / f"{prompt_name}.yaml"
    return load_prompt_template(path, variables)