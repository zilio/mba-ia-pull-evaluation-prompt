"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/bug_to_user_story_v1.yml

SIMPLIFICADO: Usa serialização nativa do LangChain para extrair prompts.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()


PROMPTS_TO_PULL = [
    {
        "hub_name": "leonanluppi/bug_to_user_story_v1",
        "output_path": "prompts/bug_to_user_story_v1.yml",
    }
]


def extract_prompt_data(prompt) -> dict:
    """
    Extrai system_prompt e user_prompt de um ChatPromptTemplate do LangChain.

    Args:
        prompt: ChatPromptTemplate retornado por hub.pull()

    Returns:
        Dicionário com system_prompt e user_prompt
    """
    data = {"system_prompt": "", "user_prompt": ""}

    for message in prompt.messages:
        class_name = message.__class__.__name__
        template = message.prompt.template if hasattr(message, "prompt") else ""

        if "System" in class_name:
            data["system_prompt"] = template
        elif "Human" in class_name or "User" in class_name:
            data["user_prompt"] = template

    return data


def pull_prompts_from_langsmith():
    """Faz pull dos prompts do LangSmith Hub e salva localmente em YAML."""

    required_vars = ["LANGSMITH_API_KEY"]
    if not check_env_vars(required_vars):
        return False

    print_section_header("Pull de Prompts do LangSmith Hub")

    success = True
    for entry in PROMPTS_TO_PULL:
        hub_name = entry["hub_name"]
        output_path = entry["output_path"]
        # Deriva a chave local a partir do nome após "/" (ex: "leonanluppi/foo_v1" → "foo_v1")
        local_key = hub_name.split("/")[-1]

        print(f"Fazendo pull: {hub_name}")
        try:
            prompt = hub.pull(hub_name)
        except Exception as e:
            print(f"  Erro ao fazer pull de '{hub_name}': {e}")
            success = False
            continue

        prompt_data = extract_prompt_data(prompt)

        yaml_data = {
            local_key: {
                "system_prompt": prompt_data["system_prompt"],
                "user_prompt": prompt_data["user_prompt"],
            }
        }

        if save_yaml(yaml_data, output_path):
            print(f"  Salvo em: {output_path}")
        else:
            print(f"  Falha ao salvar '{output_path}'")
            success = False

    return success


def main():
    """Função principal"""
    result = pull_prompts_from_langsmith()
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
