"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)

SIMPLIFICADO: Código mais limpo e direto ao ponto.
"""

import os
import sys
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header

load_dotenv()


PROMPTS_TO_PUSH = [
    {
        "yaml_path": "prompts/bug_to_user_story_v2.yml",
        "yaml_key": "bug_to_user_story_v2",
        "hub_name": "bug_to_user_story_v2",
    }
]


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt antes do push.

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    errors = []

    system_prompt = prompt_data.get("system_prompt", "").strip()
    if not system_prompt:
        errors.append("system_prompt está vazio")
    elif "TODO" in system_prompt:
        errors.append("system_prompt ainda contém TODOs pendentes")

    if not prompt_data.get("version"):
        errors.append("Campo 'version' ausente")

    techniques = prompt_data.get("techniques_applied", [])
    if len(techniques) < 2:
        errors.append(
            f"Mínimo de 2 técnicas requeridas em 'techniques_applied', "
            f"encontradas: {len(techniques)}"
        )

    return (len(errors) == 0, errors)


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do hub (sem username), ex: "bug_to_user_story_v2"
        prompt_data: Dados do prompt lidos do YAML

    Returns:
        True se sucesso, False caso contrário
    """
    username = os.getenv("USERNAME_LANGSMITH_HUB", "").strip()
    if not username:
        print("  ❌ USERNAME_LANGSMITH_HUB não configurada no .env")
        return False

    full_name = f"{username}/{prompt_name}"

    # Monta o ChatPromptTemplate a partir dos campos do YAML
    system_prompt = prompt_data.get("system_prompt", "")
    user_prompt = prompt_data.get("user_prompt", "{bug_report}")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt),
    ])

    # Monta metadados a partir dos campos do YAML
    description = str(prompt_data.get("description", "")).strip()
    tags = prompt_data.get("tags", [])
    techniques = prompt_data.get("techniques_applied", [])

    # Inclui técnicas nas tags para visibilidade no Hub
    all_tags = list(tags) + [t.lower().replace(" ", "-") for t in techniques]

    print(f"  Publicando como: {full_name}")
    print(f"  Técnicas: {', '.join(techniques)}")

    hub.push(
        full_name,
        prompt_template,
        new_repo_is_public=True,
        new_repo_description=description,
        tags=all_tags,
    )

    print(f"  ✅ Push realizado com sucesso")
    print(f"  URL: https://smith.langchain.com/prompts/{full_name}")
    return True


def main():
    """Função principal"""
    required_vars = ["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]
    if not check_env_vars(required_vars):
        return 1

    print_section_header("Push de Prompts Otimizados para o LangSmith Hub")

    success = True
    for entry in PROMPTS_TO_PUSH:
        yaml_path = entry["yaml_path"]
        yaml_key = entry["yaml_key"]
        hub_name = entry["hub_name"]

        print(f"\nCarregando: {yaml_path}")

        data = load_yaml(yaml_path)
        if data is None:
            print(f"  ❌ Falha ao carregar {yaml_path}")
            success = False
            continue

        prompt_data = data.get(yaml_key)
        if prompt_data is None:
            print(f"  ❌ Chave '{yaml_key}' não encontrada em {yaml_path}")
            success = False
            continue

        is_valid, errors = validate_prompt(prompt_data)
        if not is_valid:
            print(f"  ❌ Validação falhou:")
            for err in errors:
                print(f"     - {err}")
            success = False
            continue

        if not push_prompt_to_langsmith(hub_name, prompt_data):
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
