"""
Executa a avaliação usando a API evaluate() do LangSmith.

Diferente de src/evaluate.py (que usa loop manual), este script cria
Experiments linkados ao dataset — visíveis na aba "Experiments" do LangSmith.

Uso:
    python evaluate_experiment.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain import hub

sys.path.insert(0, str(Path(__file__).parent / "src"))
from utils import check_env_vars, get_llm as get_configured_llm
from metrics import evaluate_f1_score, evaluate_clarity, evaluate_precision

load_dotenv()


def all_metrics_evaluator(run, example):
    """
    Avaliador único que calcula as 5 métricas por exemplo e as retorna juntas.
    Assim todas aparecem no LangSmith Experiment como colunas separadas.

    Métricas base:   f1_score, clarity, precision
    Métricas derivadas: helpfulness = (clarity + precision) / 2
                        correctness  = (f1 + precision) / 2
    """
    answer = (run.outputs or {}).get("output", "")
    reference = (example.outputs or {}).get("reference", "")
    question = (example.inputs or {}).get("bug_report", "")

    if not answer:
        return [
            {"key": "f1_score",     "score": 0.0},
            {"key": "clarity",      "score": 0.0},
            {"key": "precision",    "score": 0.0},
            {"key": "helpfulness",  "score": 0.0},
            {"key": "correctness",  "score": 0.0},
        ]

    f1        = evaluate_f1_score(question, answer, reference)["score"]
    clarity   = evaluate_clarity(question, answer, reference)["score"]
    precision = evaluate_precision(question, answer, reference)["score"]

    helpfulness = round((clarity + precision) / 2, 4)
    correctness  = round((f1 + precision) / 2, 4)

    return [
        {"key": "f1_score",     "score": f1},
        {"key": "clarity",      "score": clarity},
        {"key": "precision",    "score": precision},
        {"key": "helpfulness",  "score": helpfulness},
        {"key": "correctness",  "score": correctness},
    ]


def main():
    provider = os.getenv("LLM_PROVIDER", "openai")

    required_vars = ["LANGSMITH_API_KEY", "LLM_PROVIDER"]
    if provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif provider in ["google", "gemini"]:
        required_vars.append("GOOGLE_API_KEY")

    if not check_env_vars(required_vars):
        return 1

    username = os.getenv("USERNAME_LANGSMITH_HUB", "")
    if not username:
        print("❌ USERNAME_LANGSMITH_HUB não configurada no .env")
        return 1

    project_name = os.getenv("LANGSMITH_PROJECT", "MBA")
    dataset_name = f"{project_name}-eval"
    prompt_name = f"{username}/bug_to_user_story_v2"

    print(f"Prompt: {prompt_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Projeto: {project_name}\n")

    # Carrega o prompt e monta a chain
    print(f"Carregando prompt do LangSmith Hub...")
    prompt = hub.pull(prompt_name)
    llm = get_configured_llm(temperature=0)
    chain = prompt | llm

    def target(inputs):
        response = chain.invoke(inputs)
        return {"output": response.content}

    # Avaliador único que retorna as 5 métricas por exemplo
    evaluators = [all_metrics_evaluator]

    experiment_prefix = f"v2-{username}"
    print(f"Iniciando experiment: {experiment_prefix}")
    print(f"Os resultados ficarão visíveis em: https://smith.langchain.com/datasets/{dataset_name}\n")

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_prefix,
        metadata={"prompt": prompt_name, "provider": provider},
    )

    # Calcula médias a partir dos resultados
    scores = {"f1_score": [], "clarity": [], "precision": [], "helpfulness": [], "correctness": []}

    for r in results:
        for eval_result in r.get("evaluation_results", {}).get("results", []):
            key = eval_result.key
            if key in scores and eval_result.score is not None:
                scores[key].append(eval_result.score)

    if scores["f1_score"]:
        avgs = {k: sum(v) / len(v) for k, v in scores.items() if v}

        print("\n" + "=" * 50)
        print(f"Prompt: {prompt_name}")
        print("=" * 50)
        print(f"  Helpfulness:  {avgs.get('helpfulness', 0):.4f} {'✓' if avgs.get('helpfulness', 0) >= 0.9 else '✗'}")
        print(f"  Correctness:  {avgs.get('correctness', 0):.4f} {'✓' if avgs.get('correctness', 0) >= 0.9 else '✗'}")
        print(f"  F1-Score:     {avgs.get('f1_score', 0):.4f} {'✓' if avgs.get('f1_score', 0) >= 0.9 else '✗'}")
        print(f"  Clarity:      {avgs.get('clarity', 0):.4f} {'✓' if avgs.get('clarity', 0) >= 0.9 else '✗'}")
        print(f"  Precision:    {avgs.get('precision', 0):.4f} {'✓' if avgs.get('precision', 0) >= 0.9 else '✗'}")

        all_passed = all(avgs.get(k, 0) >= 0.9 for k in scores)
        print(f"\n{'✅ APROVADO' if all_passed else '❌ REPROVADO'}")

    print(f"\nExperiment criado. Acesse o dataset para ver os resultados:")
    print(f"  https://smith.langchain.com/o/public/datasets")

    return 0


if __name__ == "__main__":
    sys.exit(main())
