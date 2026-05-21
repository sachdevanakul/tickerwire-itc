"""
scripts/run_evals.py
One command: python scripts/run_evals.py
Runs all 50 eval queries, computes all metrics, prints rich report.
"""
import asyncio, sys, os, json, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import track
from eval.metrics import evaluate_faithfulness, evaluate_correctness

console = Console()
API_URL = "http://localhost:8000/query"
EVAL_FILE = os.path.join(os.path.dirname(__file__), "../eval/dataset.json")


async def query_api(query: str) -> dict:
    """Hit the /query endpoint and collect full response."""
    start = time.perf_counter()
    full_answer = []
    citations = []
    action = "unknown"
    first_token_time = None

    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream("POST", API_URL,
                                  json={"query": query, "stream": True},
                                  headers={"Accept": "text/event-stream"}) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = json.loads(line[5:].strip())

                if payload["type"] == "meta":
                    action = payload.get("action", "unknown")

                elif payload["type"] == "token":
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - start
                    full_answer.append(payload["content"])

                elif payload["type"] == "done":
                    citations = payload.get("citations", [])

    latency = time.perf_counter() - start
    return {
        "answer": "".join(full_answer),
        "citations": citations,
        "action": action,
        "latency_s": latency,
        "first_token_s": first_token_time or latency,
    }


async def run_evals():
    with open(EVAL_FILE) as f:
        dataset = json.load(f)

    console.print("\n[bold yellow]TickerWire-ITC Evaluation Harness[/bold yellow]")
    console.print(f"Running {len(dataset)} queries against {API_URL}\n")

    results = []
    latencies, first_tokens = [], []
    correctness_scores, faithfulness_scores = [], []

    for item in track(dataset, description="Evaluating..."):
        try:
            result = await query_api(item["query"])
            latencies.append(result["latency_s"])
            first_tokens.append(result["first_token_s"])

            correctness = evaluate_correctness(result["answer"], item["ground_truth"])
            correctness_scores.append(correctness)

            # Faithfulness only for retrieval-based answers
            if result["action"] in ("direct_answer", "retrieve_then_answer") and result["citations"]:
                faith = evaluate_faithfulness(result["answer"], result["citations"])
                faithfulness_scores.append(faith)

            results.append({**item, **result, "correctness": correctness})

        except Exception as e:
            console.print(f"[red]Error on {item['id']}: {e}[/red]")
            results.append({**item, "answer": "", "latency_s": 0, "correctness": 0, "error": str(e)})

    # ── Print report ──────────────────────────────────────────────────────────
    def pct(vals): return round(statistics.mean(vals) * 100, 1) if vals else 0
    def p95(vals): return round(sorted(vals)[int(len(vals) * 0.95)], 2) if vals else 0

    table = Table(title="\n📊 Evaluation Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Target", justify="right", style="dim")

    table.add_row("Correctness (mean)", f"{pct(correctness_scores)}%", "≥ 75%")
    table.add_row("Faithfulness (mean)", f"{pct(faithfulness_scores)}%", "≥ 90%")
    table.add_row("Latency p50", f"{statistics.median(latencies):.2f}s", "—")
    table.add_row("Latency p95", f"{p95(latencies)}s", "≤ 5s")
    table.add_row("First-token p50", f"{statistics.median(first_tokens):.2f}s", "< 1.5s")
    table.add_row("Total queries", str(len(results)), "50")

    console.print(table)

    # Save JSON results
    out = "eval/results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]✓ Full results saved to {out}[/green]")

    # Fail check
    correctness_mean = statistics.mean(correctness_scores) if correctness_scores else 0
    faith_mean = statistics.mean(faithfulness_scores) if faithfulness_scores else 0
    lat_p95 = p95(latencies)

    if correctness_mean < 0.75 or faith_mean < 0.90 or lat_p95 > 5.0:
        console.print("\n[red bold]⚠️  Some targets not met — check results above.[/red bold]")
        sys.exit(1)
    else:
        console.print("\n[green bold]✅ All targets met![/green bold]\n")


if __name__ == "__main__":
    asyncio.run(run_evals())
