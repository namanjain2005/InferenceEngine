#!/usr/bin/env python3
"""Compare long-context runs from the pre-FlashAttention and Flash revisions."""

from pathlib import Path
import csv
import re


ROOT = Path("benchmark-results/flash-attention-long-context-2026-10-21")
RUNS = {
    "1.5B": ("previous-1.5B-gpu3.log", "flash-1.5B-gpu3.log", 3),
    "3B": ("previous-3B-gpu3.log", "flash-3B-gpu3.log", 3),
    "7B": ("previous-7B-gpu3.log", "flash-7B-gpu3.log", 3),
    "14B": ("previous-14B-gpu4.log", "flash-14B-gpu4.log", 4),
}
STRESS_PROMPTS = (
    "10 - Approximately 1K-token context",
    "11 - Approximately 4K-token context",
    "12 - Approximately 8K-token context",
)

PROMPT_RE = re.compile(r"^--- Prompt \d+/\d+: (.*?) ---$")
INPUT_RE = re.compile(r"^Input tokens: (\d+) ")
OUTPUT_RE = re.compile(r"^\[(custom|cublas)\] .* -> generated output:$")
OUTPUT_END_RE = re.compile(r"^\[(custom|cublas)\] --- End generated output ---$")
TIMING_RE = re.compile(
    r"^\[(custom|cublas)\] Timing: prefill (\d+) tokens in "
    r"([0-9.]+)(ms|s); generated (\d+) tokens with (\d+) decode steps in "
    r"([0-9.]+)(ms|s)"
)
ERROR_RE = re.compile(r"^Error: .*Prompt '(.+)' failed on backend '(custom|cublas)': (.*)$")


def ms(value: str, unit: str) -> float:
    return float(value) * (1000.0 if unit == "s" else 1.0)


def parse_log(path: Path) -> tuple[dict, dict]:
    cases = {}
    errors = {}
    prompt = None
    output_backend = None
    output_lines = []

    for line in path.read_text(errors="replace").splitlines():
        match = PROMPT_RE.match(line)
        if match:
            prompt = match.group(1)
            cases.setdefault(prompt, {})
            continue
        if prompt is None:
            continue

        match = INPUT_RE.match(line)
        if match:
            cases[prompt]["input_tokens"] = int(match.group(1))

        match = OUTPUT_RE.match(line)
        if match:
            output_backend = match.group(1)
            output_lines = []
        elif output_backend is not None:
            if OUTPUT_END_RE.match(line):
                cases[prompt].setdefault(output_backend, {})["output"] = "\n".join(
                    output_lines
                ).strip()
                output_backend = None
                output_lines = []
            else:
                output_lines.append(line)

        match = TIMING_RE.match(line)
        if match:
            backend, input_tokens, prefill, prefill_unit, generated, steps, decode, decode_unit = (
                match.groups()
            )
            cases[prompt].setdefault(backend, {}).update(
                input_tokens=int(input_tokens),
                prefill_ms=ms(prefill, prefill_unit),
                generated=int(generated),
                decode_steps=int(steps),
                decode_ms=ms(decode, decode_unit),
            )

        match = ERROR_RE.match(line)
        if match:
            error_prompt, backend, message = match.groups()
            errors[(error_prompt, backend)] = message

    return cases, errors


def percent(old: float, new: float) -> str:
    return f"{(new / old - 1.0) * 100.0:+.1f}%"


def timing(old: dict, new: dict, field: str) -> str:
    if field not in old or field not in new:
        return "—"
    return (
        f"{old[field]:.2f} → {new[field]:.2f} ms "
        f"({percent(old[field], new[field])})"
    )


def status(old: dict, new: dict, old_error: str | None, new_error: str | None) -> str:
    if "prefill_ms" in old and "prefill_ms" in new:
        return "matched" if old.get("output") == new.get("output") else "changed"
    if old_error and new_error:
        return "OOM on both revisions"
    if old_error:
        return "OOM on previous revision"
    if new_error:
        return "OOM on FlashAttention revision"
    return "not reached after earlier OOM"


def main() -> None:
    rows = []
    report = [
        "# Long-context FlashAttention comparison",
        "",
        "Each timing is previous attention → FlashAttention, in milliseconds. Percentages compare the new value with the previous one.",
        "",
        "The previous implementation is commit `47960ee`; FlashAttention is commit `d381615`. Both used the same expanded `BASE_PROMPTS.txt`, `--max-tokens 16`, `--linear both`, FP16 projection weights, and the same GPU for each model pair. GPU 3 ran 1.5B/3B/7B; GPU 4 ran 14B. The host was shared and GPU 3 was frequently busy, so these are single-pass measurements and small differences are noisy.",
        "",
        "Input token counts were 1,006, 4,036, and 8,076 for the approximately 1K, 4K, and 8K cases. All 1.5B, 3B, and 7B cases completed on both revisions. Both 14B runs completed 1K, then failed at 4K during custom prefill with CUDA out of memory while allocating a GEMM output. The 14B 8K case was not reached. This is reported as a model working-memory limit; it is not a shared-memory launch failure from the attention kernel.",
        "",
        "| Model | Context | Input tokens | Backend | Prefill old → new | Decode old → new | Output text | Status |",
        "|---|---|---:|---|---:|---:|---|---|",
    ]

    for model, (old_name, new_name, gpu) in RUNS.items():
        old_cases, old_errors = parse_log(ROOT / old_name)
        new_cases, new_errors = parse_log(ROOT / new_name)
        for prompt in STRESS_PROMPTS:
            old_case = old_cases.get(prompt, {})
            new_case = new_cases.get(prompt, {})
            old_tokens = old_case.get("input_tokens")
            new_tokens = new_case.get("input_tokens")
            if old_tokens is not None and new_tokens is not None and old_tokens != new_tokens:
                raise ValueError(f"Input token counts differ for {model} / {prompt}")
            input_tokens = new_tokens if new_tokens is not None else old_tokens

            for backend in ("custom", "cublas"):
                old = old_case.get(backend, {})
                new = new_case.get(backend, {})
                old_error = old_errors.get((prompt, backend))
                new_error = new_errors.get((prompt, backend))
                case_status = status(old, new, old_error, new_error)
                match = (
                    "IDENTICAL"
                    if "output" in old and "output" in new and old["output"] == new["output"]
                    else "CHANGED"
                    if "output" in old and "output" in new
                    else "—"
                )
                rows.append(
                    {
                        "model": model,
                        "gpu": gpu,
                        "prompt": prompt,
                        "input_tokens": input_tokens if input_tokens is not None else "",
                        "backend": backend,
                        "previous_prefill_ms": f"{old['prefill_ms']:.2f}" if "prefill_ms" in old else "",
                        "flash_prefill_ms": f"{new['prefill_ms']:.2f}" if "prefill_ms" in new else "",
                        "prefill_delta_pct": percent(old["prefill_ms"], new["prefill_ms"])
                        if "prefill_ms" in old and "prefill_ms" in new
                        else "",
                        "previous_decode_ms": f"{old['decode_ms']:.2f}" if "decode_ms" in old else "",
                        "flash_decode_ms": f"{new['decode_ms']:.2f}" if "decode_ms" in new else "",
                        "decode_delta_pct": percent(old["decode_ms"], new["decode_ms"])
                        if "decode_ms" in old and "decode_ms" in new
                        else "",
                        "previous_generated_tokens": old.get("generated", ""),
                        "flash_generated_tokens": new.get("generated", ""),
                        "output_text_match": match,
                        "status": case_status,
                    }
                )
                report.append(
                    f"| {model} | {prompt.split(' - ')[1]} | {input_tokens or '—'} | {backend} | "
                    f"{timing(old, new, 'prefill_ms')} | {timing(old, new, 'decode_ms')} | "
                    f"{match} | {case_status} |"
                )

    csv_path = ROOT / "long_context_comparison.csv"
    markdown_path = ROOT / "long_context_comparison.md"
    columns = list(rows[0])
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text("\n".join(report) + "\n")
    print(csv_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
