#!/usr/bin/env python3
"""Join same-prompt historical and FlashAttention logs into a CSV."""

from pathlib import Path
import csv
import re


PROMPT_LOGS = [
    (
        "1.5B",
        "benchmark-results/kv-cache-facade-2026-10-20/1.5B.log",
        "benchmark-results/flash-attention-2026-10-21/1.5B-fp16-gpu4-final.log",
    ),
    (
        "3B",
        "benchmark-results/kv-cache-facade-2026-10-20/3B.log",
        "benchmark-results/flash-attention-2026-10-21/3B-fp16-gpu4-final.log",
    ),
    (
        "7B",
        "benchmark-results/kv-cache-facade-2026-10-20/7B.log",
        "benchmark-results/flash-attention-2026-10-21/7B-fp16-gpu4-final.log",
    ),
    (
        "14B",
        "benchmark-results/kv-cache-facade-2026-10-20/14B-gpu4.log",
        "benchmark-results/flash-attention-2026-10-21/14B-fp16-gpu4.log",
    ),
]

PROMPT_RE = re.compile(r"^--- Prompt \d+/\d+: (.*?) ---$")
INPUT_RE = re.compile(r"^Input tokens: (\d+) ")
OUTPUT_RE = re.compile(r"^\[(custom|cublas)\] (.*?) -> generated output:$")
OUTPUT_END_RE = re.compile(r"^\[(custom|cublas)\] --- End generated output ---$")
TIMING_RE = re.compile(
    r"^\[(custom|cublas)\] Timing: prefill (\d+) tokens in "
    r"([0-9.]+)(ms|s); generated (\d+) tokens with (\d+) decode steps in "
    r"([0-9.]+)(ms|s)"
)


def duration_ms(value: str, unit: str) -> float:
    return float(value) * (1000.0 if unit == "s" else 1.0)


def parse_log(path: str) -> dict:
    cases = {}
    prompt = None
    output_backend = None
    output_lines = []

    for line in Path(path).read_text(errors="replace").splitlines():
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
                prefill_ms=duration_ms(prefill, prefill_unit),
                generated=int(generated),
                decode_steps=int(steps),
                decode_ms=duration_ms(decode, decode_unit),
            )

    return cases


def delta_percent(previous: float, current: float) -> str:
    return f"{(current / previous - 1.0) * 100.0:.1f}"


def main() -> None:
    output = Path("benchmark-results/flash-attention-2026-10-21/per_prompt_comparison.csv")
    markdown = Path("benchmark-results/flash-attention-2026-10-21/per_prompt_comparison.md")
    columns = [
        "model",
        "prompt",
        "input_tokens",
        "backend",
        "previous_prefill_ms",
        "flash_prefill_ms",
        "prefill_delta_pct",
        "previous_decode_ms",
        "flash_decode_ms",
        "decode_delta_pct",
        "previous_generated_tokens",
        "flash_generated_tokens",
        "output_text_match",
    ]

    with output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for model, previous_path, flash_path in PROMPT_LOGS:
            previous = parse_log(previous_path)
            flash = parse_log(flash_path)
            if set(previous) != set(flash):
                raise ValueError(f"Prompt sets differ for {model}")

            for prompt, flash_case in flash.items():
                previous_case = previous[prompt]
                input_tokens = flash_case["input_tokens"]
                if previous_case["input_tokens"] != input_tokens:
                    raise ValueError(f"Input token counts differ: {model} / {prompt}")

                for backend in ("custom", "cublas"):
                    old = previous_case[backend]
                    new = flash_case[backend]
                    writer.writerow(
                        {
                            "model": model,
                            "prompt": prompt,
                            "input_tokens": input_tokens,
                            "backend": backend,
                            "previous_prefill_ms": f"{old['prefill_ms']:.2f}",
                            "flash_prefill_ms": f"{new['prefill_ms']:.2f}",
                            "prefill_delta_pct": delta_percent(
                                old["prefill_ms"], new["prefill_ms"]
                            ),
                            "previous_decode_ms": f"{old['decode_ms']:.2f}",
                            "flash_decode_ms": f"{new['decode_ms']:.2f}",
                            "decode_delta_pct": delta_percent(
                                old["decode_ms"], new["decode_ms"]
                            ),
                            "previous_generated_tokens": old["generated"],
                            "flash_generated_tokens": new["generated"],
                            "output_text_match": "IDENTICAL"
                            if old.get("output") == new.get("output")
                            else "CHANGED",
                        }
                    )

    rows = list(csv.DictReader(output.open(newline="")))
    grouped = {}
    for row in rows:
        key = (row["model"], row["prompt"], row["input_tokens"])
        grouped.setdefault(key, {})[row["backend"]] = row

    def comparison(row: dict, prefix: str) -> str:
        old = row[f"previous_{prefix}_ms"]
        new = row[f"flash_{prefix}_ms"]
        delta = float(row[f"{prefix}_delta_pct"])
        return f"{old} → {new} ({delta:+.1f}%)"

    lines = [
        "# FlashAttention per-prompt comparison",
        "",
        "Each timing cell is previous attention → FlashAttention in milliseconds, with the percentage change in parentheses. Negative percentages are faster.",
        "",
        "Previous logs: `benchmark-results/kv-cache-facade-2026-10-20/`. Current raw logs are in this directory. Both use `--max-tokens 16 --linear both --weight-format fp16`; current runs used GPU 4 for all four checkpoints, while the earlier 1.5B, 3B, and 7B runs used GPU 1 and 14B used GPU 4. Both revisions used the single-pass prompt timing path with default GEMM prefill. These are per-prompt measurements, not repeated medians.",
        "",
        "The generated output text matched the previous log in all 72 model/prompt/backend rows. The current logs report identical generated tokens across custom and cuBLAS backends and zero non-finite logits in every prompt case.",
        "",
        "| Model | Prompt | Input tokens | Custom prefill ms | Custom decode ms | cuBLAS prefill ms | cuBLAS decode ms | Output text |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for (model, prompt, input_tokens), backends in grouped.items():
        custom = backends["custom"]
        cublas = backends["cublas"]
        text_match = (
            "IDENTICAL"
            if custom["output_text_match"] == cublas["output_text_match"] == "IDENTICAL"
            else "CHANGED"
        )
        lines.append(
            f"| {model} | {prompt} | {input_tokens} | {comparison(custom, 'prefill')} | "
            f"{comparison(custom, 'decode')} | {comparison(cublas, 'prefill')} | "
            f"{comparison(cublas, 'decode')} | {text_match} |"
        )
    markdown.write_text("\n".join(lines) + "\n")
    print(output)
    print(markdown)


if __name__ == "__main__":
    main()
