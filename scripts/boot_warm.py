"""Boot request warm-up for the GLM endpoint: wait for /health, then send the first traffic ourselves.

python3 boot_warm.py [--base http://127.0.0.1:8093] [--model NAME] [--long-tokens 16384] [--long-count 2]
                     [--wait 1800] [--log FILE]

Why (diagnostics/glm-inboot/REPORT.md): the first long cold prefill after every boot ran at 1128-1570 tok/s
against ~2000-2170 for every later one, because vLLM's own warm-up and our bench warm-ups only send short
prompts. So the first user with a long prompt paid for it. This sends, once /health is 200:
  * one short chat (the first request through the API server, tokenizer, chat template and sampler);
  * --long-count cold prefills of ~--long-tokens tokens (max_tokens=1): a unique nonce AND a unique body per
    prompt, so neither can hit the prefix cache (bench/prefill_bench.py's text generator, sized with /tokenize).
Each result is logged with its prefill tok/s, so every boot records first-vs-second long prefill for free.
The last line is "boot-warm done" (bench/cycle scripts wait for it). Exit 0 unless the server never came up.
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import prefill_bench as pb  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def healthy(base):
    try:
        return urllib.request.urlopen(base + "/health", timeout=3).status == 200
    except Exception:  # noqa: BLE001
        return False


def short_chat(base, model):
    body = dict(model=model, messages=[{"role": "user", "content": "Say hello in five words."}], max_tokens=32,
                temperature=0, chat_template_kwargs={"reasoning_effort": "low"})
    t0 = time.monotonic()
    out = json.load(pb.post(base, "/v1/chat/completions", body, timeout=600))
    return round(time.monotonic() - t0, 3), out.get("usage", {}).get("completion_tokens")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--base", default="http://127.0.0.1:8093")
    a.add_argument("--model", default="GLM-5.3-Flash-FP8")
    a.add_argument("--long-tokens", type=int, default=int(os.environ.get("BOOT_WARM_TOKENS", "16384")))
    a.add_argument("--long-count", type=int, default=int(os.environ.get("BOOT_WARM_COUNT", "2")))
    a.add_argument("--wait", type=int, default=1800, help="seconds to wait for /health")
    a = a.parse_args()
    t0 = time.monotonic()
    log(f"boot-warm: waiting for {a.base}/health (up to {a.wait} s)")
    while not healthy(a.base):
        if time.monotonic() - t0 > a.wait:
            log("boot-warm: server never became healthy; nothing sent")
            log("boot-warm done (no server)")
            return 1
        time.sleep(5)
    log(f"boot-warm: healthy after {time.monotonic() - t0:.0f} s")
    try:
        dt, n = short_chat(a.base, a.model)
        log(f"boot-warm: short chat {dt} s, {n} tokens")
        seed0 = random.SystemRandom().randrange(1 << 30)
        for i in range(a.long_count):
            body = pb.build(a.base, a.model, a.long_tokens, seed0 + i)
            prompt = f"[{uuid.uuid4().hex}] Read the numbered notes below and reply with one word.\n" + body
            r = pb.ttft(a.base, a.model, prompt)
            log(f"boot-warm: long cold prefill {i + 1}/{a.long_count}: {json.dumps(r)}")
    except Exception as exc:  # noqa: BLE001  a failed warm-up must not look like a failed boot
        log(f"boot-warm: request failed: {exc!r}")
    log(f"boot-warm done in {time.monotonic() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
