#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def load_cases(case_file: Path) -> dict:
    with case_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_case(case_file: Path, case_name: str) -> dict:
    cases = load_cases(case_file)
    for group_name in ("i2i_cases", "t2i_cases"):
        for case in cases.get(group_name, []):
            if case.get("name") == case_name:
                return case
    raise ValueError(f"Case not found: {case_name}")


def decode_first_image(resp_json: dict) -> bytes:
    data = resp_json.get("data", [])
    for item in data:
        if isinstance(item, dict) and item.get("b64_json"):
            return base64.b64decode(item["b64_json"])
    raise ValueError("No b64_json image found in response")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Bagel image-to-image via /v1/images/edits")
    parser.add_argument("--server", default="http://127.0.0.1:8091", help="Server base URL")
    parser.add_argument("--model", default=None, help="Optional model name/path")
    parser.add_argument("--case-file", default=None, help="Path to cases.json")
    parser.add_argument("--case-name", default="watercolor_preserve_scene", help="Case name in cases.json")
    parser.add_argument("--prompt", default=None, help="Override prompt")
    parser.add_argument("--image", default=None, help="Override input image path")
    parser.add_argument("--size", default=None, help="Override size, e.g. 1024x1024 or auto")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--true-cfg-scale", type=float, default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent
    case_file = Path(args.case_file) if args.case_file else script_dir / "cases.json"
    case = resolve_case(case_file, args.case_name)

    prompt = args.prompt or case["prompt"]
    image_rel = args.image or case["image"]
    image_path = Path(image_rel)
    if not image_path.is_absolute():
        image_path = repo_root / image_rel
    image_path = image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    size = args.size or case.get("size") or "auto"
    num_inference_steps = args.num_inference_steps or case.get("num_inference_steps")
    negative_prompt = args.negative_prompt or case.get("negative_prompt")
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else case.get("guidance_scale")
    true_cfg_scale = args.true_cfg_scale if args.true_cfg_scale is not None else case.get("true_cfg_scale")
    seed = args.seed if args.seed is not None else case.get("seed")

    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"{args.case_name}_{ts}"
    output_image = output_dir / f"{stem}.png"
    output_meta = output_dir / f"{stem}.json"

    endpoint = f"{args.server.rstrip('/')}/v1/images/edits"
    mime = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"

    data = {
        "prompt": prompt,
        "size": size,
        "response_format": "b64_json",
        "output_format": "png",
        "n": "1",
    }
    if args.model:
        data["model"] = args.model
    if num_inference_steps is not None:
        data["num_inference_steps"] = str(num_inference_steps)
    if guidance_scale is not None:
        data["guidance_scale"] = str(guidance_scale)
    if true_cfg_scale is not None:
        data["true_cfg_scale"] = str(true_cfg_scale)
    if negative_prompt:
        data["negative_prompt"] = negative_prompt
    if seed is not None:
        data["seed"] = str(seed)

    with image_path.open("rb") as f:
        files = {"image": (image_path.name, f, mime)}
        start = time.perf_counter()
        response = requests.post(endpoint, data=data, files=files, timeout=args.timeout)
        latency_s = time.perf_counter() - start

    response.raise_for_status()
    resp_json = response.json()
    image_bytes = decode_first_image(resp_json)
    output_image.write_bytes(image_bytes)

    meta = {
        "endpoint": endpoint,
        "status_code": response.status_code,
        "latency_seconds": latency_s,
        "case_name": args.case_name,
        "prompt": prompt,
        "image": str(image_path),
        "size": size,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "output_image": str(output_image),
        "response": resp_json,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with output_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "output_image": str(output_image),
                "output_meta": str(output_meta),
                "latency_seconds": latency_s,
                "status_code": response.status_code,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
