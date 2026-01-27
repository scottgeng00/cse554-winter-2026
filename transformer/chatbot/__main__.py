import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional, Set

import torch

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}
RESET_COMMANDS = {"reset", "/reset"}


def _load_transformer_detailed(weight_path: Optional[str]):
    transformer_dir = Path(__file__).resolve().parents[1]
    if weight_path:
        os.environ["TRANSFORMER_WEIGHT_PATH"] = weight_path

    if str(transformer_dir) not in sys.path:
        sys.path.insert(0, str(transformer_dir))

    module_name = "transformer_detailed"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = transformer_dir / "transformer-detailed.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load transformer module at {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_stop_token_ids(tokenizer) -> Set[int]:
    stop_ids: Set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    for token in ("<|eot_id|>", "<|end_of_text|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            continue
        if tokenizer.unk_token_id is not None and token_id == tokenizer.unk_token_id:
            continue
        stop_ids.add(token_id)

    return stop_ids


def _generate_reply(module, messages, max_new_tokens: int, stop_token_ids: Set[int]) -> str:
    tokenizer = module.tokenizer
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = prompt_ids[0].tolist()
    output_ids = list(input_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            next_token = module.run_one_iteration(output_ids)
            output_ids.append(next_token)
            if stop_token_ids and next_token in stop_token_ids:
                break

    new_tokens = output_ids[len(input_ids):]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return reply.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI chatbot wrapper around transformer-detailed.py",
    )
    parser.add_argument(
        "--weight-path",
        type=str,
        default=None,
        help="Path to model weights (overrides TRANSFORMER_WEIGHT_PATH).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max tokens to generate per response.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Custom system prompt (defaults to a built-in prompt).",
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Disable the system prompt entirely.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print("Loading model weights... this can take a while.")
    module = _load_transformer_detailed(args.weight_path)
    tokenizer = module.tokenizer
    stop_token_ids = _build_stop_token_ids(tokenizer)

    system_message = None
    messages = []
    if not args.no_system:
        system_prompt = args.system if args.system is not None else DEFAULT_SYSTEM_PROMPT
        system_prompt = system_prompt.strip()
        if system_prompt:
            system_message = {"role": "system", "content": system_prompt}
            messages.append(system_message)

    print("Chat ready. Type /exit to quit or /reset to clear the conversation.")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in EXIT_COMMANDS:
            break
        if lowered in RESET_COMMANDS:
            messages = [system_message] if system_message else []
            print("assistant> Conversation reset.")
            continue

        messages.append({"role": "user", "content": user_input})
        reply = _generate_reply(module, messages, args.max_new_tokens, stop_token_ids)
        messages.append({"role": "assistant", "content": reply})
        print(f"assistant> {reply}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
