"""Vision tool-calling loop: OpenAI Responses and Anthropic Messages via HTTP.

No SDK defaults, hidden oracle, or policy that chooses where the agent looks.
Provider model names are required explicitly. Images are the session's saved PNGs.
"""

import base64
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from .tools import schemas
from .provider_http import request_json
from .image_transport import lossless_data_url, deduplicate_images, inline_image_bytes

SYSTEM = """You are proofreading a segmented microscopy volume. Explore freely using the supplied tools.
Inspect the imaging evidence, choose your own views and segments, and correct false merges or splits.
Before each tool call, give a brief user-facing explanation of what you observed, what you will inspect or change next, and why.
Use set_scene to rotate, pan and zoom the persistent 3D selected-segment view; set_view controls the EM plane.
list_segments lists the colored segments in the current EM view, not the whole volume.
With no selection all visible segments are colored; with a selection only selected segments are colored.
EM intensity percentiles are measured before display transforms; adjust set_view contrast=[low,high] when the useful range is narrow, and overlay=0 to inspect unobscured membranes.
Check em_sampling after zooming: streamed worlds choose matched EM/segmentation detail automatically unless set_em_resolution pins it. Pass null to restore automatic detail. Fixed worlds require explicit source-resolution changes. Zooming beyond the finest available source only enlarges pixels.
EM and 3D share stable segment colors. Explore by imaging and clicking; there is no graph-inspection tool.
Coordinates are XYZ nanometers. Pixel clicks use the exact view_id and zero-based pixel centers.
Segment and atom IDs are decimal STRINGS; do not round them. Selection/hiding is not an edit.
Preview an edit, inspect it, then commit if warranted. Use undo_edit for segmentation edits and undo_action for the latest state change. You may inspect any segment
inside the declared volume. Slab clicks refer to the center plane. Mincut uses the reported graph
capacities, not a correctness oracle. Do not assume a preview is a good edit. End with finish when done.
No human reference or correctness feedback is available during this evaluation."""


DISPLAY_GUIDANCE = 'EM intensity percentiles are measured before display transforms; adjust set_view contrast=[low,high] when the useful range is narrow, and overlay=0 to inspect unobscured membranes.\nCheck em_sampling after zooming: streamed worlds choose matched EM/segmentation detail automatically unless set_em_resolution pins it. Pass null to restore automatic detail. Fixed worlds require explicit source-resolution changes. Zooming beyond the finest available source only enlarges pixels.\n'
BASELINE_SYSTEM = SYSTEM.replace(DISPLAY_GUIDANCE, "")

def image_paths(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "image" and isinstance(item, str):
                yield item
            else:
                yield from image_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from image_paths(item)


def content(result, directory, provider):
    text_type = "input_text" if provider == "responses" else "text"
    blocks = [{"type": text_type, "text": json.dumps(result)}]
    for name in image_paths(result):
        path = (Path(directory) / name).resolve()
        if not path.is_relative_to(Path(directory).resolve() / "frames"):
            raise ValueError("Image outside the session frames")
        encoded = base64.b64encode(path.read_bytes()).decode()
        image_url = "data:image/png;base64," + encoded
        if provider == "openrouter":
            image_url = lossless_data_url(image_url)
        if provider == "gemini":
            blocks.append({"type":"image","mime_type":"image/png","data":encoded})
        elif provider in ("responses", "chat", "openrouter"):
            blocks.append(
                {
                    **({"type": "input_image", "image_url": image_url, "detail": "high"}
                       if provider == "responses" else
                       {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}),
                }
            )
        else:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": encoded,
                    },
                }
            )
    return blocks


def post(endpoint, payload, headers):
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in ["localhost", "127.0.0.1", "::1"]
    ):
        raise ValueError(
            "Use HTTPS, or HTTP on loopback for a local provider/test server"
        )
    request = Request(
        endpoint,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json", **headers},
    )
    return request_json(request)


def run(
    session,
    model,
    provider="responses",
    max_turns=100,
    max_actions=500,
    max_output_tokens=8192,
    endpoint=None,
    resume_from=None,
    api_key=None,
    transport=post,
    model_options=None,
    objective=None,
    system_prompt=None,
    demonstrations=None,
):
    if provider not in ["responses", "anthropic", "chat", "openrouter", "gemini"]:
        raise ValueError("Unknown provider")
    if provider == "gemini" and demonstrations:
        raise ValueError("Gemini demonstrations are not implemented")
    if not model or min(max_turns, max_actions, max_output_tokens) < 1:
        raise ValueError("Model and positive budgets required")
    if objective is not None and (not isinstance(objective, str) or not objective.strip()):
        raise ValueError("Objective must be nonempty text")
    if system_prompt is not None and (not isinstance(system_prompt,str) or not system_prompt.strip()):
        raise ValueError("System prompt must be nonempty text")
    system = (SYSTEM if system_prompt is None else system_prompt) + ("\n\nTask objective:\n" + objective if objective else "")
    model_options = {} if model_options is None else model_options
    if not isinstance(model_options, dict):
        raise ValueError("model_options must be an object")
    model_options = dict(model_options)
    if provider == "anthropic":
        model_options.setdefault("cache_control", {"type": "ephemeral", "ttl": "5m"})
    elif provider == "responses":
        prefix = json.dumps([model, system, schemas()], sort_keys=True).encode()
        model_options.setdefault("prompt_cache_key", "connectome-" + hashlib.sha256(prefix).hexdigest()[:32])
    allowed = {"reasoning", "thinking", "output_config", "temperature", "top_p", "cache_control", "prompt_cache_key"}
    if provider in ("chat", "openrouter"):
        allowed = {"temperature", "top_p", "reasoning_effort", "seed"}
        if provider == "openrouter":
            allowed |= {"reasoning", "provider"}
    if provider == "gemini":
        allowed = {"thinking_level", "temperature", "top_p"}
    if not isinstance(model_options, dict) or set(model_options) - allowed:
        raise ValueError(
            "model_options may configure reasoning, thinking, output_config, temperature, top_p, or Anthropic cache_control"
        )
    if "cache_control" in model_options:
        cache = model_options["cache_control"]
        if (provider != "anthropic" or (cache is not None and (not isinstance(cache, dict)
                or set(cache) - {"type", "ttl"} or cache.get("type") != "ephemeral"
                or cache.get("ttl", "5m") not in ("5m", "1h")))):
            raise ValueError("cache_control requires Anthropic ephemeral caching with 5m or 1h TTL")
    if "prompt_cache_key" in model_options and (provider != "responses" or not isinstance(model_options["prompt_cache_key"], str) or not model_options["prompt_cache_key"]):
        raise ValueError("prompt_cache_key requires a nonempty OpenAI Responses cache key")
    key_name = {"responses": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
                "chat": "MODEL_API_KEY", "openrouter": "OPENROUTER_API_KEY", "gemini": "GEMINI_API_KEY"}[provider]
    if provider == "chat" and not endpoint:
        raise ValueError("chat requires an explicit full chat/completions endpoint")
    api_key = api_key if api_key is not None else os.environ.get(key_name)
    if not api_key:
        raise ValueError(
            f"Set {key_name} in the shell; never put credentials in the dataset or logs"
        )
    endpoint = endpoint or (
        "https://api.openai.com/v1/responses"
        if provider == "responses"
        else "https://openrouter.ai/api/v1/chat/completions" if provider == "openrouter"
        else "https://generativelanguage.googleapis.com/v1beta/interactions" if provider == "gemini"
        else "https://api.anthropic.com/v1/messages"
    )
    headers = (
        {"Authorization": f"Bearer {api_key}"}
        if provider != "anthropic"
        else {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    )
    if provider == "gemini":
        headers = {"x-goog-api-key": api_key}
    previous_interaction_id = None
    initial_started = time.monotonic()
    # Record the configured initial camera before producing the paired observation.
    session.execute("set_scene", dict(session.scene))
    start = {"state": session.execute("state"), "view": session.execute("look")}
    (session.output / "initial-observations.json").write_text(json.dumps([session.serial]) + "\n")
    from .demonstrations import load_context
    prefix = load_context(demonstrations, provider) if demonstrations else []
    history = prefix + [{"role": "user", "content": content(start, session.output, provider)}]
    if provider == "gemini":
        history = content(start, session.output, provider)
    if demonstrations:
        demo_path = Path(demonstrations).resolve()
        (session.output / "demonstrations.json").write_text(json.dumps({"manifest":str(demo_path), "sha256":hashlib.sha256(demo_path.read_bytes()).hexdigest(), "spec":json.loads(demo_path.read_text())}, indent=2))
    (session.output / "system-prompt.txt").write_text(system + "\n")
    total_actions = 0
    summary = {
        "model": model,
        "provider": provider,
        "status": "turn_budget",
        "turns": 0,
        "actions": 0,
        "usage": [],
        "new_request_usage": [],
        "max_turns": max_turns,
        "max_actions": max_actions,
        "max_output_tokens": max_output_tokens,
        "model_options": model_options,
        "objective": objective,
        "timings": {"initial_observation_seconds": time.monotonic() - initial_started,
                    "provider_seconds": 0.0, "tool_seconds": 0.0},
    }
    from .recovery import JournalTransport, atomic_json
    transport = JournalTransport(session.output, transport, resume_from=resume_from)
    started = time.monotonic()
    try:
        for turn in range(max_turns):
            if provider == "gemini":
                payload = {"model":model,"system_instruction":system,"input":history,
                    "tools":[{"type":"function",**s} for s in schemas()],
                    "generation_config":{"max_output_tokens":max_output_tokens,**model_options}}
                if previous_interaction_id:
                    payload["previous_interaction_id"] = previous_interaction_id
            elif provider == "responses":
                payload = {
                    "model": model,
                    "instructions": system,
                    "input": history,
                    "store": False,
                    "include": ["reasoning.encrypted_content"],
                    "max_output_tokens": max_output_tokens,
                    "parallel_tool_calls": False,
                    "tools": [
                        {"type": "function", "strict": False, **s} for s in schemas()
                    ],
                }
            elif provider in ("chat", "openrouter"):
                payload = {
                    "model": model, "messages": [{"role": "system", "content": system}, *history],
                    "max_tokens": max_output_tokens,
                    "tools": [{"type": "function", "function": s} for s in schemas()],
                    # OpenRouter strict routing rejects this optional parameter on
                    # many endpoints. Multiple returned calls run sequentially below.
                    **({"parallel_tool_calls": False} if provider == "chat" else {}),
                }
            else:
                payload = {
                    "model": model,
                    "system": system,
                    "messages": history,
                    "max_tokens": max_output_tokens,
                    "tools": [
                        {
                            "name": s["name"],
                            "description": s["description"],
                            "input_schema": s["parameters"],
                        }
                        for s in schemas()
                    ],
                }
            if provider != "gemini":
                payload.update({k:v for k,v in model_options.items() if not (k == "cache_control" and v is None)})
            if provider == "openrouter" and inline_image_bytes(payload) > 24 * 1024 * 1024:
                payload = deduplicate_images(payload)
            request_started = time.monotonic()
            response = transport(endpoint, payload, headers)
            if provider == "gemini":
                from .gemini import normalized_response
                response = normalized_response(response)
                previous_interaction_id = response["id"]
            summary["timings"]["provider_seconds"] += time.monotonic() - request_started
            summary["turns"] = turn + 1
            summary["usage"].append(response.get("usage", {}))
            if not transport.last_replayed:
                summary["new_request_usage"].append(response.get("usage", {}))
            with (session.output / "model-responses.jsonl").open("a") as f:
                f.write(json.dumps({"turn": turn + 1, "response": response}) + "\n")
            if provider == "responses" and response.get("status") == "incomplete":
                summary["status"] = "provider_incomplete"
                summary["incomplete_details"] = response.get("incomplete_details")
                break
            if provider == "anthropic" and response.get("stop_reason") == "max_tokens":
                summary["status"] = "provider_incomplete"
                summary["incomplete_details"] = {"reason": "max_tokens"}
                break
            if provider == "gemini" and response.get("status") not in ("completed", "requires_action"):
                summary["status"] = "provider_incomplete"
                summary["incomplete_details"] = {"status":response.get("status")}
                break
            chat_message = None
            if provider in ("chat", "openrouter"):
                choices = response.get("choices", [])
                if not choices:
                    raise ValueError("Chat provider returned no choices")
                choice = choices[0]
                if choice.get("finish_reason") in ("length", "content_filter"):
                    summary["status"] = "provider_incomplete"
                    summary["incomplete_details"] = {"reason": choice["finish_reason"]}
                    break
                chat_message = choice["message"]
            output = (
                response.get("output", [])
                if provider == "responses"
                else response.get("content", [])
            )
            if chat_message is not None:
                # Preserve provider reasoning metadata for models that require it on subsequent turns.
                history.append(chat_message)
                output = [{"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                           "input_json": c["function"]["arguments"]} for c in chat_message.get("tool_calls", [])]
            elif provider == "responses":
                history.extend(output)
            elif provider != "gemini":
                history.append({"role": "assistant", "content": output})
            calls = [
                x
                for x in output
                if x.get("type")
                == ("function_call" if provider == "responses" else "tool_use")
            ]
            if not calls:
                summary["status"] = "model_stopped_without_finish"
                break
            results = []
            for call in calls:
                if total_actions >= max_actions:
                    summary["status"] = "action_budget"
                    break
                tool_started = time.monotonic()
                before_step = session.serial
                args = None
                try:
                    args = (
                        json.loads(call["arguments"])
                        if provider == "responses"
                        else json.loads(call["input_json"]) if provider in ("chat", "openrouter")
                        else call["input"]
                    )
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be an object")
                    result = session.execute(call["name"], args)
                except (ValueError, KeyError, TypeError) as exc:
                    result = {"ok": False, "error": str(exc)}
                tool_seconds = time.monotonic() - tool_started
                summary["timings"]["tool_seconds"] += tool_seconds
                with (session.output / "model-actions.jsonl").open("a") as f:
                    f.write(json.dumps({"turn": turn + 1, "call_id": call.get("call_id", call.get("id")),
                                        "step": session.serial if session.serial > before_step else None,
                                        "name": call.get("name"), "arguments": args, "elapsed_seconds": tool_seconds}) + "\n")
                total_actions += 1
                if provider == "responses":
                    results.append(
                        {
                            "type": "function_call_output",
                            "call_id": call["call_id"],
                            "output": content(result, session.output, provider),
                        }
                    )
                elif provider == "gemini":
                    results.append({"type":"function_result","name":call["name"],"call_id":call["id"],
                        "result":content(result,session.output,provider)})
                elif provider in ("chat", "openrouter"):
                    results.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
                    # Chat tool messages accept text; send the exact tool images in a following user message.
                    observations = content(result, session.output, provider)
                    if len(observations) > 1:
                        results.append({"role": "user", "content": [
                            {"type": "text", "text": "Images returned by tool call " + call["id"]},
                            *observations[1:]]})
                else:
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "content": content(result, session.output, provider),
                            "is_error": not result["ok"],
                        }
                    )
                if session.finished:
                    summary["status"] = "finished"
                    break
            if total_actions >= max_actions and not session.finished:
                summary["status"] = "action_budget"
            if session.finished or total_actions >= max_actions:
                break
            if provider in ("chat", "openrouter"):
                # All tool results must precede the next user observation, including when
                # a server ignores parallel_tool_calls=False and returns multiple calls.
                history.extend(x for x in results if x["role"] == "tool")
                history.extend(x for x in results if x["role"] == "user")
            elif provider == "gemini":
                history = results
            elif provider == "responses":
                history.extend(results)
            else:
                history.append({"role": "user", "content": results})
    except Exception as exc:
        summary["status"] = "provider_or_runtime_error"
        if hasattr(exc, "provider_detail"):
            summary["provider_error"] = exc.provider_detail
        raise
    finally:
        from .usage import summarize_usage
        summary["usage_summary"] = summarize_usage(provider, summary["usage"])
        summary["new_request_usage_summary"] = summarize_usage(provider, summary["new_request_usage"])
        summary["replayed_turns"] = transport.replayed
        summary["actions"] = total_actions
        summary["elapsed_seconds"] = time.monotonic() - started
        (session.output / "harness.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    atomic_json(session.output / "recovery-state.json",
                {"status": summary["status"], "turns": summary["turns"],
                 "actions": summary["actions"]})
    return summary
