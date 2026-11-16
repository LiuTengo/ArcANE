"""Shared OpenAI structured-output call helper.

`parse_call` issues one chat-completions request with `response_format` set
to a Pydantic schema and returns the parsed instance as a dict. Models that
reject custom temperature (gpt-5.x reasoning models) are auto-detected on
first 400 and the parameter is dropped on retry.

Lives in its own module so pipeline.py and text_grounding.py share both the
implementation and the `_MODELS_REJECT_TEMPERATURE` cache — one rejection
teaches every stage.
"""

import asyncio
import json
import logging
import re
from typing import Type

from openai import AsyncOpenAI
from pydantic import BaseModel

from .config import MAX_RETRIES, RETRY_BACKOFF_BASE

logger = logging.getLogger(__name__)

_MODELS_REJECT_TEMPERATURE: set[str] = set()
# model -> "schema" | "object" | "none"; unset models default to "schema".
_JSON_MODE: dict[str, str] = {}

def _json_schema_prompt(schema: Type[BaseModel]) -> str:
    return (
        "\n\nReturn only a single valid JSON object that conforms to this "
        "JSON Schema. Do not wrap it in markdown fences or add commentary.\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )


def _extract_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating markdown fences."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


async def _parse_json_mode(client: AsyncOpenAI, model: str, system: str,
                           user: str, schema: Type[BaseModel],
                           temperature: float | None, mode: str) -> dict:
    """Fallback for gateways without json_schema structured outputs."""
    system_msg = system.rstrip() + _json_schema_prompt(schema)
    base = {"model": model}
    if mode == "object":
        base["response_format"] = {"type": "json_object"}
    if temperature is not None:
        base["temperature"] = temperature

    async def request(user_content: str) -> str:
        kwargs = dict(base)
        kwargs["messages"] = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_content},
        ]
        resp = await client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):
            raise RuntimeError(f"model refused: {msg.refusal}")
        return msg.content or ""

    content = await request(user)
    try:
        data = _extract_json_object(content)
        return schema.model_validate(data).model_dump()
    except ValueError as first_err:
        # One repair pass with the invalid output and concrete schema errors.
        repair = (
            user
            + "\n\nYour previous response was:\n" + content
            + "\n\nIt failed validation with these errors:\n" + str(first_err)
            + "\n\nReturn ONLY a corrected JSON object that satisfies the "
              "JSON Schema above."
        )
        content = await request(repair)
        data = _extract_json_object(content)
        return schema.model_validate(data).model_dump()


async def parse_call(client: AsyncOpenAI, sem: asyncio.Semaphore,
                     model: str, system: str, user: str,
                     schema: Type[BaseModel], temperature: float) -> dict:
    cur_temp = None if model in _MODELS_REJECT_TEMPERATURE else temperature
    mode = _JSON_MODE.get(model, "schema")
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                if mode != "schema":
                    return await _parse_json_mode(
                        client, model, system, user, schema, cur_temp, mode)
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "response_format": schema,
                }
                if cur_temp is not None:
                    kwargs["temperature"] = cur_temp
                resp = await client.beta.chat.completions.parse(**kwargs)
            msg = resp.choices[0].message
            if getattr(msg, "refusal", None):
                raise RuntimeError(f"model refused: {msg.refusal}")
            parsed = msg.parsed
            if parsed is None:
                raise RuntimeError("structured-output parse returned None")
            return parsed.model_dump()
        except Exception as e:
            err = str(e)
            low = err.lower()
            if "temperature" in low and cur_temp is not None:
                _MODELS_REJECT_TEMPERATURE.add(model)
                logger.info(f"[api] {model} rejects custom temperature; dropping")
                cur_temp = None
                continue
            if (mode == "schema" and "response_format" in low
                    and ("unavailable" in low or "not support" in low
                         or "unsupported" in low)):
                mode = "object"
                _JSON_MODE[model] = mode
                logger.info(
                    f"[api] {model} rejects json_schema response_format; "
                    "falling back to json_object")
                continue
            if (mode == "object" and "response_format" in low
                    and ("unavailable" in low or "not support" in low
                         or "unsupported" in low)):
                mode = "none"
                _JSON_MODE[model] = mode
                logger.info(
                    f"[api] {model} rejects json_object response_format; "
                    "falling back to plain text + local JSON parsing")
                continue
            wait = RETRY_BACKOFF_BASE ** (attempt + 1)
            logger.warning(f"[api] error attempt {attempt+1}: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(wait)
            else:
                raise
