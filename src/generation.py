from typing import Any
    
import os
import json
import asyncio
import random
from typing import Any, Dict, List, Optional

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None  # type: ignore

try:
    import anthropic
except Exception:
    anthropic = None  # type: ignore



async def generate_response(model: str, prompt_data: dict, max_retries: int = 5) -> dict:
    messages = prompt_data.get("messages", [])
    temperature = prompt_data.get("temperature", 0.9)
    max_tokens = prompt_data.get("max_tokens", 384)
    top_p = prompt_data.get("top_p", 0.95)

    return await generate_chat(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        max_retries=max_retries,
    )

DEEPINFRA_MODELS = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-235B-A22B",
    "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/DeepSeek-V3.1",
    "deepseek-ai/DeepSeek-R1-0528",
}

CLAUDE_MODELS = {
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-sonnet-4-0",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-haiku-4-5-20251001",
}


_openai_client: Optional[Any] = None
_anthropic_client: Optional[Any] = None


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found")
        if AsyncOpenAI is None:
            raise RuntimeError("openai package not available or incompatible version")
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


def _get_anthropic_client() -> Any:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not found")
        if anthropic is None:
            raise RuntimeError("anthropic package not available or incompatible version")
        _anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
    return _anthropic_client


async def _generate_with_openai(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    max_retries: int = 5,
) -> str:
    client = _get_openai_client()

    # GPT-5.x and o-series use max_completion_tokens and ignore temperature/top_p.
    is_new_openai = model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4")

    last_error = None
    for attempt in range(max_retries):
        try:
            if is_new_openai:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                )
            else:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )

            content = resp.choices[0].message.content or ""
            return content
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            
            # Check if it's a rate limit error (429)
            if "429" in error_str or "rate_limit_error" in error_str or "Rate limit" in error_str:
                if attempt < max_retries - 1:
                    # Always use exponential backoff, ignore Retry-After header
                    # OpenAI: Longer wait times (20s → 40s → 80s → 160s → 320s) with significant variation
                    base_wait = 20 * (2 ** attempt)
                    jitter = random.uniform(5, 15)  # 5-15 second variation
                    wait_time = base_wait + jitter
                    print(f"OpenAI rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                    
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"OpenAI rate limit exceeded after {max_retries} retries")
                    raise e
            else:
                # For non-rate-limit errors, don't retry
                raise e
    
    # This should never be reached, but just in case
    raise last_error or RuntimeError("Max retries exceeded")


async def _generate_with_anthropic(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    max_retries: int = 5,
) -> str:
    client = _get_anthropic_client()
    
    # Convert OpenAI format messages to Anthropic format
    system_message = ""
    anthropic_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_message = msg["content"]
        else:
            anthropic_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    last_error = None
    for attempt in range(max_retries):
        try:
            # Newer Anthropic models reject simultaneous temperature + top_p.
            # Keep temperature (the spec axis) and drop top_p.
            resp = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message if system_message else None,
                messages=anthropic_messages,
            )
            
            content = resp.content[0].text if resp.content else ""
            return content
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            
            # Check if it's a rate limit error (429)
            if "429" in error_str or "rate_limit_error" in error_str:
                if attempt < max_retries - 1:
                    # Always use exponential backoff, ignore Retry-After header
                    # Anthropic has stricter rate limits, so use longer wait times with more variation
                    # Start with 30 seconds, then 60, 120, 240, 480s with large jitter
                    base_wait = 30 * (2 ** attempt)
                    jitter = random.uniform(10, 30)  # Much larger jitter (10-30 seconds)
                    wait_time = base_wait + jitter
                    print(f"Anthropic rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                    
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"Anthropic rate limit exceeded after {max_retries} retries")
                    raise e
            else:
                # For non-rate-limit errors, don't retry
                raise e
    
    # This should never be reached, but just in case
    raise last_error or RuntimeError("Max retries exceeded")


async def _generate_with_deepinfra(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    max_retries: int = 5,
) -> str:
    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        raise ValueError("DEEPINFRA_API_KEY environment variable not found")

    url = "https://api.deepinfra.com/v1/openai/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    
    # Add reasoning_effort for reasoning models (Turbo and GPT-OSS models)
    if "R1" in model or "gpt-oss" in model.lower():
        payload["reasoning_effort"] = "low"

    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        url,
        "-H",
        f"Authorization: Bearer {api_key}",
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(payload),
    ]

    last_error = None
    for attempt in range(max_retries):
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                error_msg = stderr.decode('utf-8', 'ignore')
                # Check if it's a rate limit error
                if "429" in error_msg or "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                    if attempt < max_retries - 1:
                        # Always use exponential backoff, ignore Retry-After header
                        # DeepInfra: Longer wait times (15s → 30s → 60s → 120s → 240s) with variation
                        base_wait = 15 * (2 ** attempt)
                        jitter = random.uniform(3, 10)  # 3-10 second variation
                        wait_time = base_wait + jitter
                        print(f"DeepInfra rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                        
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print(f"DeepInfra rate limit exceeded after {max_retries} retries")
                        raise RuntimeError(f"DeepInfra curl failed: {error_msg}")
                else:
                    raise RuntimeError(f"DeepInfra curl failed: {error_msg}")
            
            data = json.loads(stdout.decode("utf-8"))
            content = data["choices"][0]["message"]["content"] or ""
            
            # Append reasoning content for models that support it
            if "reasoning_content" in data["choices"][0]["message"]:
                reasoning = data["choices"][0]["message"]["reasoning_content"]
                if reasoning:
                    content = f"[REASONING]\n{reasoning}\n[REASONING]\n\n{content}"
            
            return content
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            
            # Check if it's a rate limit error in the response data
            if "429" in error_str or "rate limit" in error_str.lower():
                if attempt < max_retries - 1:
                    # Fallback to exponential backoff for exception-based rate limits
                    # DeepInfra: Longer wait times (15s → 30s → 60s) with variation
                    base_wait = 15 * (2 ** attempt)
                    jitter = random.uniform(3, 10)  # 3-10 second variation
                    wait_time = base_wait + jitter
                    print(f"DeepInfra rate limit hit (exception), waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"DeepInfra rate limit exceeded after {max_retries} retries")
                    raise e
            else:
                # For non-rate-limit errors, don't retry
                raise e
    
    # This should never be reached, but just in case
    raise last_error or RuntimeError("Max retries exceeded")


async def generate_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.9,
    max_tokens: int = 384,
    top_p: float = 0.95,
    max_retries: int = 5,
) -> Dict[str, Any]:
    last_error: Optional[str] = None
    for attempt in range(max_retries):
        try:
            if model in DEEPINFRA_MODELS:
                content = await _generate_with_deepinfra(
                    model, messages, temperature, max_tokens, top_p, max_retries
                )
            elif model in CLAUDE_MODELS:
                content = await _generate_with_anthropic(
                    model, messages, temperature, max_tokens, top_p, max_retries
                )
            else:
                content = await _generate_with_openai(
                    model, messages, temperature, max_tokens, top_p, max_retries
                )
            return {
                "response": content,
                "model": model,
                "success": True,
                "attempt": attempt + 1,
            }
        except Exception as e:
            last_error = str(e)
            # Exponential backoff: 1s, 2s, 4s, 8s, 16s
            wait_time = 1.0 * (2 ** attempt)
            await asyncio.sleep(wait_time)

    return {
        "response": None,
        "model": model,
        "error": last_error or "Max retries exceeded",
        "success": False,
    }



