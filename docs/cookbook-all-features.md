# lm15 Cookbook: Using All 69 Features (Runnable)

This document is designed to be run as a notebook or script (e.g., using a tool like Quarto or by copying into Jupyter). It demonstrates how to use the 69 cross-SDK features supported by `lm15-python2` across OpenAI, Anthropic, and Gemini.

## Setup & Initialization

First, initialize the providers. This notebook looks for `.env` in the current working directory and its parents, so it works whether you run it from the repo root or from `lm15-python2/docs/`. We'll also define a handy `execute()` function that runs a request across all three providers (and lets us toggle between streaming and completion).

```python
import os
import shlex
import pprint
import dataclasses
from pathlib import Path
from lm15.providers import OpenAILM, AnthropicLM, GeminiLM
from lm15.types import StreamDeltaEvent, TextDelta, ThinkingDelta

# 1. Load keys from the repo-root .env file, independent of notebook cwd.
def find_dotenv(filename=".env"):
    start = Path.cwd().resolve()
    candidates = [start, *start.parents]

    # If your runner executes from lm15-python2/docs, this catches the repo root.
    try:
        doc_dir = Path("lm15-python2/docs").resolve()
        candidates.extend([doc_dir, *doc_dir.parents])
    except Exception:
        pass

    seen = set()
    for directory in candidates:
        if directory in seen:
            continue
        seen.add(directory)
        path = directory / filename
        if path.exists():
            return path
    return None


def load_env_file(path, *, override=True):
    """Load shell-style .env lines like: export OPENAI_API_KEY="sk-..."."""
    if path is None:
        return {}

    loaded = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        # shlex handles quotes and shell-style escaping robustly.
        try:
            parsed = shlex.split(value, posix=True)
            value = parsed[0] if parsed else ""
        except ValueError:
            value = value.strip('"\'')

        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = os.environ.get(key, "")
    return loaded


def mask(value):
    if not value:
        return "missing"
    if len(value) <= 12:
        return "set"
    return f"{value[:7]}...{value[-4:]}"


env_path = find_dotenv()
loaded = load_env_file(env_path)
print(f"Loaded .env from: {env_path or 'not found'}")
for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
    print(f"  {key}: {mask(os.environ.get(key))}")

# 2. Initialize the three major providers.
lms = {
    "openai": OpenAILM(api_key=os.environ.get("OPENAI_API_KEY", "")),
    "anthropic": AnthropicLM(api_key=os.environ.get("ANTHROPIC_API_KEY", "")),
    "gemini": GeminiLM(api_key=os.environ.get("GEMINI_API_KEY", "")),
}

# 3. Map default models for each provider.
# These are the models used by the cross-SDK curl fixture suite.
# You can replace them with newer models after confirming your account has access.
MODELS = {
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-sonnet-4-5",
    "gemini": "gemini-3-flash-preview",
}

def execute(req, providers=None, stream=None):
    """Run a request and collect all provider results.

    Args:
        req: The lm15 Request to run.
        providers: Optional iterable of provider names.
        stream: True for streaming only, False for complete() only, None for both.
    """
    providers = list(providers or lms.keys())
    stream_modes = (False, True) if stream is None else (stream,)
    responses = []
    streams = []
    errors = []

    for name in providers:
        required_key = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }[name]

        if not os.environ.get(required_key):
            errors.append({
                "provider": name,
                "error": f"Skipped: {required_key} is not set",
            })
            continue

        lm = lms[name]

        # Replace the placeholder model in the request with the provider-specific model.
        provider_req = dataclasses.replace(req, model=MODELS[name])

        for is_stream in stream_modes:
            try:
                if is_stream:
                    streams.append({
                        "provider": name,
                        "model": MODELS[name],
                        "events": list(lm.stream(provider_req)),
                    })
                else:
                    responses.append({
                        "provider": name,
                        "model": MODELS[name],
                        "response": lm.complete(provider_req),
                    })
            except Exception as e:
                errors.append({
                    "provider": name,
                    "model": MODELS[name],
                    "stream": is_stream,
                    "error": f"{type(e).__name__}: {e}",
                })

    return {"responses": responses, "streams": streams, "errors": errors}


def pretty(value, *, width=100):
    """Pretty-print nested execute() results in notebooks/scripts."""
    pprint.pp(value, width=width, sort_dicts=False, compact=False)
```
```output
Loaded .env from: /home/maxime/Projects/lm15-dev/.env
  OPENAI_API_KEY: sk-proj...mtMA
  ANTHROPIC_API_KEY: sk-ant-...awAA
  GEMINI_API_KEY: AIzaSyB...nH1w


⏳ running…
```

---

## 1. Chat & Multi-turn
**(Features: `basic_text`, `multi_turn`)**

Basic text generation and multi-turn conversations use `Message.user` and `Message.assistant` inside a `Request`.

```python ✓
from lm15.types import Request, Message
```

```
# basic_text
req_basic = Request(
    model="placeholder", 
    messages=(Message.user("Say hello."),)
)
r = execute(req_basic)
pretty(r)
```
```output | ✓ 7.8s | 27 vars
{'responses': [{'provider': 'openai',
                'model': 'gpt-5.4-mini',
                'response': Response(
    text='Hello!',
    model='gpt-5.4-mini-2026-03-17',
    finish_reason='stop',
    usage=Usage(input_tokens=9, output_tokens=6, total_tokens=15, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_0d023e6760ec43900069f219b83120819c9739094737beb9ec',
    provider_data=<dict: 35 keys>,
)},
               {'provider': 'anthropic',
                'model': 'claude-sonnet-4-5',
                'response': Response(
    text='Hello! How can I help you today?',
    model='claude-sonnet-4-5-20250929',
    finish_reason='stop',
    usage=Usage(input_tokens=10, output_tokens=12, total_tokens=22, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_01VMNkrriLtrTn9eGLopJ4eX',
    provider_data=<dict: 9 keys>,
)},
               {'provider': 'gemini',
                'model': 'gemini-3-flash-preview',
                'response': Response(
    text='Hello! How can I help you today?',
    model='gemini-3-flash-preview',
    finish_reason='stop',
    usage=Usage(input_tokens=4, output_tokens=9, total_tokens=93, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=80, input_audio_tokens=None, output_audio_tokens=None),
    id='vhnyaaTiNLPf_uMP6NO-kQg',
    provider_data=<dict: 4 keys>,
)}],
 'streams': [{'provider': 'openai',
              'model': 'gpt-5.4-mini',
              'events': [StreamStartEvent(id='resp_0b8614212067d7f60069f219b8ff24819dab2df73cd22effec',
                                          model='gpt-5.4-mini-2026-03-17',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='Hello', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='!', part_index=0, type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=9, output_tokens=6, total_tokens=15, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 34 keys>,
    type='end',
)]},
             {'provider': 'anthropic',
              'model': 'claude-sonnet-4-5',
              'events': [StreamStartEvent(id='msg_016d6G3PpU3aoqJUx4CeMFPK',
                                          model='claude-sonnet-4-5-20250929',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='Hello! How',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' can I help you today?',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    type='end',
)]},
             {'provider': 'gemini',
              'model': 'gemini-3-flash-preview',
              'events': [StreamDeltaEvent(delta=TextDelta(text='Hello! How can I help',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' you today?',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='', part_index=0, type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=4, output_tokens=9, total_tokens=54, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=41, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 4 keys>,
    type='end',
)]}],
 'errors': []}
```

```py
# multi_turn
req_multi = Request(
    model="placeholder",
    messages=(
        Message.user("What is 2 + 2? Reply with one word."),
        Message.assistant("four"),
        Message.user("Repeat your previous answer in uppercase."),
    )
)
r = execute(req_multi)
pretty(r)
```
```output
{'responses': [{'provider': 'openai',
                'model': 'gpt-5.4-mini',
                'response': Response(
    text='FOUR',
    model='gpt-5.4-mini-2026-03-17',
    finish_reason='stop',
    usage=Usage(input_tokens=37, output_tokens=6, total_tokens=43, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_08a35d847d09e2620069f219c225788196b973500e06ae1de6',
    provider_data=<dict: 35 keys>,
)},
               {'provider': 'anthropic',
                'model': 'claude-sonnet-4-5',
                'response': Response(
    text='FOUR',
    model='claude-sonnet-4-5-20250929',
    finish_reason='stop',
    usage=Usage(input_tokens=36, output_tokens=5, total_tokens=41, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_0146dnoWm2HgbphEDLLmqVW7',
    provider_data=<dict: 9 keys>,
)},
               {'provider': 'gemini',
                'model': 'gemini-3-flash-preview',
                'response': Response(
    text='FOUR',
    model='gemini-3-flash-preview',
    finish_reason='stop',
    usage=Usage(input_tokens=24, output_tokens=1, total_tokens=98, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=73, input_audio_tokens=None, output_audio_tokens=None),
    id='zRnyaZijEMmt1MkPwI2LsQk',
    provider_data=<dict: 4 keys>,
)}],
 'streams': [{'provider': 'openai',
              'model': 'gpt-5.4-mini',
              'events': [StreamStartEvent(id='resp_0b28cc15ee52b9230069f219c4473881978329ed80ac49680e',
                                          model='gpt-5.4-mini-2026-03-17',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='FO', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='UR', part_index=0, type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=37, output_tokens=6, total_tokens=43, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 34 keys>,
    type='end',
)]},
             {'provider': 'anthropic',
              'model': 'claude-sonnet-4-5',
              'events': [StreamStartEvent(id='msg_01HPBwwcK3ESBrmrNzRXuqgM',
                                          model='claude-sonnet-4-5-20250929',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='FOUR', part_index=0, type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    type='end',
)]},
             {'provider': 'gemini',
              'model': 'gemini-3-flash-preview',
              'events': [StreamDeltaEvent(delta=TextDelta(text='FOUR', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='', part_index=0, type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=24, output_tokens=1, total_tokens=146, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=121, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 4 keys>,
    type='end',
)]}],
 'errors': []}


⏳ running…
```

```py
# multi_turn
req_assistant_prefill = Request(
    model="placeholder",
    messages=(
        Message.user("What is 2 + 2? Reply with one word."),
        Message.assistant("As a pirate I should say")
    )
)
r = execute(req_assistant_prefill)
pretty(r)
```
```output | ✓ 64.6s | 27 vars
{'responses': [{'provider': 'openai',
                'model': 'gpt-5.4-mini',
                'response': Response(
    text='4',
    model='gpt-5.4-mini-2026-03-17',
    finish_reason='stop',
    usage=Usage(input_tokens=31, output_tokens=11, total_tokens=42, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_03cdb6d734eecac20069f219f600c48193b86ef137b1eba7a2',
    provider_data=<dict: 35 keys>,
)},
               {'provider': 'anthropic',
                'model': 'claude-sonnet-4-5',
                'response': Response(
    text=' "Four"',
    model='claude-sonnet-4-5-20250929',
    finish_reason='stop',
    usage=Usage(input_tokens=28, output_tokens=6, total_tokens=34, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_017MXinooXWrgQVGyxmthwRu',
    provider_data=<dict: 9 keys>,
)}],
 'streams': [{'provider': 'openai',
              'model': 'gpt-5.4-mini',
              'events': [StreamStartEvent(id='resp_07c7c8dcb07a18f90069f219f6f49881a2affbd050435766f2',
                                          model='gpt-5.4-mini-2026-03-17',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='4', part_index=0, type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=31, output_tokens=11, total_tokens=42, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 34 keys>,
    type='end',
)]},
             {'provider': 'anthropic',
              'model': 'claude-sonnet-4-5',
              'events': [StreamStartEvent(id='msg_01JhJQr4TqYAXt85JHFMVGqd',
                                          model='claude-sonnet-4-5-20250929',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text=':', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='\n\nFour',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    type='end',
)]},
             {'provider': 'gemini',
              'model': 'gemini-3-flash-preview',
              'events': [StreamDeltaEvent(delta=TextDelta(text=': four!',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' (Arrr!)',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='\n', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='', part_index=0, type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=21, output_tokens=8, total_tokens=29, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 4 keys>,
    type='end',
)]}],
 'errors': [{'provider': 'gemini',
             'model': 'gemini-3-flash-preview',
             'stream': False,
             'error': 'TransportError: read timed out waiting for headers: The read operation '
                      'timed out'}]}
```

---

## 2. System Prompts
**(Features: `system_prompt`)**

System prompts establish the behavior of the model.

```python
req_system = Request(
    model="placeholder",
    system="You are an angry pirate.",
    messages=(Message.user("How are you today?"),)
)
pretty(execute(req_system))
```
```output | ✓ 20.0s | 24 vars
{'responses': [{'provider': 'openai',
                'model': 'gpt-5.4-mini',
                'response': Response(
    text='Arrr, I’m doin’ fine enough, matey—ready to help ye with whatever be on yer mind.',
    model='gpt-5.4-mini-2026-03-17',
    finish_reason='stop',
    usage=Usage(input_tokens=21, output_tokens=29, total_tokens=50, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_0793609c529c78f10069f23eda943c81928b67f72f53fa3a0c',
    provider_data=<dict: 35 keys>,
)},
               {'provider': 'anthropic',
                'model': 'claude-sonnet-4-5',
                'response': Response(
    text="Arrr, I be in a FOUL mood, ye scurvy dog! \n\nThe seas have been cruel, me grog ration's been cut, and some bilge rat made off with me favorite cutlass! Me ship's got barnacles thick as me beard, the crew's been nothin' but mutinous scallywags, and don't even get me started on the blasted parrot that won't stop squawkin' about crackers!\n\nWhat be YE wantin', eh? Speak up before I lose what little patience I got left! *slams fist on table* \n\nARRRGHHH! 🏴\u200d☠️",
    model='claude-sonnet-4-5-20250929',
    finish_reason='stop',
    usage=Usage(input_tokens=19, output_tokens=153, total_tokens=172, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_019dk86YrReG28yC4eFPXxJJ',
    provider_data=<dict: 9 keys>,
)},
               {'provider': 'gemini',
                'model': 'gemini-3-flash-preview',
                'response': Response(
    text="HOW AM I?! HOW DO YE THINK I AM, YE BILGE-SUCKING LANDLUBBER?! \n\nMe boots be full o' seawater, me first mate is a half-witted barnacle who couldn't find his own backside with both hands and a lantern, and some scurvy dog drank the last of me grog! I’ve got a splinter the size of a harpoon in me wooden leg, and the wind is blowin' the wrong way!\n\nWhy are ye standin' there flappin' yer gums?! Unless ye got a map to some buried gold or a bottle o' rum, get off me deck before I keelhaul ye and feed what’s left to the sharks! ARRRRRGGGH!",
    model='gemini-3-flash-preview',
    finish_reason='stop',
    usage=Usage(input_tokens=12, output_tokens=156, total_tokens=509, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=341, input_audio_tokens=None, output_audio_tokens=None),
    id='6j7yaZa7IPqq1MkPsfO06Qk',
    provider_data=<dict: 4 keys>,
)}],
 'streams': [{'provider': 'openai',
              'model': 'gpt-5.4-mini',
              'events': [StreamStartEvent(id='resp_0553f2d5357641170069f23edda72c81909a8e10afa8a772aa',
                                          model='gpt-5.4-mini-2026-03-17',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='Arr', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='r', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=',', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' I', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' be', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' do', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='in', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='’', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' well', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' enough',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=',', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' mate', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='y', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='—', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='ready', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' to', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' sail', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' through',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' any', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' question',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' ye', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='’ve', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' got', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='.', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' How', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' can', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' I', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' help', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' ye', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' today', part_index=0, type='text'),
… 97 more lines
```

---

## 3. Model Configuration
**(Features: `temperature`, `max_tokens`, `max_output_tokens`, `top_p`, `top_k`, `stop_sequences`)**

Universal generation knobs live in `Config`. `lm15` abstracts away provider-specific names (like `max_output_tokens` vs `max_tokens`).

```python
from lm15.types import Config

req_config = Request(
    model="placeholder",
    messages=(Message.user("Write a haiku about the ocean."),),
    config=Config(
        temperature=0.7,
        max_tokens=50,       # Maps to maxOutputTokens in Gemini natively
        #top_p=0.8,
        top_k=10,            # Some providers may ignore this if unsupported natively
        stop=("water", "Water")  # Stop sequences
    )
)
pretty(execute(req_config))
```
```output | ✓ 10.1s | 26 vars
{'responses': [{'provider': 'openai',
                'model': 'gpt-5.4-mini',
                'response': Response(
    text='Salt breath on blue waves  \nMoonlight combs the restless deep  \nShore listens, still, soft',
    model='gpt-5.4-mini-2026-03-17',
    finish_reason='stop',
    usage=Usage(input_tokens=14, output_tokens=25, total_tokens=39, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_06c9527855f0fbd20069f23f565f3881959a533d5aaf028881',
    provider_data=<dict: 35 keys>,
)},
               {'provider': 'anthropic',
                'model': 'claude-sonnet-4-5',
                'response': Response(
    text='Waves kiss distant shore\nSalt spray dancing in the wind\nDeep blue calls me home',
    model='claude-sonnet-4-5-20250929',
    finish_reason='stop',
    usage=Usage(input_tokens=15, output_tokens=21, total_tokens=36, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_01V61e3CFDR4bYFLDvcdcHw1',
    provider_data=<dict: 9 keys>,
)},
               {'provider': 'gemini',
                'model': 'gemini-3-flash-preview',
                'response': Response(
    text='Blue waves',
    model='gemini-3-flash-preview',
    finish_reason='length',
    usage=Usage(input_tokens=9, output_tokens=2, total_tokens=55, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=44, input_audio_tokens=None, output_audio_tokens=None),
    id='Xz_yaaHjGd2h1MkP0Z-2kA8',
    provider_data=<dict: 4 keys>,
)}],
 'streams': [{'provider': 'openai',
              'model': 'gpt-5.4-mini',
              'events': [StreamStartEvent(id='resp_09d15fd09d08e2740069f23f5853bc8191b6acacabd60bc791',
                                          model='gpt-5.4-mini-2026-03-17',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='Blue', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' waves', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' breathe',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' and', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' rise', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='  \n', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='Salt', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' wind', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' whispers',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' through',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' the', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' foam', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='  \n', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='Moon', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='light', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' holds', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' the', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' sea', part_index=0, type='text'),
                                          type='delta'),
                         StreamEndEvent(
    finish_reason='stop',
    usage=Usage(input_tokens=14, output_tokens=22, total_tokens=36, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    provider_data=<dict: 34 keys>,
    type='end',
)]},
             {'provider': 'anthropic',
              'model': 'claude-sonnet-4-5',
              'events': [StreamStartEvent(id='msg_0178wfFRTsaFrazpbpuTAFAY',
                                          model='claude-sonnet-4-5-20250929',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='Waves', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' kiss distant shore\n'
                                                               'Salt spray dances with the wind\n'
                                                               'Deep blue holds',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' secrets',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
… 17 more lines
```

---

## 4. Media & Files
**(Features: `image_url`, `image_file`, `image_base64`, `image_inline`, `image_file_uri`, `audio_inline`, `video_file`, `file_input`, `pdf_base64`, `pdf_inline`)**

`lm15` media part factories (`image`, `audio`, `video`, `document`, `binary`) abstract how media is passed to the provider. You can pass a path, raw bytes, or a URL.

*Note: For a cross-provider runnable cookbook, we fetch this public image ourselves and send it inline. Provider-side URL fetching can fail when the origin blocks provider fetchers or requires a browser-like/user-agent request. Local paths work seamlessly via `image(path="./img.png")`.*

```python
import urllib.request
from lm15.types import image, text

IMAGE_URL = "https://www.gstatic.com/webp/gallery/1.jpg"
image_request = urllib.request.Request(
    IMAGE_URL,
    headers={"User-Agent": "lm15-cookbook/1.0"},
)
with urllib.request.urlopen(image_request, timeout=15) as response:
    image_bytes = response.read()
    image_media_type = response.headers.get_content_type()

req_media = Request(
    model="placeholder",
    messages=(
        Message.user([
            text("Describe this image:"),
            # image_base64 / image_inline
            image(data=image_bytes, media_type=image_media_type, detail="low"),
        ]),
    )
)
pretty(execute(req_media))
```
```output | ✓ 40.1s | 35 vars
{'responses': [{'provider': 'openai',
                'model': 'gpt-5.4-mini',
                'response': Response(
    text='A dramatic mountain landscape with a deep fjord or river valley running through the center. Steep, rugged cliffs rise on both sides, covered in green vegetation and dark rocky slopes. In the foreground, there’s a sharp rocky outcrop overlooking the view. The water below winds into the distance toward the horizon under a bright blue sky with thin clouds.',
    model='gpt-5.4-mini-2026-03-17',
    finish_reason='stop',
    usage=Usage(input_tokens=270, output_tokens=74, total_tokens=344, cache_read_tokens=0, cache_write_tokens=None, reasoning_tokens=0, input_audio_tokens=None, output_audio_tokens=None),
    id='resp_0ed8b9d1224b8d2b0069f2594ca3648191aefeac9bcaeda440',
    provider_data=<dict: 35 keys>,
)},
               {'provider': 'anthropic',
                'model': 'claude-sonnet-4-5',
                'response': Response(
    text="# Image Description\n\nThis breathtaking photograph captures a dramatic **Norwegian fjord landscape** from an elevated viewpoint. The composition features:\n\n## Foreground\n- A **rocky outcrop** in the lower left, likely the vantage point from which the photo was taken\n- Patches of **snow** visible on the dark rocks\n\n## Main Valley\n- A deep, **U-shaped glacial valley** carved between towering mountains\n- A **narrow fjord or lake** snaking through the valley floor, its dark blue waters creating a striking contrast\n- Small patches of **bright green vegetation** along the water's edge\n\n## Mountains\n- **Steep-sided mountains** rising dramatically on both sides of the valley\n- Slopes covered in **green vegetation** at lower elevations\n- **Snow-capped peaks** visible in the distance\n- The distinctive steep cliff face on the right side of the valley\n\n## Sky\n- A **bright blue sky** with wispy white clouds\n- Atmospheric haze creating layers of depth in the distant mountains\n\nThe image exemplifies classic **Scandinavian fjord scenery**, possibly from locations like Geirangerfjord or similar Norwegian landscapes, showcasing the dramatic topography created by ancient glacial activity.",
    model='claude-sonnet-4-5-20250929',
    finish_reason='stop',
    usage=Usage(input_tokens=295, output_tokens=272, total_tokens=567, cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=None, input_audio_tokens=None, output_audio_tokens=None),
    id='msg_01HJC1fjbsbnW35u7xUJqdAz',
    provider_data=<dict: 9 keys>,
)},
               {'provider': 'gemini',
                'model': 'gemini-3-flash-preview',
                'response': Response(
    text='This breathtaking high-angle landscape photograph captures a vast, deep valley or fjord carved between soaring, steep-sided mountains. \n\nIn the foreground, the edge of a rugged, jagged rock formation anchors the left side of the frame, giving the viewer a sense of standing on a high mountain summit. The weathered rock shows patches of grey and white, suggesting mineral deposits or lichen.\n\nThe center of the image is dominated by a long, narrow body of deep blue water that snakes through the floor of the valley. Along the thin margins of the water, small, vibrant emerald-green patches of land are visible, possibly indicating remote farmsteads or small villages nestled at the base of the cliffs.\n\nThe mountains themselves are massive and dramatic. Their lower slopes are covered in dark green vegetation, which gradually gives way to bare, grey rock faces as they reach toward the sky. The sunlight hits the scene from the upper right, casting the left side of the valley into deep shadow while brightly illuminating the right-hand slopes, highlighting their textured ridges and crevasses.\n\nIn the far distance, more mountain ranges fade into a hazy blue, with some of the highest peaks showing small, lingering patches of white snow. Above it all is a bright, clear blue sky streaked with thin, wispy horizontal clouds. The overall impression is one of immense scale, quiet majesty, and the wild beauty of a Nordic landscape.',
    model='gemini-3-flash-preview',
    finish_reason='stop',
    usage=Usage(input_tokens=1085, output_tokens=284, total_tokens=2076, cache_read_tokens=None, cache_write_tokens=None, reasoning_tokens=707, input_audio_tokens=None, output_audio_tokens=None),
    id='bVnyaeHHCbjf_uMP94GWiQ0',
    provider_data=<dict: 4 keys>,
)}],
 'streams': [{'provider': 'openai',
              'model': 'gpt-5.4-mini',
              'events': [StreamStartEvent(id='resp_0613986c436706c80069f25950ecd48191b9ef4b56b072f9e3',
                                          model='gpt-5.4-mini-2026-03-17',
                                          type='start'),
                         StreamDeltaEvent(delta=TextDelta(text='A', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' dramatic',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' mountain',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' valley',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' with', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' steep', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=',', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' rugged',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' slopes',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' surrounding',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' a', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' long', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=',', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' narrow',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' blue', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' fj', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='ord', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' or', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' river', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='.', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' In', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' the', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' foreground',
                                                          part_index=0,
                                                          type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text=' there', part_index=0, type='text'),
                                          type='delta'),
                         StreamDeltaEvent(delta=TextDelta(text='’s', part_index=0, type='text'),
… 317 more lines
```

---

## 5. Tools & Function Calling
**(Features: `tools`, `multi_turn_tool_result`, `multi_turn_function_response`)**

Define tools natively and feed results back in a `Message.tool`.

```python
from lm15.types import FunctionTool

weather_tool = FunctionTool(
    name="get_weather",
    description="Get the current weather for a city",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

req_tools = Request(
    model="placeholder",
    messages=(Message.user("What is the weather in Montreal?"),),
    tools=(weather_tool,)
)

# Run complete to see tool calls parsed into Python objects
execute(req_tools, stream=False)
```

---

## 6. Tool Choice & Constraints
**(Features: `tool_choice_auto`, `tool_choice_required`, `tool_choice_none`, `tool_choice_specific`, `tool_choice_any`, `tool_config_auto`, `tool_config_any`, `tool_config_none`, `parallel_tool_calls`, `max_tool_calls`)**

Use `ToolChoice` inside `Config` to control how the model uses tools. `lm15` handles translating these into the specific provider constraints.

```python
from lm15.types import ToolChoice

# Force the model to use the 'get_weather' tool (tool_choice_specific / tool_choice_required)
# We also set parallel=False (disables parallel tool calls natively across providers)
req_forced = dataclasses.replace(
    req_tools,
    config=Config(tool_choice=ToolChoice.from_tools("get_weather", mode="required", parallel=False))
)

execute(req_forced, stream=False)
```

---

## 7. Built-in Tools
**(Features: `web_search`, `code_interpreter`, `container`, `google_search`, `code_execution`)**

`lm15` translates generic built-in tools to the exact provider-native string representations (e.g., `googleSearch` for Gemini, `web_search_20250305` for Anthropic).

```python
from lm15.types import BuiltinTool

search_tool = BuiltinTool("web_search")

req_builtin = Request(
    model="placeholder",
    messages=(Message.user("What happened in the news today?"),),
    tools=(search_tool,)
)

# Let's run this specifically on Gemini and Anthropic
# (OpenAI requires a specific model mapping for its web_search)
execute(req_builtin, stream=True, providers=["gemini", "anthropic"])
```

---

## 8. Structured Output
**(Features: `structured_output`, `structured_output_json_object`, `response_mime_type`, `response_schema`, `output_config`)**

Define response formats cleanly. `lm15` formats it for the specific provider (`response_format` for OpenAI, `generationConfig.responseSchema` for Gemini, `output_config` for Anthropic).

```python
recipe_schema = {
    "type": "json_schema",
    "name": "recipe",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "ingredients": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name", "ingredients"],
        "additionalProperties": False
    }
}

req_json = Request(
    model="placeholder",
    messages=(Message.user("Give me a cookie recipe."),),
    config=Config(response_format=recipe_schema)
)

execute(req_json, stream=False)
```

---

## 9. Reasoning & Thinking
**(Features: `reasoning`, `thinking`, `thinking_budget`)**

Enable models that think before they answer.

```python
from lm15.types import Reasoning

# Note: Providers require specific models for reasoning. 
# We'll test Anthropic's claude-3-7-sonnet-latest natively here.

req_reasoning = Request(
    model="claude-3-7-sonnet-latest",
    messages=(Message.user("What is 143 times 27? Think carefully."),),
    config=Config(
        reasoning=Reasoning(
            effort="high",
            thinking_budget=1024
        )
    )
)

print("\n\033[1m========== ANTHROPIC (Reasoning) ==========\033[0m")
try:
    for event in lms["anthropic"].stream(req_reasoning):
        if isinstance(event, StreamDeltaEvent):
            if isinstance(event.delta, TextDelta):
                print(event.delta.text, end="", flush=True)
            elif isinstance(event.delta, ThinkingDelta):
                # We render thoughts in gray
                print(f"\033[90m{event.delta.text}\033[0m", end="", flush=True)
    print()
except Exception as e:
    print(f"Error: {e}")
```
