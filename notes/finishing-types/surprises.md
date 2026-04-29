# Surprise Review — `lm15/types.py`

Scope: only `lm15/types.py`. Cross-references to `lm15/result.py`,
`lm15/live.py`, `lm15/protobuf.py`, `tests/test_types.py` were used to
validate hypotheses, but recommendations are for `types.py` itself.

The module is generally careful: discriminated unions, frozen dataclasses,
explicit forbidden-part tables, a `_check_streamable_partition` consistency
check at import time, and structured validation helpers. The remaining
surprises are the kind that survive once the obvious ones are gone:
contract leaks at the validation/type boundary, asymmetric checks across
sibling variants, sentinels that become observable, and quietly-mutating
"normalization" that diverges from declared types.

## Global surprise

| Area | Surprise score | Main reason |
|---|---:|---|
| Hashability of frozen dataclasses containing JsonObject | 0.85 | `frozen=True` strongly implies hashable; `_FrozenJsonObject` silently breaks it |
| `_MediaMixin.data` accepts `bytes` despite `data: str | None` annotation | 0.80 | No isinstance check; `__repr__` later crashes on the stored value |
| `FunctionTool.from_fn` schema for `Annotated[...]` and unknown types | 0.75 | Silent fallback to `{"type": "string"}` |
| Sentinel `Usage.total_tokens = -1` is observable | 0.70 | Field type is `int`, value `-1` is a magic value, not `None` |
| Type annotations vs stored canonical form (Sequence → tuple, str → tuple) | 0.65 | Field types lie about what is actually held after `__post_init__` |
| `tool_result`/`ToolResultPart.content: Sequence[Part]` but only `ToolResultContentPart` allowed | 0.65 | Type alias is broad, runtime is narrow |
| `_decode_data` runs eagerly in `__post_init__`; `bytes` property is uncached | 0.60 | Doubled work, perf footgun for large payloads |
| Hand-maintained `Literal` vocabularies vs structural tables | 0.55 | `PartType`, `DeltaType`, `FinishReason`, `ErrorCode`, `StreamEventType` not cross-checked |
| `Response.json` is a `@property` that does work and raises | 0.55 | Properties shouldn't throw or re-parse on every access |
| `ToolCallInfo` duplicates `ToolCallPart` minus `type` | 0.50 | Two types for one concept; conversion required |
| `ToolChoice.allowed` declared `Sequence[str | Tool] | Tool`, stored as `tuple[str, ...]` | 0.55 | Type-level lie + post_init mutation |
| `Reasoning(effort='off')` is the *default*; meaning of `Config(reasoning=None)` vs `Config(reasoning=Reasoning())` differs | 0.55 | Two ways to "not reason," only documented once |
| `_freeze_required_json_object`, `_freeze_optional_json_object`, `_validate_json_object` unused | 0.30 | Dead code |
| `PartType`/`FinishReason` both contain `"tool_call"` | 0.35 | Same string, different namespaces |
| Asymmetric empty-text rules: `TextPart('')`/`ThinkingPart('')` ok, `RefusalPart('')` rejected | 0.35 | Inconsistent permissiveness |
| `Config(extensions={}) != Config()` | 0.30 | None vs empty-dict not normalized |
| `ImageGenerationResponse` and `AudioGenerationResponse` carry no `model`/`usage`/`id` | 0.45 | Asymmetric vs `Response`/`EmbeddingResponse`; lose billing/identity |
| `BatchResponse.status: str` is free-form | 0.35 | No vocabulary; cross-provider semantics undefined |

---

## Highest-priority surprises

### 1. Frozen dataclasses with `JsonObject` fields are silently unhashable

- **Type:** `invariant_violation`, `hidden_contract`
- **Location:** `ToolCallPart`, `ToolResultPart` (indirectly via Part content),
  `FunctionTool`, `BuiltinTool`, `Config`, `Reasoning` (via `Config`), `Request`,
  `Response`, `StreamEvent`, `LiveConfig`, `LiveClientEvent`, `LiveServerEvent`,
  `ToolCallInfo` — anything that holds a `JsonObject`/`Extensions`/`ProviderData`.
- **Surprising spans:** every `_freeze_field(self, "...")` call.
- **Expected connection:** `@dataclass(frozen=True, slots=True)` is the
  conventional Python signal for "value object — hashable, usable as a dict
  key, comparable, suitable for sets and `lru_cache`." Module docstring
  reinforces this: *"Frozen + slotted dataclasses throughout. Objects are
  immutable after construction."*
- **Observed connection:** `_FrozenJsonObject` extends `dict` and overrides
  `__setitem__` etc., but `dict.__hash__` is still `None`. Therefore *any*
  dataclass that calls `_freeze_field` produces an instance that:
  ```python
  >>> hash(ToolCallPart(id="1", name="f", input={}))
  TypeError: unhashable type: '_FrozenJsonObject'
  ```
  It also silently propagates: `Message` containing a `ToolCallPart` is
  unhashable; `Request` containing such a `Message` is unhashable; the
  promise of "value object" partially fails depending on which part variants
  are inside.
- **Why it matters:** users will reasonably try to deduplicate tool calls,
  cache responses by `Request`, key telemetry by `ToolCallInfo`, etc.
  Today they get a runtime `TypeError` from a place that read like a value
  type. The asymmetry across part types (TextPart hashes; ToolCallPart
  doesn't) is itself confusing.
- **Confidence:** high (verified at the REPL).
- **Suggested slow action:** make `_FrozenJsonObject` and `_FrozenJsonArray`
  truly hashable by computing a hash over `(("k", v), ...)` / tuple of items
  on first access (cached). Then frozen dataclass hash works automatically.
  Alternatively: document explicitly which types are hashable and which
  are not, and add `__hash__ = None` on the unhashable ones so the failure
  is upfront, not nested.

### 2. `_MediaMixin.data: str | None` accepts `bytes` and corrupts `__repr__`

- **Type:** `type_shape_mismatch`, `confirmed bug`
- **Location:** `_MediaMixin.__post_init__` (lines ~322), `__repr__` (lines ~328),
  `_base64_summary` (lines ~302).
- **Surprising spans:**
  ```python
  data: str | None = None
  ...
  if self.data is not None:
      _decode_data(self.__class__.__name__, self.data)
  ```
  `_decode_data` calls `base64.b64decode(data, validate=True)` which accepts
  *bytes* arguments too, so passing `bytes` makes it through validation.
- **Expected connection:** field annotation `str | None`. Sibling validator
  `AudioDelta.__post_init__` does explicitly check
  `if not isinstance(self.data, str)`. A symmetric check would belong here.
- **Observed connection:**
  ```python
  >>> ImagePart(media_type="image/png", data=b"aGVsbG8=")  # passes
  >>> repr(_)
  TypeError: startswith first arg must be bytes or a tuple of bytes, not str
  ```
  The bytes form survives construction, then poisons `__repr__`,
  serialization, and any downstream code that calls `.startswith("data:")`.
- **Why it matters:** users are very likely to pass raw `bytes` because (a)
  the factory `image(data=b"...")` accepts bytes and silently encodes, (b)
  many SDKs return bytes. The Part constructor accepting bytes silently
  diverges from the factory's contract.
- **Confidence:** high.
- **Suggested slow action:** add `if self.data is not None and not
  isinstance(self.data, str): raise TypeError(...)` in `_MediaMixin.__post_init__`
  (mirroring `AudioDelta`). Also tighten `_decode_data` to require `str`.

### 3. Sentinel `total_tokens = -1` is observable, type-confusing, and brittle

- **Type:** `hidden_contract`, `name_behavior_mismatch`
- **Location:** `Usage` definition, lines ~1547-1576.
- **Surprising spans:**
  ```python
  total_tokens: int = -1
  ...
  if self.total_tokens == -1:
      object.__setattr__(self, "total_tokens", computed_total)
  ```
- **Expected connection:** `Usage` is a frozen value object; readers expect
  `total_tokens` is either a non-negative integer or `None`. The class
  comment promises "After `__post_init__` runs, `total_tokens` is always a
  non-negative `int`." That's true on the public path, but...
- **Observed connection:** `-1` becomes a *valid* construction-time value
  with magic meaning. Code paths that bypass `__post_init__` (proto
  decoding, `dataclasses.replace`, deserialization, `astuple`, fuzzers,
  third-party tools that introspect dataclass defaults) will see `-1`.
  The `dataclasses.fields(Usage)[2].default` is `-1`, not the documented
  invariant. This is also subtle on `replace`: `replace(u,
  input_tokens=10)` recomputes nothing — you may end up with
  `total_tokens != input_tokens + output_tokens` if the original was set
  to a "compute-me" sentinel (you can't reach this state because
  `__post_init__` has already run, but it's still a footgun on serialization
  layers that bypass it).
- **Why it matters:** sentinels-as-default are exactly the pattern this
  module otherwise avoids (cf. `_MISSING` for `_field_default`). Using
  `int` plus magic value is a regression in style for one class.
- **Confidence:** medium-high.
- **Suggested slow action:** make `total_tokens: int | None = None`,
  treat `None` as "compute," and store an `int` after post-init. Or drop
  the field entirely and expose it as `@property` (one source of truth).

### 4. Type annotations diverge from canonical stored shape

- **Type:** `type_shape_mismatch`, `documentation_mismatch`
- **Location:**
  - `Message.parts: Part | Sequence[Part]` → stored as `tuple[Part, ...]`
  - `Config.stop: str | Sequence[str] | None` → stored as `tuple[str, ...]`
  - `Request.messages: Sequence[Message]` → `tuple[Message, ...]`
  - `Request.tools: Sequence[Tool]` → `tuple[Tool, ...]`
  - `ToolChoice.allowed: Sequence[str | Tool] | Tool` → `tuple[str, ...]`
  - `EmbeddingRequest.inputs: str | Sequence[str]` → `tuple[str, ...]`
  - `Request.system: SystemContent | None` → after `_normalize_system`,
    `str | tuple[Part, ...] | None`
  - `LiveConfig.tools: Sequence[Tool]` → `tuple[Tool, ...]`
- **Expected connection:** field annotation = field type.
- **Observed connection:** every one of these is "input type" not "stored
  type." Static checkers will report incorrect attribute types; users
  iterating a second time think they may be iterating a generator;
  `mypy` will not flag indexing patterns; `dataclasses.replace` accepts
  the input type but stores the canonical type.
- **Why it matters:** the "value object" promise weakens. The user that
  does `cfg = replace(cfg, stop="x")` then reads `cfg.stop` finds a
  tuple, not a string. Round-tripping types through serializers also has
  to know which fields are normalized.
- **Confidence:** high.
- **Suggested slow action:**
  - Either expose a "smart constructor" pattern (factory function for
    inputs, narrow type on the dataclass), or
  - Annotate stored canonical types and make the public construction go
    through factories that accept the wider type. `Message.parts:
    tuple[Part, ...]` and `Message.user(content)` covering str input is
    already that pattern — extend it consistently.

### 5. `ToolResultPart.content: Sequence[Part]` is broader than runtime allows

- **Type:** `type_shape_mismatch`, `documentation_mismatch`
- **Location:** lines ~417-447.
- **Surprising spans:**
  ```python
  class ToolResultPart:
      content: Sequence["Part"]
      ...
      if any(isinstance(p, _TOOL_RESULT_FORBIDDEN_PARTS) for p in self.content):
          raise TypeError(...)
  ```
- **Expected connection:** A `ToolResultContentPart` alias exists exactly
  for this purpose; the field should reference it.
- **Observed connection:** the field type is `Sequence[Part]`; static
  analysis happily accepts `[ToolCallPart(...)]`; the rejection is only at
  runtime via the `_TOOL_RESULT_FORBIDDEN_PARTS` tuple.
- **Why it matters:** users guided by IDE autocomplete will assume
  arbitrary parts work. The rejection messaging is good, but the type
  system could prevent the mistake.
- **Confidence:** high.
- **Suggested slow action:** change the field annotation to
  `Sequence[ToolResultContentPart]`. Same applies symmetrically to
  `LiveClientEvent.content: Sequence[Part]`.

### 6. `FunctionTool.from_fn` silently degrades unknown annotations to `string`

- **Type:** `error_handling_gap`, `name_behavior_mismatch`
- **Location:** `_json_schema_for_annotation` final fallback (lines ~1280-1283).
- **Surprising spans:**
  ```python
  return {"type": "string"}
  ```
- **Expected connection:** "Best-effort JSON Schema for common Python
  annotations" suggests fallthrough is conservative. A more honest fallback
  would be `{}` (no constraint) or raising.
- **Observed connection:** `Annotated[int, ...]`, custom classes, pydantic
  models, dataclasses, `datetime`, `Decimal`, `Path`, etc. all become
  `{"type": "string"}` with no warning. The model receives a tool spec
  that misrepresents the parameter.
- **Why it matters:** this is the schema sent to the model for tool
  selection. Wrong types here produce wrong tool calls. Failures are
  silent and only surface when calls don't validate downstream.
  Additionally `Annotated` is not stripped to its base type — a likely
  oversight.
- **Confidence:** high (verified for `Annotated[int, "x"]`).
- **Suggested slow action:**
  - Strip `Annotated` to its inner type before dispatch (`if origin is
    Annotated: return _json_schema_for_annotation(args[0])`).
  - On unknown annotations, emit `{}` (any) rather than `string`, or
    raise / warn. At minimum document the fallback in the docstring.

### 7. `Response.json` is a `@property` that does parsing work and raises

- **Type:** `name_behavior_mismatch`, `lifecycle_resource_surprise`
- **Location:** lines ~1614-1639.
- **Surprising spans:**
  ```python
  @property
  def json(self) -> Any:
      ...
      raise ValueError(...)
  ```
- **Expected connection:** Python convention for `@property` is "fast,
  side-effect-free, doesn't raise except `AttributeError`." Users will
  call this in debuggers, in `repr`-likes, in template engines.
- **Observed connection:** every access re-runs regex + `json.loads`, and
  failure modes are `ValueError`. `repr(response)` is safe but
  `response.json` in a debugger pretty-printer can raise.
- **Why it matters:** debugging UX and surprise on every read. Caching is
  also impossible because the dataclass is `frozen=True` (no place to put
  the cached value without `slots=True` mutation tricks).
- **Confidence:** high.
- **Suggested slow action:** rename to a method `parse_json()`, optionally
  with `default=_RAISE` parameter to allow `parse_json(default=None)`. Keep
  the property only if it returns the cached parse and never raises.

### 8. Eager `_decode_data` on construction; `bytes` property re-decodes every time

- **Type:** `lifecycle_resource_surprise`
- **Location:** `_MediaMixin.__post_init__` and `_MediaMixin.bytes`.
- **Surprising spans:**
  ```python
  if self.data is not None:
      _decode_data(self.__class__.__name__, self.data)  # validate
  ...
  @property
  def bytes(self) -> bytes:
      return _decode_data(self.__class__.__name__, self.data)
  ```
- **Expected connection:** value-object construction should be cheap;
  `.bytes` reads should be cached.
- **Observed connection:** constructing an `AudioPart` with a 5MB base64
  payload took ~90ms in a quick probe (validation is a full b64 decode).
  Every `.bytes` access re-decodes (~70ms each). Adapters that pass the
  same part through several layers may re-validate and re-decode many
  times.
- **Why it matters:** invisible cost. Some pipelines construct many
  thousands of parts during streaming reassembly (`AudioDelta` → assemble
  → `AudioPart`). Multiple decodes per part add up.
- **Confidence:** medium (measured locally; varies by host).
- **Suggested slow action:**
  - Validate the *shape* of base64 data without full decode (regex on
    `^[A-Za-z0-9+/=\s]*$` plus length-mod-4 check).
  - Cache decoded `bytes` (e.g., a private slot populated lazily on first
    read). Or expose only `decode_bytes()` as a method to make cost
    explicit.

### 9. `ToolChoice.allowed` lies about its type *and* mutates input

- **Type:** `type_shape_mismatch`, `name_behavior_mismatch`
- **Location:** `ToolChoice.__post_init__`, lines ~1393-1417.
- **Surprising spans:**
  ```python
  allowed: Sequence[str | Tool] | Tool = ()
  ...
  allowed = tuple(item.name if isinstance(item, (FunctionTool, BuiltinTool))
                  else item for item in raw_allowed)
  ```
- **Expected connection:** `allowed` typed as either tools or names; after
  construction, you'd reasonably expect to read back what you put in.
- **Observed connection:** Tools are coerced to their `.name` strings.
  Static type says `Sequence[str | Tool] | Tool`, runtime reads back
  `tuple[str, ...]`. Round-trip property `allowed_tools` / original
  references are lost.
- **Why it matters:** Code that does `cfg.tool_choice.allowed` expects to
  iterate over `Tool | str`; it actually only iterates over `str`.
- **Confidence:** high.
- **Suggested slow action:** rename annotation to canonical form
  `tuple[str, ...]` and have `ToolChoice` accept `Tool` via a factory or
  `__init__` adapter; or add a separate `allowed_tools: tuple[str, ...]`
  field with a clear name.

---

## Medium-priority surprises

### 10. `Reasoning(effort='off')` is the *default*, but `Config.reasoning=None` ≠ `Reasoning()`

- **Type:** `documentation_mismatch`, `hidden_contract`
- **Location:** `Reasoning` and `Config.reasoning`.
- The comment in the module says reasoning configuration is universal. But
  `Config(reasoning=None)` means "don't override defaults" and
  `Config(reasoning=Reasoning())` means "explicitly disable reasoning."
  The two are observably different (e.g., on Anthropic models that
  reason by default). Nothing in `Reasoning`'s docstring says that
  `Reasoning(effort='off')` is *not* equivalent to "no reasoning config."
- **Suggested slow action:** document the tri-state explicitly, or collapse
  to two states (`reasoning: Reasoning | None` where `None` always means
  "no opinion" and `Reasoning(effort='off')` always means "force off").

### 11. `ToolCallInfo` duplicates `ToolCallPart` minus `type`

- **Type:** `over_connected_concept`
- **Location:** `ToolCallInfo` (lines ~2017-2030); used in `result.py:510`
  and `live.py:184` to copy from `ToolCallPart`.
- The docstring acknowledges they're the same shape. Two types for the
  same identity invites silent drift (e.g., adding `provider_metadata` to
  one and forgetting the other) and forces conversion code at every
  callback boundary.
- **Suggested slow action:** drop `ToolCallInfo` and pass `ToolCallPart`
  directly to callbacks (it already has the same data and a discriminator
  the callback can ignore). Or make `ToolCallInfo` the canonical thing
  and have `ToolCallPart` compose it.

### 12. Hand-maintained `Literal` vocabularies vs structural tables

- **Type:** `documentation_mismatch`, `over_connected_concept`
- **Location:** `PartType`, `DeltaType`, `FinishReason`, `ErrorCode`,
  `ERROR_CODES` tuple, `StreamEventType`, role-set in `Message.__post_init__`,
  encoding set in `AudioFormat`, mode set in `ToolChoice`, effort set in
  `Reasoning`, finish-reason set in `Response`.
- The streamable/non-streamable partition has a runtime consistency check;
  these vocabularies don't. Adding a new `PartType` literal without
  adding a class (or vice versa) goes unnoticed until a stringly-typed
  consumer hits it.
- `Role` literal vs the inline `{"user","assistant","tool","developer"}`
  set, and `FinishReason` literal vs the inline set in `Response`, are
  duplicated again at runtime.
- **Suggested slow action:** generate `PartType`/`DeltaType` values from
  `PART_TYPES`/`DELTA_TYPES`, or assert at import time that
  `set(get_args(PartType)) == set(PART_TYPES.keys())`. Replace
  `ERROR_CODES` with `frozenset(get_args(ErrorCode))`.

### 13. `Config(extensions={}) != Config()`

- **Type:** `symmetry_asymmetry_violation`
- `extensions={}` and `extensions=None` are semantically the same ("no
  provider extensions") but compare unequal and serialize differently.
- **Suggested slow action:** in `__post_init__`, normalize empty dict to
  `None` (or always to `{}`), once, consistently.

### 14. `ImageGenerationResponse` / `AudioGenerationResponse` lack `model`/`usage`/`id`

- **Type:** `symmetry_asymmetry_violation`, `missing_expected_edge`
- `Response` has `id`, `model`, `usage`, `provider_data`. `EmbeddingResponse`
  has `model`, `usage`, `provider_data`. Image/Audio generation responses
  carry only the artifact + `provider_data`.
- This drops billing telemetry and request identity for two endpoints,
  while preserving them for two others. Adapters can't surface a stable
  identity to users for image/audio generations even when the provider
  returns one.
- **Suggested slow action:** add `model: str`, `usage: Usage`, `id: str |
  None` to both, defaulting `usage` to `Usage()`.

### 15. `BatchResponse.status: str` is a free-form vocabulary

- **Type:** `error_handling_gap`, `documentation_mismatch`
- Other vocabularies (`FinishReason`, `ErrorCode`, etc.) are typed
  `Literal[...]`. Batch status is the most natural place for a
  vocabulary (`"queued" | "running" | "completed" | "failed" | "cancelled"`)
  but it's plain `str`. Cross-provider polling code has to invent its own
  taxonomy.
- **Suggested slow action:** define a `BatchStatus` literal and validate.

### 16. `_freeze_required_json_object`, `_freeze_optional_json_object`, `_validate_json_object` are unused

- **Type:** `orphan_span`
- All three helpers exist alongside `_freeze_field` and `_freeze_json_object`,
  but only the latter two are used in `types.py`, and a project-wide grep
  shows no other callers.
- **Suggested slow action:** delete or document them as part of the public
  helper surface for adapters.

### 17. Asymmetric empty-text rules

- `TextPart('')` allowed; `ThinkingPart('')` allowed; `RefusalPart('')`
  rejected. `TextDelta('')` allowed; `ThinkingDelta('')` allowed.
- The reasoning isn't documented. If empty text is allowed for streaming
  reasons, `RefusalPart` should be aligned (a redacted refusal might have
  empty text in some providers).

### 18. `PartType` and `FinishReason` both contain `"tool_call"`

- Same string token, two namespaces. Some serializers may use a single
  flat dispatcher and mis-route. Not currently a bug, but a footgun for
  anyone designing a dispatch table by string alone.

---

## Lower-priority observations

- `tool_call(id: str, name: str, input: JsonObject)` shadows `input` builtin.
  Consider `arguments` or `args_obj` (or document and live with it; it's
  consistent with the design principle of calling tool args "input" everywhere).
- `_is_json_value` and `_freeze_json_object` both walk the structure; the
  redundant `isinstance(value, dict) or not _is_json_value(value)` check in
  `_freeze_json_object` is a safe but confusing duplicate.
- `_freeze_json_value` uses a generator expression in
  `_FrozenJsonArray(_freeze_json_value(item) for item in value)` — works
  because `list` accepts a generator, but reading it asks "is this a list
  of dicts of generators or not?" One line of clarity wouldn't hurt.
- `LiveServerEvent.tool_call_delta` allowed fields include `id`/`name`
  but only `input_delta` is required. Without `id`, consumers must remember
  the "current tool call" — implicit state that adapters need to handle.
  Consider requiring `id` for at least the first delta of a call.
- Module-level `import math`, `import base64`, `import json`, `import re`,
  `import inspect`, `import mimetypes`, `import pathlib` are all done
  inside functions, presumably for import speed. That's fine, but it's
  inconsistent with `from collections.abc import Sequence` etc. at module
  top. Decide once.
- `_field_is_set` returns `value != default`. For mutable-ish defaults
  (`()` vs `[]`), users could pass `[]` and have it count as "set" because
  `[] != ()`. Surprising for a normalization helper. Currently no code path
  passes `[]` for tuple-defaulted fields, but it's brittle.
- `EmbeddingResponse.usage` defaults to `Usage()` (zeros). For a free local
  embedder this is correct, but for an adapter that forgot to populate it,
  the zero is silently propagated rather than flagged.
- `tool_result(id, content, *, name=None, is_error=False)` accepts a `str`
  via `_normalize_parts` but doesn't accept a `bytes` payload, even
  though tool results are sometimes binary. Probably intentional (caller
  must wrap in `ImagePart`/`AudioPart`/etc.), worth documenting.

---

## Make this unsurprising — concrete recommendations

1. **Make hashability honest.** Decide once: either `_FrozenJsonObject`
   computes a stable hash, or every dataclass that contains one declares
   `__hash__ = None` (so the failure is upfront). Document the choice in
   the module docstring next to "frozen + slotted dataclasses throughout."

2. **Tighten `_MediaMixin.data`.** Add `isinstance(self.data, str)` to
   `__post_init__`, mirroring `AudioDelta`. Same for `ImageDelta`. Add a
   regression test passing `bytes`.

3. **Drop the `-1` sentinel on `Usage.total_tokens`.** Use `None` and a
   real type `int | None`. Or expose as a `@property` with no field.

4. **Align stored type with annotation, or add a public smart-constructor
   layer.** The cleanest path:
   - Annotate fields as their canonical form (`tuple[Part, ...]`,
     `tuple[str, ...]`, etc.).
   - Have factories (`Message.user`, `tool_result`, `Config()` accepting
     a wider input) do the normalization.
   - Stop calling `__post_init__` mutation "normalization" without
     reflecting it in types.

5. **Narrow `ToolResultPart.content` to `Sequence[ToolResultContentPart]`.**
   Same for `LiveClientEvent.content`.

6. **Strip `Annotated` and stop falling through to `"string"` silently.**
   In `_json_schema_for_annotation`, handle `Annotated`, and for unknown
   types emit `{}` or raise — never silently lie.

7. **Replace `Response.json` property with `parse_json()` method.**
   Properties shouldn't raise or repeatedly reparse. Optionally accept a
   `default` parameter.

8. **Cache decoded media bytes; validate base64 cheaply.** Either use a
   shape regex on construction and only decode in `bytes`, or decode once
   on first `.bytes` access and cache.

9. **Make `ToolChoice.allowed` either always strings or always tools.**
   Don't accept tools and silently coerce; either reject tools at the
   boundary (with an `allowed_tools=` param that does the coercion) or
   keep references intact.

10. **Cross-check `Literal` vocabularies against structural tables at
    import time.** Mirror `_check_streamable_partition()` for `PartType`,
    `DeltaType`, `ErrorCode`/`ERROR_CODES`, `FinishReason`, `Role`,
    `StreamEventType`, `AudioFormat.encoding`, `ToolChoice.mode`, and
    `Reasoning.effort`.

11. **Either delete `ToolCallInfo` or stop having two types for it.**
    Pass `ToolCallPart` to callbacks directly.

12. **Add `model`, `usage`, `id` to `ImageGenerationResponse` and
    `AudioGenerationResponse`.** Consistency with the rest of the
    EndpointResponse union.

13. **Define `BatchStatus` literal.** Pick a small vocabulary; have
    adapters map provider statuses to it.

14. **Delete unused JSON helpers** (`_freeze_required_json_object`,
    `_freeze_optional_json_object`, `_validate_json_object`) or move them
    to a public helpers module if adapters need them.

15. **Decide one rule for empty text.** Either uniformly allow it across
    all text-bearing parts (with a note on streaming use) or uniformly
    forbid it.

16. **Normalize `extensions={} ↔ extensions=None`** in one direction in
    every `__post_init__` so equality/serialization are stable.
