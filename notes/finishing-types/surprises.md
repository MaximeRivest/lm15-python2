# Surprises in `lm15/types.py`

This is a review of elements that stood out as surprising while reading `lm15/types.py`. Some are likely intentional trade-offs; others look like possible bugs or typing/runtime mismatches worth deciding on before calling the type layer finished.

## Highest-signal surprises

### 1. JSON containers are not fully immutable

`_FrozenJsonObject` and `_FrozenJsonArray` are meant to make JSON payloads recursively read-only while staying dict/list-compatible (`lm15/types.py:108`, `lm15/types.py:125`). The list wrapper blocks `+=` via `__iadd__`, but the dict wrapper does not block `|=` / `__ior__`.

Example:

```python
from lm15.types import ToolCallPart

p = ToolCallPart("id", "name", {})
x = p.input
x |= {"mutated": True}
assert p.input["mutated"] is True
```

If you write `p.input |= {...}` directly, the frozen dataclass raises during field reassignment, but the underlying dict can already have been mutated through an alias. This is probably the clearest actual immutability hole.

### 2. Events are tagged optional-field records, not proper variant dataclasses

The file-level design principles say Parts and Deltas are proper discriminated unions where fields that do not belong to a variant do not exist. That is true for `Part` and `Delta`, but not for:

- `StreamEvent` (`lm15/types.py:891`)
- `LiveClientEvent` (`lm15/types.py:1749`)
- `LiveServerEvent` (`lm15/types.py:1773`)

Those are single dataclasses with many optional fields plus `_require_fields()` / `_forbid_fields()` validation. So a `StreamEvent(type="delta")` still has `.id`, `.model`, `.finish_reason`, `.usage`, `.error`, and `.provider_data` attributes, just set to `None`. This is a different shape from the Part/Delta philosophy and may surprise callers expecting all discriminated types to behave the same way.

### 3. `_forbid_fields()` is a hard-coded cross-event field inventory

`_forbid_fields()` uses one manually maintained `all_variant_fields` set for stream and live event validation (`lm15/types.py:724`). It includes fields from multiple event families: `delta`, `finish_reason`, `provider_data`, `input_delta`, etc.

That is concise, but fragile: adding a new optional field to `StreamEvent`, `LiveClientEvent`, or `LiveServerEvent` silently requires remembering to add it to this unrelated global set. Otherwise the field may never be forbidden on variants where it does not belong.

### 4. `ToolResultPart` runtime validation allows `ThinkingPart`, but the public content type excludes it

The type alias for tool-result content excludes `ThinkingPart` (`ToolResultContentPart`, `lm15/types.py:438`). However, `ToolResultPart.__post_init__()` only rejects `ToolCallPart` and nested `ToolResultPart`; it accepts every other `Part`, including `ThinkingPart` (`lm15/types.py:347`).

So this works at runtime:

```python
from lm15.types import ToolResultPart, ThinkingPart

ToolResultPart(id="call", content=[ThinkingPart("internal")])
```

This may be intentional, but it is a clear typing/runtime mismatch. If tool results should not contain model reasoning traces, direct construction currently bypasses that policy.

### 5. `SystemContent` and `PartInput` typing is broader than the runtime policy

`SystemContent` is declared as `str | Part | Sequence[Part]` (`lm15/types.py:442`), but `_normalize_system()` rejects `ToolCallPart`, `ToolResultPart`, `ThinkingPart`, and `RefusalPart` (`lm15/types.py:698`). Similarly, user/developer messages can type-accept any `Part` through `PartInput`, but `_validate_message_parts()` rejects model/tool protocol parts (`lm15/types.py:681`).

That makes the static API look more permissive than the runtime API. A narrower alias for prompt-safe parts might make the policy more visible.

### 6. Text-bearing parts and deltas do not validate text type or emptiness

`TextPart`, `ThinkingPart`, `RefusalPart`, `TextDelta`, and `ThinkingDelta` accept `text` but do not validate that it is a string or non-empty (`lm15/types.py:276`, `lm15/types.py:366`, `lm15/types.py:374`, `lm15/types.py:757`, `lm15/types.py:768`).

That is surprising because many other identifiers and strings are checked for non-emptiness. For example, `CitationPart` requires at least one field, media addresses cannot be empty, and `ErrorDetail.code` is validated. If empty text is valid, this is fine; if not, it is an inconsistent validation boundary.

### 7. Media delta validation is asymmetric

Inline media parts validate base64 data through `_decode_data()` (`lm15/types.py:235`, `lm15/types.py:246`). `ImageDelta` also validates that inline data is base64 (`lm15/types.py:796`). `AudioDelta` only checks that `data != ""` and does not validate base64 (`lm15/types.py:779`).

This may reflect provider streaming behavior, but it is a surprising asymmetry: image chunks are forced to be valid base64, while audio chunks are merely non-empty strings.

### 8. `ImageDelta` is described as a fragment but requires a complete media address

`ImageDelta` is documented as an image fragment, yet validates exactly one of `data`, `url`, or `file_id` using `_validate_media()` (`lm15/types.py:796`). That means every image delta is a complete address-bearing unit, unlike `AudioDelta`, which can stream raw data chunks.

If image streaming means “one completed image appeared”, the current shape is fine. If it is intended to represent partial image data chunks, the “exactly one address” rule is surprising.

### 9. Live and non-live tool-call deltas differ on empty strings

`ToolCallDelta.input` is required by the dataclass constructor but can be an empty string (`lm15/types.py:816`). In contrast, `LiveServerEvent(type="tool_call_delta")` requires `input_delta`, and `_require_fields()` treats `""` as unset (`lm15/types.py:1738`, `lm15/types.py:1788`).

So an empty tool-call delta is accepted in the normal stream path but rejected in the live path.

### 10. `Usage.total_tokens` is stricter than many provider payloads

`Usage` requires `total_tokens == input_tokens + output_tokens` whenever `total_tokens` is supplied (`lm15/types.py:1259`). That is clean, but surprising for a provider-normalization type because providers may expose separate reasoning, cache, or audio token counts. The class has fields for those dimensions, but the total invariant ignores them.

This forces adapters to decide whether `output_tokens` includes reasoning/audio tokens or to discard/reshape provider totals before constructing `Usage`.

## Other notable surprises

### 11. `Reasoning(effort="off", thinking_budget=...)` silently drops budgets

`Reasoning.__post_init__()` normalizes both `thinking_budget` and `total_budget` to `None` when `effort == "off"` (`lm15/types.py:1076`). That is sensible normalization, but it silently discards explicit user input instead of raising. This could hide configuration mistakes.

### 12. `FunctionTool.from_fn()` has intentionally shallow schema inference, with odd edge cases

`FunctionTool.from_fn()` infers JSON Schema from annotations (`lm15/types.py:1002`), but the helper is deliberately best-effort (`lm15/types.py:953`). A few implications are surprising:

- `Optional[Any]` / `Any | None` falls through to an `anyOf` that includes `NoneType` as `{"type": "string"}`.
- Homogeneous tuples and variadic tuples are collapsed to array items from `args[0]`.
- Unknown annotations become `{"type": "string"}`.
- Parameter defaults are only used for `required`; defaults are not included in the schema.

This is probably okay for a convenience helper, but it should not be mistaken for robust schema generation.

### 13. `EmbeddingResponse` has much lighter validation than neighboring response types

`EmbeddingRequest` validates model and inputs (`lm15/types.py:1441`), and many response types validate required IDs/statuses or concrete payload classes. `EmbeddingResponse`, however, only tuple-ifies vectors and freezes `provider_data` (`lm15/types.py:1461`). It does not check:

- `model` is non-empty
- `usage` is a `Usage`
- vectors are non-empty
- vector elements are numeric / finite

This may be intentional to keep response validation narrow, but it is noticeably lighter than the rest of the file.

### 14. Generated-media request fields are barely validated

`ImageGenerationRequest.size`, `AudioGenerationRequest.voice`, and `AudioGenerationRequest.format` are optional strings without empty-string validation (`lm15/types.py:1583`, `lm15/types.py:1619`). By contrast, `_PromptRequest` validates `prompt`, media parts validate `media_type`, and many names/IDs reject empty strings.

### 15. `FileUploadRequest` is an endpoint request but does not inherit `_ModelRequest`

`FileUploadRequest` appears in `EndpointRequest` (`lm15/types.py:1643`) but does not subclass `_ModelRequest`; its `model` is optional and only `""` is rejected (`lm15/types.py:1480`). This may be right because some file uploads are provider/account scoped rather than model scoped, but it breaks the otherwise common “endpoint request has model” pattern.

### 16. `EndpointRequest` excludes live/realtime types

`EndpointRequest` and `EndpointResponse` include standard request/response families (`lm15/types.py:1643`) but exclude `LiveConfig`, `LiveClientEvent`, and `LiveServerEvent`. That may be a deliberate distinction between request/response endpoints and realtime sessions, but the name “EndpointRequest” is broad enough that the exclusion stands out.

### 17. Runtime validation is narrow enough to admit bools and non-ints in numeric fields

Several helpers validate values by comparison rather than strict type checks, for example `_validate_positive()`, `_validate_non_negative()`, and `_validate_part_index()` (`lm15/types.py:201`, `lm15/types.py:206`, `lm15/types.py:211`). This means `True` can pass as an integer-like value in places such as `max_tokens=True` or `part_index=True`, because `bool` is a subclass of `int` in Python.

This matches the module’s stated “deliberately narrow” validation philosophy, but is still a Python gotcha worth noting.

### 18. `Message.__post_init__()` rejects strings despite a very string-friendly API elsewhere

Direct construction with `Message(role="user", parts="hello")` raises and tells the caller to use `Message.user("text")` (`lm15/types.py:575`). That is a defensible guard against strings being treated as `Sequence[str]`, but it is slightly surprising because `PartInput`, `SystemContent`, and the static constructors are all string-friendly.

### 19. The assistant tool-result error message says “Cannot convert”

If an assistant message contains a `ToolResultPart`, the error is `"Cannot convert assistant messages containing ToolResultPart objects"` (`lm15/types.py:689`). This is probably copied from adapter/conversion code. In the core constructor, “convert” is a surprising verb; “assistant messages cannot contain...” would match nearby errors better.

### 20. `StreamablePart` / `NonStreamablePart` is not mechanically linked to `Delta`

`StreamablePart` and `NonStreamablePart` are manually defined aliases (`lm15/types.py:857`). `Delta` is also manually defined (`lm15/types.py:845`). There is no runtime consistency check tying these together, even though the docstring emphasizes this boundary.

That means adding a new delta or part variant requires updating several places by hand: `Part`, `Delta`, `StreamablePart`, `NonStreamablePart`, and possibly dispatch tables and adapters.

## Summary

The file is internally disciplined, especially around frozen dataclasses, variant unions for Parts/Deltas, and provider metadata separation. The biggest surprises are not the core vocabulary choices; they are the edge inconsistencies around runtime validation and “one representation per concept”:

1. The frozen JSON dict has a real `|=` mutation escape hatch.
2. Event types use a different representation style than Part/Delta variants.
3. Several aliases are broader or narrower than the runtime policy.
4. Validation strictness varies noticeably across similar-looking types.

## Surprising Elements in `@lm15/types.py`

- **Frozen JSON collections (`_FrozenJsonObject` and `_FrozenJsonArray`)**: They inherit from Python's builtin `dict` and `list` but aggressively override all mutating methods (`__setitem__`, `pop`, `append`, etc.) with a `_readonly` method to enforce deep immutability at runtime.
- **`developer` message role**: Includes an explicit `Message.developer()` constructor to inject high-authority instructions anywhere in the conversation. The docstring notes it handles the polyfill for non-OpenAI models via a `[developer]` prefix.
- **Explicit `ThinkingPart` and `RefusalPart`**: Integrated natively as first-class core vocabulary variants to deal with reasoning traces (with a `redacted` flag) and safety refusals explicitly.
- **The `Reasoning` configuration**: Introduces unified abstractions like `adaptive` and `xhigh` effort levels, as well as `thinking_budget` and `total_budget` capabilities to govern internal model thinking.
- **`cache: bool = True` on `Request`**: Native inclusion of prompt caching in the top-level request object, expecting the adapter layer to automatically figure out cache breakpoints.
- **`_json_schema_for_annotation`**: A built-in, best-effort reflection utility to turn Python type annotations (like `list`, `dict`, `Union`, `Literal`) into JSON Schema parameters for tool usage.
- **Strict `Extensions` / `ProviderData` Escape Hatches**: The module rigorously forces all provider-specific configs and outputs into explicit `extensions` and `provider_data` fields, preserving the universal structures.
- **Live/Realtime protocols**: Real-time audio/video streaming (`LiveConfig`, `LiveClientEvent`, `LiveServerEvent`), embedding requests, and batch operations are fully typed and included in this same core definitions file.

# Surprises in lm15/types.py

A scan for things that surprised me — design quirks, asymmetries, latent
bugs, and places where the code disagrees with itself or with the stated
design principles.

## Discriminated-union claim is partly aspirational

The module docstring says "Fields that don't belong to a variant don't
exist on it — accessing them raises AttributeError." That's true for the
`Part` and `Delta` unions (each variant is its own dataclass). It is
**not** true for `StreamEvent`, `LiveClientEvent`, and `LiveServerEvent`,
which are single dataclasses with a `type` field and a flat union of
optional fields (`id`, `model`, `delta`, `finish_reason`, `usage`,
`error`, `data`, `text`, `content`, `name`, `input`, `input_delta`,
`provider_data`). All those fields exist on every instance, regardless
of `type`. The `_forbid_fields` runtime check papers over this, but the
type system doesn't reflect the discriminated-union story for these
three classes.

## `_forbid_fields` uses a hard-coded "all variant fields" set

`_forbid_fields` carries a literal set of field names:

    {"id", "model", "delta", "finish_reason", "usage", "error",
     "provider_data", "data", "text", "content", "name", "input",
     "input_delta"}

Adding a new field to `StreamEvent`, `LiveClientEvent`, or
`LiveServerEvent` without updating that set silently disables the
"forbidden field" check for it. This is a footgun given that
`PART_TYPES`/`DELTA_TYPES` go to some lengths to derive themselves from
the union. The Live/Stream side of the module didn't get the same
treatment.

## `_field_is_set` treats `0` and `False` as "set", but `()` and `""` as "unset"

    return value is not None and value != () and value != ""

This is asymmetric. An empty tuple counts as unset (used for `content`
defaulting to `()`), but `0`, `False`, and `[]` would count as set. It
mostly works because no current field has `0`/`False` defaults that need
to mean "unset", but it's a subtle invariant that future fields could
trip over. In particular, `False` for `is_error` is intentionally a real
value — but `is_error` isn't in the validated set, which is why it
escapes notice.

## `bool` is `int`, but `_is_json_value` checks `bool` before `float`

`_is_json_value` does `isinstance(value, (bool, int, str))` first, then
the float branch. That ordering is fine for acceptance, but it's worth
noting that `True`/`False` go through the int path and bypass the
`math.isfinite` check (good — but easy to get wrong if reordered).
NaN/Infinity floats are correctly rejected; `bool` instances would
incorrectly serialize as `true`/`false` either way, so behavior is
correct but the ordering matters and isn't commented.

## `JsonObject` lies about being a `dict`

`JsonObject` is aliased to `dict[str, JsonValue]`, but in practice every
`JsonObject`-typed field on these dataclasses is a `_FrozenJsonObject`
(a dict subclass that raises `TypeError` on mutation). Callers who
type-check against `dict` and try to mutate will get a runtime error
that the type checker can't see. The frozen-ness is invisible to mypy.

## `_FrozenJsonObject.__hash__` is still the default (unhashable)

Both `_FrozenJsonObject` and `_FrozenJsonArray` inherit from `dict` /
`list`, which set `__hash__ = None`. So even though these objects are
"immutable", they aren't hashable, which means dataclasses containing
them can't be put into sets or used as dict keys. That conflicts with
the "frozen + slotted dataclasses throughout" feel of the file — the
top-level dataclasses are hashable in principle (via `frozen=True`),
but any dataclass that holds a `JsonObject` is effectively not.

## `Reasoning.__post_init__` silently rewrites your inputs

If `effort == "off"`, `thinking_budget` and `total_budget` are silently
forced to `None` — even if you explicitly passed positive ints. There's
no warning or error; your construction arguments quietly disappear.
The docstring mentions normalization but most users won't notice that
their explicit budget got dropped.

## `Usage.total_tokens` zero-vs-supplied ambiguity

    if self.total_tokens == 0 and computed_total:
        object.__setattr__(self, "total_tokens", computed_total)
    elif self.total_tokens != computed_total:
        raise ValueError(...)

If you legitimately have `input_tokens == 0` and `output_tokens == 0`
and pass `total_tokens=0`, fine. But if you pass `total_tokens=0` while
`input_tokens + output_tokens > 0`, the code silently overwrites your
zero with the sum rather than raising. So `0` is treated as "I didn't
supply this" — there is no way to assert "the total is genuinely zero
even though input+output isn't" (which is admittedly nonsensical, but
the sentinel-as-None pattern is surprising in a class that otherwise
uses `None` for absence elsewhere.)

## `FileUploadRequest` defaults are surprisingly permissive

`FileUploadRequest` defaults `model=None`, `filename="file.bin"`,
`bytes_data=b""`, `media_type="application/octet-stream"` — and then
`__post_init__` rejects empty `bytes_data`. So the dataclass has a
default that immediately fails validation if you instantiate it with no
arguments. The default value exists only so `bytes_data` can be a
positional/keyword field after the other defaulted fields.

## `tool_result` factory accepts `Sequence`, but a `str` is also a `Sequence`

`tool_result(...)` does `isinstance(content, str)` first, so it's safe.
But the type alias `ToolResultContent = str | ToolResultContentPart |
Sequence[ToolResultContentPart]` is technically ambiguous since `str`
is a `Sequence[str]`. The runtime ordering handles it; mypy users may
see odd inferences.

## `Message.parts` annotation is a lie post-construction

Declared as `Part | Sequence[Part]`, but after `__post_init__` it's
always `tuple[Part, ...]`. Callers who read `msg.parts` will see a
tuple, but the type says they might see a single Part. The same trick
is used in many other classes (`Request.messages`, `Request.tools`,
`Config.stop`, etc.). This is a pervasive pattern: input-friendly
annotation, normalized-to-tuple at runtime, mismatched type after
construction. A `__class_getitem__`-style normalized type would match
reality better.

## `Message._validate_message_parts` quietly forbids `CitationPart` for tool-role messages

`tool` role: only `ToolResultPart`. OK.
`assistant` role: anything except `ToolResultPart`. So an assistant can
emit citations, refusals, thinking — fine.
`user`/`developer`: forbids `ToolCallPart`, `ToolResultPart`,
`ThinkingPart`, `RefusalPart` — but **allows `CitationPart`**. A user
message can contain a citation, which is semantically odd (citations
are typically a model-emitted artifact). Either intentional or an
oversight worth confirming.

## `Response.json` strips one layer of triple-backtick fencing

The property silently strips a leading/trailing ```` ``` ```` block
before parsing. That's pragmatic but it's a magical behavior tucked
inside a property: a user calling `response.json` doesn't know that
their model's markdown wrapping was stripped, and they get no signal
about it. Also — only one layer is stripped, and the closing fence
must be on its own line (`lines[-1].strip() == "```"`).

## `_json_schema_for_annotation` falls back to `"string"` for unknown types

The fallback for any unrecognized annotation is `{"type": "string"}`.
That is a strong, silent default — `Decimal`, `datetime`, `pathlib.Path`,
custom classes will all be advertised to the model as strings, which
might be wrong. No warning is emitted.

## `_json_schema_for_annotation` reuses the name `schema` with two different types

In the `Literal` branch and the `list/tuple/set` branch, a local
`schema: JsonObject = ...` is declared. Then the `dict` branch reuses
the name `schema = {"type": "object"}` without an annotation. A reader
might briefly wonder whether `schema` is the same variable reused
across branches (it is, lexically — Python doesn't have block scope).
Minor, but the redeclared annotation suggests the author thought of
them as separate.

## `_json_schema_for_annotation` returns `JsonObject` containing arbitrary `Literal` values

For `Literal[1, 2, 3]` it returns `{"enum": [1, 2, 3], "type": "integer"}`.
For `Literal[some_enum_member]` it would return whatever
`type(some_enum_member).__name__`-typed thing is in `_JSON_SCHEMA_TYPES`
(probably nothing → no `type` key). And critically, the `enum` list is
a regular Python `list[Any]` containing the literal values — which may
not be JSON-representable (e.g. `Literal[MyEnum.X]`). The resulting
"JsonObject" can fail `_is_json_value` if the schema is then frozen.

## `FunctionTool.from_fn` uses `str` as the default annotation

If a parameter has no annotation, it's treated as `str`. That's a
reasonable default but invisible to callers. A function like
`def f(x): ...` becomes a tool with a `string` parameter.

## `_PromptRequest.__post_init__` — `if not self.prompt`

`if not self.prompt` rejects `""` and any falsy prompt. But the
annotation is `str`, not `str | None`, so users would normally only
hit this with `""`. Other places use `== ""` (the `id` check on
`Response`, `system` check, etc.). Inconsistent style.

## `Response.id` accepts `None` but rejects `""` — yet `model` rejects both

`Response.id` is `str | None`; passing `""` raises with "use None when
unavailable". But `Response.model` requires non-empty string. Asymmetric
treatment of the two identity-ish fields, justified but worth flagging.

## `BatchRequest.model` must equal every nested `Request.model`

Subtle constraint that's not obvious from the type signature: the outer
`BatchRequest.model` is the source of truth, and any inner `Request`
with a different model raises. So `BatchRequest` is really "batch of
requests for one model", not "batch of arbitrary requests".

## `EmbeddingResponse` has no validation on `model`

Unlike `Response`, `EmbeddingResponse.model` is annotated `str` but
never validated for non-empty. A bare `EmbeddingResponse(model="", ...)`
will succeed.

## `LiveConfig` inherits from `_ModelRequest` but is not a "request"

The class is called `LiveConfig` but it's a `_ModelRequest` subclass.
Names like `LiveConfig` suggest configuration, but it sits in the
request hierarchy. It's also not in `EndpointRequest`. So the type
hierarchy and the EndpointRequest union diverge: not all
`_ModelRequest` subclasses are endpoint requests.

## `EndpointRequest` and `EndpointResponse` exclude Live events entirely

`LiveConfig`, `LiveClientEvent`, `LiveServerEvent` are not part of
`EndpointRequest`/`EndpointResponse`. Not necessarily wrong, but the
"Endpoint" naming may suggest "all endpoints" when it really means
"non-streaming, non-live endpoints".

## `ToolCallInfo` duplicates `ToolCallPart` fields and validation

Three classes share `id` / `name` / `input` shape and validation
(`ToolCallPart`, `ToolCallInfo`, plus `LiveServerEvent.type=="tool_call"`).
The validation is reimplemented each time. A small mixin or shared
constructor helper would be consistent with `_MediaMixin`.

## Imports inside functions

`base64`, `math`, `inspect`, `json` are imported inside their respective
functions/methods (`_decode_data`, `_is_json_value`, `_encode_data`,
`from_fn`, `Response.json`). This is presumably to keep module-import
time low, but it's inconsistent with the top-level imports of `dataclass`,
`field`, `Sequence`, etc. Not wrong, just stylistically split.

## `DeltaType` literal duplicates the per-class `type` literals

`DeltaType = Literal["text", "thinking", "audio", "image", "tool_call",
"citation"]` is declared once, but each delta class also independently
declares its own `type: Literal["..."]`. There's no static link between
the union literal and the per-variant literals, so adding a delta class
requires updating both. Same pattern with `PartType`. Compare with
`PART_TYPES` / `DELTA_TYPES` runtime tables, which **are** derived from
the union — the literal-vs-class duplication is the one place that
isn't single-sourced.

## `ImagePart.detail` is image-only

The other media parts don't have a `detail` field. This is OpenAI's
image-detail knob leaking into the universal vocabulary. Justifiable,
but it does mean the universal core has one provider-flavored field on
exactly one part.

## `ThinkingPart.redacted` has no analog in `ThinkingDelta`

If a streamed thinking fragment is redacted, there's no way to convey
that through `ThinkingDelta`. So redaction can only be expressed at the
final `Part` level, not mid-stream — meaning the assembler must know
out-of-band whether to set `redacted=True` on the assembled part.

## `CitationDelta` allows `text`/`url`/`title` but `CitationPart` *also* allows just `text`

Both require at least one of those three fields. But the streaming
shape `CitationDelta` doesn't include the same notion of accumulating
into a `CitationPart` cleanly — there's no `part_index` semantic for
"these three deltas combine into one citation". The relationship
between successive `CitationDelta`s is unspecified: do they replace,
append, or merge?

## `AudioDelta` requires non-empty `data` but `ImageDelta` allows `data=None`

`AudioDelta.data` is `str` (required, validated non-empty). `ImageDelta.data`
is `str | None` (because images can be addressed by url/file_id mid-stream).
But `AudioDelta` doesn't have `url`/`file_id` — audio cannot be
streamed-by-reference even though images can. Possibly intentional
(realtime audio is always inline), but it's an asymmetry.

## `_normalize_system` returns `str | tuple[Part, ...] | None`

Three different shapes, depending on input. Callers downstream must
handle all three. Compare with `_normalize_parts` which always returns
`tuple[Part, ...]`. A consistent return type (always a tuple, with a
`TextPart` wrapper for strings) would be simpler — but presumably the
distinction is preserved so adapters can pick the most native
representation per provider.

## `Config.stop = ()` default but typed as `str | Sequence[str] | None`

The default is `()` (a tuple), the annotation includes `None`, and
`__post_init__` normalizes `None` to `()` and `str` to `(str,)`. So
`None` and `()` are equivalent post-construction, but `None` is in the
type and `()` is the default. Slightly muddy.

## `_MediaMixin` is not a dataclass and has no `__init__`

It's a mixin that declares fields as class-level annotations
(`type`, `media_type`, `data`, `url`, `file_id`) without defaults. The
real dataclasses inherit from it and supply the fields. This works but
type checkers may complain about the mixin's unbound annotations.
Slots are also empty (`__slots__ = ()`), which is required for the
slotted-dataclass-with-mixin pattern but easy to forget.

## Inheriting `__post_init__` with `slots=True` and `frozen=True`

`_ModelRequest.__post_init__(self)` is called explicitly from subclass
`__post_init__`s, because dataclass inheritance does not auto-chain
`__post_init__`. If a future subclass forgets the explicit super-call,
it loses the `model` validation silently. This is the common Python
inheritance gotcha; nothing alerts the author.

## `Request` validates `tool_choice.allowed ⊆ tool_names` but `LiveConfig` does not

`Request.__post_init__` checks that every name in
`config.tool_choice.allowed` exists in `tools`. `LiveConfig` doesn't
take a `Config` (and so has no `tool_choice`), so no such check exists
for live sessions. If a Live API gains tool-choice semantics later,
this asymmetry will need to be addressed.

