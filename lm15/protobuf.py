"""Protobuf conversion helpers for the lm15 core type system.

The canonical schema lives at ``proto/lm15/v1/lm15.proto``.  This module
provides runtime conversion between idiomatic Python dataclasses in
``lm15.types`` and protobuf messages generated from that schema.

The module uses protobuf reflection over an embedded descriptor set rather than
importing generated ``*_pb2.py`` code.  That keeps the package independent of a
particular local ``protoc`` version and avoids generated-code/runtime skew.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.json_format import MessageToDict, ParseDict

from .types import (
    AudioDelta,
    AudioFormat,
    AudioGenerationRequest,
    AudioGenerationResponse,
    AudioPart,
    BatchRequest,
    BatchResponse,
    BuiltinTool,
    CitationDelta,
    CitationPart,
    Config,
    DocumentPart,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorDetail,
    FileUploadRequest,
    FileUploadResponse,
    FunctionTool,
    ImageDelta,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImagePart,
    LiveClientAudioEvent,
    LiveClientEndAudioEvent,
    LiveClientEvent,
    LiveClientInterruptEvent,
    LiveClientTextEvent,
    LiveClientToolResultEvent,
    LiveClientVideoEvent,
    LiveConfig,
    LiveServerAudioEvent,
    LiveServerErrorEvent,
    LiveServerEvent,
    LiveServerInterruptedEvent,
    LiveServerTextEvent,
    LiveServerToolCallDeltaEvent,
    LiveServerToolCallEvent,
    LiveServerTurnEndEvent,
    Message,
    Part,
    Reasoning,
    RefusalPart,
    Request,
    Response,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    Tool,
    ToolCallDelta,
    ToolCallInfo,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    Usage,
    VideoPart,
)

_DESCRIPTOR_SET_B64 = """
CuUFChxnb29nbGUvcHJvdG9idWYvc3RydWN0LnByb3RvEg9nb29nbGUucHJvdG9idWYimAEKBlN0cnVjdBI7CgZmaWVsZHMYASADKAsyIy5nb29nbGUucHJv
dG9idWYuU3RydWN0LkZpZWxkc0VudHJ5UgZmaWVsZHMaUQoLRmllbGRzRW50cnkSEAoDa2V5GAEgASgJUgNrZXkSLAoFdmFsdWUYAiABKAsyFi5nb29nbGUu
cHJvdG9idWYuVmFsdWVSBXZhbHVlOgI4ASKyAgoFVmFsdWUSOwoKbnVsbF92YWx1ZRgBIAEoDjIaLmdvb2dsZS5wcm90b2J1Zi5OdWxsVmFsdWVIAFIJbnVs
bFZhbHVlEiMKDG51bWJlcl92YWx1ZRgCIAEoAUgAUgtudW1iZXJWYWx1ZRIjCgxzdHJpbmdfdmFsdWUYAyABKAlIAFILc3RyaW5nVmFsdWUSHwoKYm9vbF92
YWx1ZRgEIAEoCEgAUglib29sVmFsdWUSPAoMc3RydWN0X3ZhbHVlGAUgASgLMhcuZ29vZ2xlLnByb3RvYnVmLlN0cnVjdEgAUgtzdHJ1Y3RWYWx1ZRI7Cgps
aXN0X3ZhbHVlGAYgASgLMhouZ29vZ2xlLnByb3RvYnVmLkxpc3RWYWx1ZUgAUglsaXN0VmFsdWVCBgoEa2luZCI7CglMaXN0VmFsdWUSLgoGdmFsdWVzGAEg
AygLMhYuZ29vZ2xlLnByb3RvYnVmLlZhbHVlUgZ2YWx1ZXMqGwoJTnVsbFZhbHVlEg4KCk5VTExfVkFMVUUQAEKBAQoTY29tLmdvb2dsZS5wcm90b2J1ZkIL
U3RydWN0UHJvdG9QAVoxZ2l0aHViLmNvbS9nb2xhbmcvcHJvdG9idWYvcHR5cGVzL3N0cnVjdDtzdHJ1Y3RwYvgBAaICA0dQQqoCHkdvb2dsZS5Qcm90b2J1
Zi5XZWxsS25vd25UeXBlc2IGcHJvdG8zCv4DCh5nb29nbGUvcHJvdG9idWYvd3JhcHBlcnMucHJvdG8SD2dvb2dsZS5wcm90b2J1ZiIjCgtEb3VibGVWYWx1
ZRIUCgV2YWx1ZRgBIAEoAVIFdmFsdWUiIgoKRmxvYXRWYWx1ZRIUCgV2YWx1ZRgBIAEoAlIFdmFsdWUiIgoKSW50NjRWYWx1ZRIUCgV2YWx1ZRgBIAEoA1IF
dmFsdWUiIwoLVUludDY0VmFsdWUSFAoFdmFsdWUYASABKARSBXZhbHVlIiIKCkludDMyVmFsdWUSFAoFdmFsdWUYASABKAVSBXZhbHVlIiMKC1VJbnQzMlZh
bHVlEhQKBXZhbHVlGAEgASgNUgV2YWx1ZSIhCglCb29sVmFsdWUSFAoFdmFsdWUYASABKAhSBXZhbHVlIiMKC1N0cmluZ1ZhbHVlEhQKBXZhbHVlGAEgASgJ
UgV2YWx1ZSIiCgpCeXRlc1ZhbHVlEhQKBXZhbHVlGAEgASgMUgV2YWx1ZUJ8ChNjb20uZ29vZ2xlLnByb3RvYnVmQg1XcmFwcGVyc1Byb3RvUAFaKmdpdGh1
Yi5jb20vZ29sYW5nL3Byb3RvYnVmL3B0eXBlcy93cmFwcGVyc/gBAaICA0dQQqoCHkdvb2dsZS5Qcm90b2J1Zi5XZWxsS25vd25UeXBlc2IGcHJvdG8zCqJg
ChJsbTE1L3YxL2xtMTUucHJvdG8SB2xtMTUudjEaHGdvb2dsZS9wcm90b2J1Zi9zdHJ1Y3QucHJvdG8aHmdvb2dsZS9wcm90b2J1Zi93cmFwcGVycy5wcm90
byL+AwoEUGFydBInCgR0ZXh0GAEgASgLMhEubG0xNS52MS5UZXh0UGFydEgAUgR0ZXh0EioKBWltYWdlGAIgASgLMhIubG0xNS52MS5JbWFnZVBhcnRIAFIF
aW1hZ2USKgoFYXVkaW8YAyABKAsyEi5sbTE1LnYxLkF1ZGlvUGFydEgAUgVhdWRpbxIqCgV2aWRlbxgEIAEoCzISLmxtMTUudjEuVmlkZW9QYXJ0SABSBXZp
ZGVvEjMKCGRvY3VtZW50GAUgASgLMhUubG0xNS52MS5Eb2N1bWVudFBhcnRIAFIIZG9jdW1lbnQSNAoJdG9vbF9jYWxsGAYgASgLMhUubG0xNS52MS5Ub29s
Q2FsbFBhcnRIAFIIdG9vbENhbGwSOgoLdG9vbF9yZXN1bHQYByABKAsyFy5sbTE1LnYxLlRvb2xSZXN1bHRQYXJ0SABSCnRvb2xSZXN1bHQSMwoIdGhpbmtp
bmcYCCABKAsyFS5sbTE1LnYxLlRoaW5raW5nUGFydEgAUgh0aGlua2luZxIwCgdyZWZ1c2FsGAkgASgLMhQubG0xNS52MS5SZWZ1c2FsUGFydEgAUgdyZWZ1
c2FsEjMKCGNpdGF0aW9uGAogASgLMhUubG0xNS52MS5DaXRhdGlvblBhcnRIAFIIY2l0YXRpb25CBgoEa2luZCIvCghQYXJ0TGlzdBIjCgVwYXJ0cxgBIAMo
CzINLmxtMTUudjEuUGFydFIFcGFydHMiHgoIVGV4dFBhcnQSEgoEdGV4dBgBIAEoCVIEdGV4dCJ7CgtNZWRpYVNvdXJjZRIdCgptZWRpYV90eXBlGAEgASgJ
UgltZWRpYVR5cGUSFAoEZGF0YRgCIAEoDEgAUgRkYXRhEhIKA3VybBgDIAEoCUgAUgN1cmwSGQoHZmlsZV9pZBgEIAEoCUgAUgZmaWxlSWRCCAoGc291cmNl
ImcKCUltYWdlUGFydBIsCgZzb3VyY2UYASABKAsyFC5sbTE1LnYxLk1lZGlhU291cmNlUgZzb3VyY2USLAoGZGV0YWlsGAIgASgOMhQubG0xNS52MS5JbWFn
ZURldGFpbFIGZGV0YWlsIjkKCUF1ZGlvUGFydBIsCgZzb3VyY2UYASABKAsyFC5sbTE1LnYxLk1lZGlhU291cmNlUgZzb3VyY2UiOQoJVmlkZW9QYXJ0EiwK
BnNvdXJjZRgBIAEoCzIULmxtMTUudjEuTWVkaWFTb3VyY2VSBnNvdXJjZSI8CgxEb2N1bWVudFBhcnQSLAoGc291cmNlGAEgASgLMhQubG0xNS52MS5NZWRp
YVNvdXJjZVIGc291cmNlImEKDFRvb2xDYWxsUGFydBIOCgJpZBgBIAEoCVICaWQSEgoEbmFtZRgCIAEoCVIEbmFtZRItCgVpbnB1dBgDIAEoCzIXLmdvb2ds
ZS5wcm90b2J1Zi5TdHJ1Y3RSBWlucHV0IpYBCg5Ub29sUmVzdWx0UGFydBIOCgJpZBgBIAEoCVICaWQSJwoHY29udGVudBgCIAMoCzINLmxtMTUudjEuUGFy
dFIHY29udGVudBIwCgRuYW1lGAMgASgLMhwuZ29vZ2xlLnByb3RvYnVmLlN0cmluZ1ZhbHVlUgRuYW1lEhkKCGlzX2Vycm9yGAQgASgIUgdpc0Vycm9yIj4K
DFRoaW5raW5nUGFydBISCgR0ZXh0GAEgASgJUgR0ZXh0EhoKCHJlZGFjdGVkGAIgASgIUghyZWRhY3RlZCIhCgtSZWZ1c2FsUGFydBISCgR0ZXh0GAEgASgJ
UgR0ZXh0IqQBCgxDaXRhdGlvblBhcnQSLgoDdXJsGAEgASgLMhwuZ29vZ2xlLnByb3RvYnVmLlN0cmluZ1ZhbHVlUgN1cmwSMgoFdGl0bGUYAiABKAsyHC5n
b29nbGUucHJvdG9idWYuU3RyaW5nVmFsdWVSBXRpdGxlEjAKBHRleHQYAyABKAsyHC5nb29nbGUucHJvdG9idWYuU3RyaW5nVmFsdWVSBHRleHQiUQoHTWVz
c2FnZRIhCgRyb2xlGAEgASgOMg0ubG0xNS52MS5Sb2xlUgRyb2xlEiMKBXBhcnRzGAIgAygLMg0ubG0xNS52MS5QYXJ0UgVwYXJ0cyJYCg1TeXN0ZW1Db250
ZW50EhQKBHRleHQYASABKAlIAFIEdGV4dBIpCgVwYXJ0cxgCIAEoCzIRLmxtMTUudjEuUGFydExpc3RIAFIFcGFydHNCBgoEa2luZCK2AgoFRGVsdGESKAoE
dGV4dBgBIAEoCzISLmxtMTUudjEuVGV4dERlbHRhSABSBHRleHQSNAoIdGhpbmtpbmcYAiABKAsyFi5sbTE1LnYxLlRoaW5raW5nRGVsdGFIAFIIdGhpbmtp
bmcSKwoFYXVkaW8YAyABKAsyEy5sbTE1LnYxLkF1ZGlvRGVsdGFIAFIFYXVkaW8SKwoFaW1hZ2UYBCABKAsyEy5sbTE1LnYxLkltYWdlRGVsdGFIAFIFaW1h
Z2USNQoJdG9vbF9jYWxsGAUgASgLMhYubG0xNS52MS5Ub29sQ2FsbERlbHRhSABSCHRvb2xDYWxsEjQKCGNpdGF0aW9uGAYgASgLMhYubG0xNS52MS5DaXRh
dGlvbkRlbHRhSABSCGNpdGF0aW9uQgYKBGtpbmQiPgoJVGV4dERlbHRhEhIKBHRleHQYASABKAlSBHRleHQSHQoKcGFydF9pbmRleBgCIAEoBVIJcGFydElu
ZGV4IkIKDVRoaW5raW5nRGVsdGESEgoEdGV4dBgBIAEoCVIEdGV4dBIdCgpwYXJ0X2luZGV4GAIgASgFUglwYXJ0SW5kZXgitwEKCkF1ZGlvRGVsdGESHQoK
cGFydF9pbmRleBgBIAEoBVIJcGFydEluZGV4EjsKCm1lZGlhX3R5cGUYAiABKAsyHC5nb29nbGUucHJvdG9idWYuU3RyaW5nVmFsdWVSCW1lZGlhVHlwZRIU
CgRkYXRhGAMgASgJSABSBGRhdGESEgoDdXJsGAQgASgJSABSA3VybBIZCgdmaWxlX2lkGAUgASgJSABSBmZpbGVJZEIICgZzb3VyY2UitwEKCkltYWdlRGVs
dGESHQoKcGFydF9pbmRleBgBIAEoBVIJcGFydEluZGV4EjsKCm1lZGlhX3R5cGUYAiABKAsyHC5nb29nbGUucHJvdG9idWYuU3RyaW5nVmFsdWVSCW1lZGlh
VHlwZRIUCgRkYXRhGAMgASgJSABSBGRhdGESEgoDdXJsGAQgASgJSABSA3VybBIZCgdmaWxlX2lkGAUgASgJSABSBmZpbGVJZEIICgZzb3VyY2UipAEKDVRv
b2xDYWxsRGVsdGESFAoFaW5wdXQYASABKAlSBWlucHV0Eh0KCnBhcnRfaW5kZXgYAiABKAVSCXBhcnRJbmRleBIsCgJpZBgDIAEoCzIcLmdvb2dsZS5wcm90
b2J1Zi5TdHJpbmdWYWx1ZVICaWQSMAoEbmFtZRgEIAEoCzIcLmdvb2dsZS5wcm90b2J1Zi5TdHJpbmdWYWx1ZVIEbmFtZSLEAQoNQ2l0YXRpb25EZWx0YRIw
CgR0ZXh0GAEgASgLMhwuZ29vZ2xlLnByb3RvYnVmLlN0cmluZ1ZhbHVlUgR0ZXh0Ei4KA3VybBgCIAEoCzIcLmdvb2dsZS5wcm90b2J1Zi5TdHJpbmdWYWx1
ZVIDdXJsEjIKBXRpdGxlGAMgASgLMhwuZ29vZ2xlLnByb3RvYnVmLlN0cmluZ1ZhbHVlUgV0aXRsZRIdCgpwYXJ0X2luZGV4GAQgASgFUglwYXJ0SW5kZXgi
xAEKC1N0cmVhbUV2ZW50EisKBXN0YXJ0GAEgASgLMhMubG0xNS52MS5TdGFydEV2ZW50SABSBXN0YXJ0EisKBWRlbHRhGAIgASgLMhMubG0xNS52MS5EZWx0
YUV2ZW50SABSBWRlbHRhEiUKA2VuZBgDIAEoCzIRLmxtMTUudjEuRW5kRXZlbnRIAFIDZW5kEisKBWVycm9yGAQgASgLMhMubG0xNS52MS5FcnJvckV2ZW50
SABSBWVycm9yQgcKBWV2ZW50Im4KClN0YXJ0RXZlbnQSLAoCaWQYASABKAsyHC5nb29nbGUucHJvdG9idWYuU3RyaW5nVmFsdWVSAmlkEjIKBW1vZGVsGAIg
ASgLMhwuZ29vZ2xlLnByb3RvYnVmLlN0cmluZ1ZhbHVlUgVtb2RlbCIyCgpEZWx0YUV2ZW50EiQKBWRlbHRhGAEgASgLMg4ubG0xNS52MS5EZWx0YVIFZGVs
dGEiqgEKCEVuZEV2ZW50EjoKDWZpbmlzaF9yZWFzb24YASABKA4yFS5sbTE1LnYxLkZpbmlzaFJlYXNvblIMZmluaXNoUmVhc29uEiQKBXVzYWdlGAIgASgL
Mg4ubG0xNS52MS5Vc2FnZVIFdXNhZ2USPAoNcHJvdmlkZXJfZGF0YRgDIAEoCzIXLmdvb2dsZS5wcm90b2J1Zi5TdHJ1Y3RSDHByb3ZpZGVyRGF0YSI4CgpF
cnJvckV2ZW50EioKBWVycm9yGAEgASgLMhQubG0xNS52MS5FcnJvckRldGFpbFIFZXJyb3IikgEKC0Vycm9yRGV0YWlsEiYKBGNvZGUYASABKA4yEi5sbTE1
LnYxLkVycm9yQ29kZVIEY29kZRIYCgdtZXNzYWdlGAIgASgJUgdtZXNzYWdlEkEKDXByb3ZpZGVyX2NvZGUYAyABKAsyHC5nb29nbGUucHJvdG9idWYuU3Ry
aW5nVmFsdWVSDHByb3ZpZGVyQ29kZSJ1CgRUb29sEjMKCGZ1bmN0aW9uGAEgASgLMhUubG0xNS52MS5GdW5jdGlvblRvb2xIAFIIZnVuY3Rpb24SMAoHYnVp
bHRpbhgCIAEoCzIULmxtMTUudjEuQnVpbHRpblRvb2xIAFIHYnVpbHRpbkIGCgRraW5kIpsBCgxGdW5jdGlvblRvb2wSEgoEbmFtZRgBIAEoCVIEbmFtZRI+
CgtkZXNjcmlwdGlvbhgCIAEoCzIcLmdvb2dsZS5wcm90b2J1Zi5TdHJpbmdWYWx1ZVILZGVzY3JpcHRpb24SNwoKcGFyYW1ldGVycxgDIAEoCzIXLmdvb2ds
ZS5wcm90b2J1Zi5TdHJ1Y3RSCnBhcmFtZXRlcnMiUgoLQnVpbHRpblRvb2wSEgoEbmFtZRgBIAEoCVIEbmFtZRIvCgZjb25maWcYAiABKAsyFy5nb29nbGUu
cHJvdG9idWYuU3RydWN0UgZjb25maWciYQoMVG9vbENhbGxJbmZvEg4KAmlkGAEgASgJUgJpZBISCgRuYW1lGAIgASgJUgRuYW1lEi0KBWlucHV0GAMgASgL
MhcuZ29vZ2xlLnByb3RvYnVmLlN0cnVjdFIFaW5wdXQi+wEKCVJlYXNvbmluZxIwCgZlZmZvcnQYASABKA4yGC5sbTE1LnYxLlJlYXNvbmluZ0VmZm9ydFIG
ZWZmb3J0EkQKD3RoaW5raW5nX2J1ZGdldBgCIAEoCzIbLmdvb2dsZS5wcm90b2J1Zi5JbnQzMlZhbHVlUg50aGlua2luZ0J1ZGdldBI+Cgx0b3RhbF9idWRn
ZXQYAyABKAsyGy5nb29nbGUucHJvdG9idWYuSW50MzJWYWx1ZVILdG90YWxCdWRnZXQSNgoHc3VtbWFyeRgEIAEoCzIcLmdvb2dsZS5wcm90b2J1Zi5TdHJp
bmdWYWx1ZVIHc3VtbWFyeSKLAQoKVG9vbENob2ljZRIrCgRtb2RlGAEgASgOMhcubG0xNS52MS5Ub29sQ2hvaWNlTW9kZVIEbW9kZRIYCgdhbGxvd2VkGAIg
AygJUgdhbGxvd2VkEjYKCHBhcmFsbGVsGAMgASgLMhouZ29vZ2xlLnByb3RvYnVmLkJvb2xWYWx1ZVIIcGFyYWxsZWwi4AMKBkNvbmZpZxI6CgptYXhfdG9r
ZW5zGAEgASgLMhsuZ29vZ2xlLnByb3RvYnVmLkludDMyVmFsdWVSCW1heFRva2VucxI+Cgt0ZW1wZXJhdHVyZRgCIAEoCzIcLmdvb2dsZS5wcm90b2J1Zi5E
b3VibGVWYWx1ZVILdGVtcGVyYXR1cmUSMQoFdG9wX3AYAyABKAsyHC5nb29nbGUucHJvdG9idWYuRG91YmxlVmFsdWVSBHRvcFASMAoFdG9wX2sYBCABKAsy
Gy5nb29nbGUucHJvdG9idWYuSW50MzJWYWx1ZVIEdG9wSxISCgRzdG9wGAUgAygJUgRzdG9wEkAKD3Jlc3BvbnNlX2Zvcm1hdBgGIAEoCzIXLmdvb2dsZS5w
cm90b2J1Zi5TdHJ1Y3RSDnJlc3BvbnNlRm9ybWF0EjQKC3Rvb2xfY2hvaWNlGAcgASgLMhMubG0xNS52MS5Ub29sQ2hvaWNlUgp0b29sQ2hvaWNlEjAKCXJl
YXNvbmluZxgIIAEoCzISLmxtMTUudjEuUmVhc29uaW5nUglyZWFzb25pbmcSNwoKZXh0ZW5zaW9ucxgJIAEoCzIXLmdvb2dsZS5wcm90b2J1Zi5TdHJ1Y3RS
CmV4dGVuc2lvbnMi/QEKB1JlcXVlc3QSFAoFbW9kZWwYASABKAlSBW1vZGVsEiwKCG1lc3NhZ2VzGAIgAygLMhAubG0xNS52MS5NZXNzYWdlUghtZXNzYWdl
cxIuCgZzeXN0ZW0YAyABKAsyFi5sbTE1LnYxLlN5c3RlbUNvbnRlbnRSBnN5c3RlbRIjCgV0b29scxgEIAMoCzINLmxtMTUudjEuVG9vbFIFdG9vbHMSJwoG
Y29uZmlnGAUgASgLMg8ubG0xNS52MS5Db25maWdSBmNvbmZpZxIwCgVjYWNoZRgGIAEoCzIaLmdvb2dsZS5wcm90b2J1Zi5Cb29sVmFsdWVSBWNhY2hlIuYD
CgVVc2FnZRIhCgxpbnB1dF90b2tlbnMYASABKANSC2lucHV0VG9rZW5zEiMKDW91dHB1dF90b2tlbnMYAiABKANSDG91dHB1dFRva2VucxIhCgx0b3RhbF90
b2tlbnMYAyABKANSC3RvdGFsVG9rZW5zEkcKEWNhY2hlX3JlYWRfdG9rZW5zGAQgASgLMhsuZ29vZ2xlLnByb3RvYnVmLkludDY0VmFsdWVSD2NhY2hlUmVh
ZFRva2VucxJJChJjYWNoZV93cml0ZV90b2tlbnMYBSABKAsyGy5nb29nbGUucHJvdG9idWYuSW50NjRWYWx1ZVIQY2FjaGVXcml0ZVRva2VucxJGChByZWFz
b25pbmdfdG9rZW5zGAYgASgLMhsuZ29vZ2xlLnByb3RvYnVmLkludDY0VmFsdWVSD3JlYXNvbmluZ1Rva2VucxJJChJpbnB1dF9hdWRpb190b2tlbnMYByAB
KAsyGy5nb29nbGUucHJvdG9idWYuSW50NjRWYWx1ZVIQaW5wdXRBdWRpb1Rva2VucxJLChNvdXRwdXRfYXVkaW9fdG9rZW5zGAggASgLMhsuZ29vZ2xlLnBy
b3RvYnVmLkludDY0VmFsdWVSEW91dHB1dEF1ZGlvVG9rZW5zIvwBCghSZXNwb25zZRIOCgJpZBgBIAEoCVICaWQSFAoFbW9kZWwYAiABKAlSBW1vZGVsEioK
B21lc3NhZ2UYAyABKAsyEC5sbTE1LnYxLk1lc3NhZ2VSB21lc3NhZ2USOgoNZmluaXNoX3JlYXNvbhgEIAEoDjIVLmxtMTUudjEuRmluaXNoUmVhc29uUgxm
aW5pc2hSZWFzb24SJAoFdXNhZ2UYBSABKAsyDi5sbTE1LnYxLlVzYWdlUgV1c2FnZRI8Cg1wcm92aWRlcl9kYXRhGAYgASgLMhcuZ29vZ2xlLnByb3RvYnVm
LlN0cnVjdFIMcHJvdmlkZXJEYXRhInkKEEVtYmVkZGluZ1JlcXVlc3QSFAoFbW9kZWwYASABKAlSBW1vZGVsEhYKBmlucHV0cxgCIAMoCVIGaW5wdXRzEjcK
CmV4dGVuc2lvbnMYAyABKAsyFy5nb29nbGUucHJvdG9idWYuU3RydWN0UgpleHRlbnNpb25zIrgBChFFbWJlZGRpbmdSZXNwb25zZRIUCgVtb2RlbBgBIAEo
CVIFbW9kZWwSKQoHdmVjdG9ycxgCIAMoCzIPLmxtMTUudjEuVmVjdG9yUgd2ZWN0b3JzEiQKBXVzYWdlGAMgASgLMg4ubG0xNS52MS5Vc2FnZVIFdXNhZ2US
PAoNcHJvdmlkZXJfZGF0YRgEIAEoCzIXLmdvb2dsZS5wcm90b2J1Zi5TdHJ1Y3RSDHByb3ZpZGVyRGF0YSIgCgZWZWN0b3ISFgoGdmFsdWVzGAEgAygBUgZ2
YWx1ZXMi2gEKEUZpbGVVcGxvYWRSZXF1ZXN0EjIKBW1vZGVsGAEgASgLMhwuZ29vZ2xlLnByb3RvYnVmLlN0cmluZ1ZhbHVlUgVtb2RlbBIaCghmaWxlbmFt
ZRgCIAEoCVIIZmlsZW5hbWUSHQoKYnl0ZXNfZGF0YRgDIAEoDFIJYnl0ZXNEYXRhEh0KCm1lZGlhX3R5cGUYBCABKAlSCW1lZGlhVHlwZRI3CgpleHRlbnNp
b25zGAUgASgLMhcuZ29vZ2xlLnByb3RvYnVmLlN0cnVjdFIKZXh0ZW5zaW9ucyJiChJGaWxlVXBsb2FkUmVzcG9uc2USDgoCaWQYASABKAlSAmlkEjwKDXBy
b3ZpZGVyX2RhdGEYAiABKAsyFy5nb29nbGUucHJvdG9idWYuU3RydWN0Ugxwcm92aWRlckRhdGEiqQEKDEJhdGNoUmVxdWVzdBIyCgVtb2RlbBgBIAEoCzIc
Lmdvb2dsZS5wcm90b2J1Zi5TdHJpbmdWYWx1ZVIFbW9kZWwSLAoIcmVxdWVzdHMYAiADKAsyEC5sbTE1LnYxLlJlcXVlc3RSCHJlcXVlc3RzEjcKCmV4dGVu
c2lvbnMYAyABKAsyFy5nb29nbGUucHJvdG9idWYuU3RydWN0UgpleHRlbnNpb25zInUKDUJhdGNoUmVzcG9uc2USDgoCaWQYASABKAlSAmlkEhYKBnN0YXR1
cxgCIAEoCVIGc3RhdHVzEjwKDXByb3ZpZGVyX2RhdGEYAyABKAsyFy5nb29nbGUucHJvdG9idWYuU3RydWN0Ugxwcm92aWRlckRhdGEisQEKFkltYWdlR2Vu
ZXJhdGlvblJlcXVlc3QSFAoFbW9kZWwYASABKAlSBW1vZGVsEhYKBnByb21wdBgCIAEoCVIGcHJvbXB0EjAKBHNpemUYAyABKAsyHC5nb29nbGUucHJvdG9i
dWYuU3RyaW5nVmFsdWVSBHNpemUSNwoKZXh0ZW5zaW9ucxgEIAEoCzIXLmdvb2dsZS5wcm90b2J1Zi5TdHJ1Y3RSCmV4dGVuc2lvbnMiiwIKF0ltYWdlR2Vu
ZXJhdGlvblJlc3BvbnNlEioKBmltYWdlcxgBIAMoCzISLmxtMTUudjEuSW1hZ2VQYXJ0UgZpbWFnZXMSPAoNcHJvdmlkZXJfZGF0YRgCIAEoCzIXLmdvb2ds
ZS5wcm90b2J1Zi5TdHJ1Y3RSDHByb3ZpZGVyRGF0YRIsCgJpZBgDIAEoCzIcLmdvb2dsZS5wcm90b2J1Zi5TdHJpbmdWYWx1ZVICaWQSMgoFbW9kZWwYBCAB
KAsyHC5nb29nbGUucHJvdG9idWYuU3RyaW5nVmFsdWVSBW1vZGVsEiQKBXVzYWdlGAUgASgLMg4ubG0xNS52MS5Vc2FnZVIFdXNhZ2Ui6QEKFkF1ZGlvR2Vu
ZXJhdGlvblJlcXVlc3QSFAoFbW9kZWwYASABKAlSBW1vZGVsEhYKBnByb21wdBgCIAEoCVIGcHJvbXB0EjIKBXZvaWNlGAMgASgLMhwuZ29vZ2xlLnByb3Rv
YnVmLlN0cmluZ1ZhbHVlUgV2b2ljZRI0CgZmb3JtYXQYBCABKAsyHC5nb29nbGUucHJvdG9idWYuU3RyaW5nVmFsdWVSBmZvcm1hdBI3CgpleHRlbnNpb25z
GAUgASgLMhcuZ29vZ2xlLnByb3RvYnVmLlN0cnVjdFIKZXh0ZW5zaW9ucyKJAgoXQXVkaW9HZW5lcmF0aW9uUmVzcG9uc2USKAoFYXVkaW8YASABKAsyEi5s
bTE1LnYxLkF1ZGlvUGFydFIFYXVkaW8SPAoNcHJvdmlkZXJfZGF0YRgCIAEoCzIXLmdvb2dsZS5wcm90b2J1Zi5TdHJ1Y3RSDHByb3ZpZGVyRGF0YRIsCgJp
ZBgDIAEoCzIcLmdvb2dsZS5wcm90b2J1Zi5TdHJpbmdWYWx1ZVICaWQSMgoFbW9kZWwYBCABKAsyHC5nb29nbGUucHJvdG9idWYuU3RyaW5nVmFsdWVSBW1v
ZGVsEiQKBXVzYWdlGAUgASgLMg4ubG0xNS52MS5Vc2FnZVIFdXNhZ2Ui1wMKD0VuZHBvaW50UmVxdWVzdBIsCgdyZXF1ZXN0GAEgASgLMhAubG0xNS52MS5S
ZXF1ZXN0SABSB3JlcXVlc3QSSAoRZW1iZWRkaW5nX3JlcXVlc3QYAiABKAsyGS5sbTE1LnYxLkVtYmVkZGluZ1JlcXVlc3RIAFIQZW1iZWRkaW5nUmVxdWVz
dBJMChNmaWxlX3VwbG9hZF9yZXF1ZXN0GAMgASgLMhoubG0xNS52MS5GaWxlVXBsb2FkUmVxdWVzdEgAUhFmaWxlVXBsb2FkUmVxdWVzdBI8Cg1iYXRjaF9y
ZXF1ZXN0GAQgASgLMhUubG0xNS52MS5CYXRjaFJlcXVlc3RIAFIMYmF0Y2hSZXF1ZXN0ElsKGGltYWdlX2dlbmVyYXRpb25fcmVxdWVzdBgFIAEoCzIfLmxt
MTUudjEuSW1hZ2VHZW5lcmF0aW9uUmVxdWVzdEgAUhZpbWFnZUdlbmVyYXRpb25SZXF1ZXN0ElsKGGF1ZGlvX2dlbmVyYXRpb25fcmVxdWVzdBgGIAEoCzIf
LmxtMTUudjEuQXVkaW9HZW5lcmF0aW9uUmVxdWVzdEgAUhZhdWRpb0dlbmVyYXRpb25SZXF1ZXN0QgYKBGtpbmQi6gMKEEVuZHBvaW50UmVzcG9uc2USLwoI
cmVzcG9uc2UYASABKAsyES5sbTE1LnYxLlJlc3BvbnNlSABSCHJlc3BvbnNlEksKEmVtYmVkZGluZ19yZXNwb25zZRgCIAEoCzIaLmxtMTUudjEuRW1iZWRk
aW5nUmVzcG9uc2VIAFIRZW1iZWRkaW5nUmVzcG9uc2USTwoUZmlsZV91cGxvYWRfcmVzcG9uc2UYAyABKAsyGy5sbTE1LnYxLkZpbGVVcGxvYWRSZXNwb25z
ZUgAUhJmaWxlVXBsb2FkUmVzcG9uc2USPwoOYmF0Y2hfcmVzcG9uc2UYBCABKAsyFi5sbTE1LnYxLkJhdGNoUmVzcG9uc2VIAFINYmF0Y2hSZXNwb25zZRJe
ChlpbWFnZV9nZW5lcmF0aW9uX3Jlc3BvbnNlGAUgASgLMiAubG0xNS52MS5JbWFnZUdlbmVyYXRpb25SZXNwb25zZUgAUhdpbWFnZUdlbmVyYXRpb25SZXNw
b25zZRJeChlhdWRpb19nZW5lcmF0aW9uX3Jlc3BvbnNlGAYgASgLMiAubG0xNS52MS5BdWRpb0dlbmVyYXRpb25SZXNwb25zZUgAUhdhdWRpb0dlbmVyYXRp
b25SZXNwb25zZUIGCgRraW5kIn4KC0F1ZGlvRm9ybWF0EjIKCGVuY29kaW5nGAEgASgOMhYubG0xNS52MS5BdWRpb0VuY29kaW5nUghlbmNvZGluZxIfCgtz
YW1wbGVfcmF0ZRgCIAEoBVIKc2FtcGxlUmF0ZRIaCghjaGFubmVscxgDIAEoBVIIY2hhbm5lbHMi2AIKCkxpdmVDb25maWcSFAoFbW9kZWwYASABKAlSBW1v
ZGVsEi4KBnN5c3RlbRgCIAEoCzIWLmxtMTUudjEuU3lzdGVtQ29udGVudFIGc3lzdGVtEiMKBXRvb2xzGAMgAygLMg0ubG0xNS52MS5Ub29sUgV0b29scxIy
CgV2b2ljZRgEIAEoCzIcLmdvb2dsZS5wcm90b2J1Zi5TdHJpbmdWYWx1ZVIFdm9pY2USNwoMaW5wdXRfZm9ybWF0GAUgASgLMhQubG0xNS52MS5BdWRpb0Zv
cm1hdFILaW5wdXRGb3JtYXQSOQoNb3V0cHV0X2Zvcm1hdBgGIAEoCzIULmxtMTUudjEuQXVkaW9Gb3JtYXRSDG91dHB1dEZvcm1hdBI3CgpleHRlbnNpb25z
GAcgASgLMhcuZ29vZ2xlLnByb3RvYnVmLlN0cnVjdFIKZXh0ZW5zaW9ucyLpAgoPTGl2ZUNsaWVudEV2ZW50EjAKBWF1ZGlvGAEgASgLMhgubG0xNS52MS5M
aXZlQ2xpZW50QXVkaW9IAFIFYXVkaW8SMAoFdmlkZW8YAiABKAsyGC5sbTE1LnYxLkxpdmVDbGllbnRWaWRlb0gAUgV2aWRlbxItCgR0ZXh0GAMgASgLMhcu
bG0xNS52MS5MaXZlQ2xpZW50VGV4dEgAUgR0ZXh0EkAKC3Rvb2xfcmVzdWx0GAQgASgLMh0ubG0xNS52MS5MaXZlQ2xpZW50VG9vbFJlc3VsdEgAUgp0b29s
UmVzdWx0EjwKCWludGVycnVwdBgFIAEoCzIcLmxtMTUudjEuTGl2ZUNsaWVudEludGVycnVwdEgAUglpbnRlcnJ1cHQSOgoJZW5kX2F1ZGlvGAYgASgLMhsu
bG0xNS52MS5MaXZlQ2xpZW50RW5kQXVkaW9IAFIIZW5kQXVkaW9CBwoFZXZlbnQiJQoPTGl2ZUNsaWVudEF1ZGlvEhIKBGRhdGEYASABKAxSBGRhdGEiJQoP
TGl2ZUNsaWVudFZpZGVvEhIKBGRhdGEYASABKAxSBGRhdGEiJAoOTGl2ZUNsaWVudFRleHQSEgoEdGV4dBgBIAEoCVIEdGV4dCJPChRMaXZlQ2xpZW50VG9v
bFJlc3VsdBIOCgJpZBgBIAEoCVICaWQSJwoHY29udGVudBgCIAMoCzINLmxtMTUudjEuUGFydFIHY29udGVudCIVChNMaXZlQ2xpZW50SW50ZXJydXB0IhQK
EkxpdmVDbGllbnRFbmRBdWRpbyKyAwoPTGl2ZVNlcnZlckV2ZW50EjAKBWF1ZGlvGAEgASgLMhgubG0xNS52MS5MaXZlU2VydmVyQXVkaW9IAFIFYXVkaW8S
LQoEdGV4dBgCIAEoCzIXLmxtMTUudjEuTGl2ZVNlcnZlclRleHRIAFIEdGV4dBI6Cgl0b29sX2NhbGwYAyABKAsyGy5sbTE1LnYxLkxpdmVTZXJ2ZXJUb29s
Q2FsbEgAUgh0b29sQ2FsbBJCCgtpbnRlcnJ1cHRlZBgEIAEoCzIeLmxtMTUudjEuTGl2ZVNlcnZlckludGVycnVwdGVkSABSC2ludGVycnVwdGVkEjcKCHR1
cm5fZW5kGAUgASgLMhoubG0xNS52MS5MaXZlU2VydmVyVHVybkVuZEgAUgd0dXJuRW5kEjAKBWVycm9yGAYgASgLMhgubG0xNS52MS5MaXZlU2VydmVyRXJy
b3JIAFIFZXJyb3ISSgoPdG9vbF9jYWxsX2RlbHRhGAcgASgLMiAubG0xNS52MS5MaXZlU2VydmVyVG9vbENhbGxEZWx0YUgAUg10b29sQ2FsbERlbHRhQgcK
BWV2ZW50IiUKD0xpdmVTZXJ2ZXJBdWRpbxISCgRkYXRhGAEgASgMUgRkYXRhIiQKDkxpdmVTZXJ2ZXJUZXh0EhIKBHRleHQYASABKAlSBHRleHQiZwoSTGl2
ZVNlcnZlclRvb2xDYWxsEg4KAmlkGAEgASgJUgJpZBISCgRuYW1lGAIgASgJUgRuYW1lEi0KBWlucHV0GAMgASgLMhcuZ29vZ2xlLnByb3RvYnVmLlN0cnVj
dFIFaW5wdXQimgEKF0xpdmVTZXJ2ZXJUb29sQ2FsbERlbHRhEh8KC2lucHV0X2RlbHRhGAEgASgJUgppbnB1dERlbHRhEiwKAmlkGAIgASgLMhwuZ29vZ2xl
LnByb3RvYnVmLlN0cmluZ1ZhbHVlUgJpZBIwCgRuYW1lGAMgASgLMhwuZ29vZ2xlLnByb3RvYnVmLlN0cmluZ1ZhbHVlUgRuYW1lIhcKFUxpdmVTZXJ2ZXJJ
bnRlcnJ1cHRlZCI5ChFMaXZlU2VydmVyVHVybkVuZBIkCgV1c2FnZRgBIAEoCzIOLmxtMTUudjEuVXNhZ2VSBXVzYWdlIj0KD0xpdmVTZXJ2ZXJFcnJvchIq
CgVlcnJvchgBIAEoCzIULmxtMTUudjEuRXJyb3JEZXRhaWxSBWVycm9yKmIKBFJvbGUSFAoQUk9MRV9VTlNQRUNJRklFRBAAEg0KCVJPTEVfVVNFUhABEhIK
DlJPTEVfQVNTSVNUQU5UEAISDQoJUk9MRV9UT09MEAMSEgoOUk9MRV9ERVZFTE9QRVIQBCq3AQoMRmluaXNoUmVhc29uEh0KGUZJTklTSF9SRUFTT05fVU5T
UEVDSUZJRUQQABIWChJGSU5JU0hfUkVBU09OX1NUT1AQARIYChRGSU5JU0hfUkVBU09OX0xFTkdUSBACEhsKF0ZJTklTSF9SRUFTT05fVE9PTF9DQUxMEAMS
IAocRklOSVNIX1JFQVNPTl9DT05URU5UX0ZJTFRFUhAEEhcKE0ZJTklTSF9SRUFTT05fRVJST1IQBSr4AQoPUmVhc29uaW5nRWZmb3J0EiAKHFJFQVNPTklO
R19FRkZPUlRfVU5TUEVDSUZJRUQQABIYChRSRUFTT05JTkdfRUZGT1JUX09GRhABEh0KGVJFQVNPTklOR19FRkZPUlRfQURBUFRJVkUQAhIcChhSRUFTT05J
TkdfRUZGT1JUX01JTklNQUwQAxIYChRSRUFTT05JTkdfRUZGT1JUX0xPVxAEEhsKF1JFQVNPTklOR19FRkZPUlRfTUVESVVNEAUSGQoVUkVBU09OSU5HX0VG
Rk9SVF9ISUdIEAYSGgoWUkVBU09OSU5HX0VGRk9SVF9YSElHSBAHKvUCCglFcnJvckNvZGUSGgoWRVJST1JfQ09ERV9VTlNQRUNJRklFRBAAEhMKD0VSUk9S
X0NPREVfQVVUSBABEhYKEkVSUk9SX0NPREVfQklMTElORxACEhkKFUVSUk9SX0NPREVfUkFURV9MSU1JVBADEh4KGkVSUk9SX0NPREVfSU5WQUxJRF9SRVFV
RVNUEAQSHQoZRVJST1JfQ09ERV9DT05URVhUX0xFTkdUSBAFEhYKEkVSUk9SX0NPREVfVElNRU9VVBAGEhUKEUVSUk9SX0NPREVfU0VSVkVSEAcSFwoTRVJS
T1JfQ09ERV9QUk9WSURFUhAIEiAKHEVSUk9SX0NPREVfVU5TVVBQT1JURURfTU9ERUwQCRIiCh5FUlJPUl9DT0RFX1VOU1VQUE9SVEVEX0ZFQVRVUkUQChId
ChlFUlJPUl9DT0RFX05PVF9DT05GSUdVUkVEEAsSGAoURVJST1JfQ09ERV9UUkFOU1BPUlQQDCpvCgtJbWFnZURldGFpbBIcChhJTUFHRV9ERVRBSUxfVU5T
UEVDSUZJRUQQABIUChBJTUFHRV9ERVRBSUxfTE9XEAESFQoRSU1BR0VfREVUQUlMX0hJR0gQAhIVChFJTUFHRV9ERVRBSUxfQVVUTxADKocBCg5Ub29sQ2hv
aWNlTW9kZRIgChxUT09MX0NIT0lDRV9NT0RFX1VOU1BFQ0lGSUVEEAASGQoVVE9PTF9DSE9JQ0VfTU9ERV9BVVRPEAESHQoZVE9PTF9DSE9JQ0VfTU9ERV9S
RVFVSVJFRBACEhkKFVRPT0xfQ0hPSUNFX01PREVfTk9ORRADKpIBCg1BdWRpb0VuY29kaW5nEh4KGkFVRElPX0VOQ09ESU5HX1VOU1BFQ0lGSUVEEAASGAoU
QVVESU9fRU5DT0RJTkdfUENNMTYQARIXChNBVURJT19FTkNPRElOR19PUFVTEAISFgoSQVVESU9fRU5DT0RJTkdfTVAzEAMSFgoSQVVESU9fRU5DT0RJTkdf
QUFDEARCQwoLZGV2LmxtMTUudjFCCUxtMTVQcm90b1ABWidnaXRodWIuY29tL2xtMTUvbG0xNS9nZW4vbG0xNS92MTtsbTE1djFiBnByb3RvMw==
"""


def descriptor_set_bytes() -> bytes:
    """Return the embedded ``FileDescriptorSet`` bytes for ``lm15.v1``."""
    return base64.b64decode(_DESCRIPTOR_SET_B64)


@lru_cache(maxsize=1)
def proto_module() -> SimpleNamespace:
    """Return dynamic protobuf classes/enums for the canonical lm15 schema."""
    fds = descriptor_pb2.FileDescriptorSet()
    fds.ParseFromString(descriptor_set_bytes())

    pool = descriptor_pool.DescriptorPool()
    for file_descriptor in fds.file:
        pool.Add(file_descriptor)

    def cls(name: str):
        descriptor = pool.FindMessageTypeByName(f"lm15.v1.{name}")
        return message_factory.GetMessageClass(descriptor)

    def enum_value(enum_name: str, value_name: str) -> int:
        enum = pool.FindEnumTypeByName(f"lm15.v1.{enum_name}")
        return enum.values_by_name[value_name].number

    names = [
        "Part",
        "MediaSource",
        "TextPart",
        "ImagePart",
        "AudioPart",
        "VideoPart",
        "DocumentPart",
        "ToolCallPart",
        "ToolResultPart",
        "ThinkingPart",
        "RefusalPart",
        "CitationPart",
        "Message",
        "SystemContent",
        "Delta",
        "StreamEvent",
        "Request",
        "Response",
        "Usage",
        "Tool",
        "ToolChoice",
        "Reasoning",
        "Config",
        "ErrorDetail",
        "EmbeddingRequest",
        "EmbeddingResponse",
        "Vector",
        "FileUploadRequest",
        "FileUploadResponse",
        "BatchRequest",
        "BatchResponse",
        "ImageGenerationRequest",
        "ImageGenerationResponse",
        "AudioGenerationRequest",
        "AudioGenerationResponse",
        "EndpointRequest",
        "EndpointResponse",
        "AudioFormat",
        "LiveConfig",
        "LiveClientEvent",
        "LiveServerEvent",
        "LiveServerToolCallDelta",
        "ToolCallInfo",
    ]
    ns = SimpleNamespace(**{name: cls(name) for name in names})

    enum_names = {
        "ROLE_UNSPECIFIED": ("Role", "ROLE_UNSPECIFIED"),
        "ROLE_USER": ("Role", "ROLE_USER"),
        "ROLE_ASSISTANT": ("Role", "ROLE_ASSISTANT"),
        "ROLE_TOOL": ("Role", "ROLE_TOOL"),
        "ROLE_DEVELOPER": ("Role", "ROLE_DEVELOPER"),
        "FINISH_REASON_UNSPECIFIED": ("FinishReason", "FINISH_REASON_UNSPECIFIED"),
        "FINISH_REASON_STOP": ("FinishReason", "FINISH_REASON_STOP"),
        "FINISH_REASON_LENGTH": ("FinishReason", "FINISH_REASON_LENGTH"),
        "FINISH_REASON_TOOL_CALL": ("FinishReason", "FINISH_REASON_TOOL_CALL"),
        "FINISH_REASON_CONTENT_FILTER": ("FinishReason", "FINISH_REASON_CONTENT_FILTER"),
        "FINISH_REASON_ERROR": ("FinishReason", "FINISH_REASON_ERROR"),
        "REASONING_EFFORT_OFF": ("ReasoningEffort", "REASONING_EFFORT_OFF"),
        "REASONING_EFFORT_ADAPTIVE": ("ReasoningEffort", "REASONING_EFFORT_ADAPTIVE"),
        "REASONING_EFFORT_MINIMAL": ("ReasoningEffort", "REASONING_EFFORT_MINIMAL"),
        "REASONING_EFFORT_LOW": ("ReasoningEffort", "REASONING_EFFORT_LOW"),
        "REASONING_EFFORT_MEDIUM": ("ReasoningEffort", "REASONING_EFFORT_MEDIUM"),
        "REASONING_EFFORT_HIGH": ("ReasoningEffort", "REASONING_EFFORT_HIGH"),
        "REASONING_EFFORT_XHIGH": ("ReasoningEffort", "REASONING_EFFORT_XHIGH"),
        "ERROR_CODE_AUTH": ("ErrorCode", "ERROR_CODE_AUTH"),
        "ERROR_CODE_BILLING": ("ErrorCode", "ERROR_CODE_BILLING"),
        "ERROR_CODE_RATE_LIMIT": ("ErrorCode", "ERROR_CODE_RATE_LIMIT"),
        "ERROR_CODE_INVALID_REQUEST": ("ErrorCode", "ERROR_CODE_INVALID_REQUEST"),
        "ERROR_CODE_CONTEXT_LENGTH": ("ErrorCode", "ERROR_CODE_CONTEXT_LENGTH"),
        "ERROR_CODE_TIMEOUT": ("ErrorCode", "ERROR_CODE_TIMEOUT"),
        "ERROR_CODE_SERVER": ("ErrorCode", "ERROR_CODE_SERVER"),
        "ERROR_CODE_PROVIDER": ("ErrorCode", "ERROR_CODE_PROVIDER"),
        "ERROR_CODE_UNSUPPORTED_MODEL": ("ErrorCode", "ERROR_CODE_UNSUPPORTED_MODEL"),
        "ERROR_CODE_UNSUPPORTED_FEATURE": ("ErrorCode", "ERROR_CODE_UNSUPPORTED_FEATURE"),
        "ERROR_CODE_NOT_CONFIGURED": ("ErrorCode", "ERROR_CODE_NOT_CONFIGURED"),
        "ERROR_CODE_TRANSPORT": ("ErrorCode", "ERROR_CODE_TRANSPORT"),
        "IMAGE_DETAIL_LOW": ("ImageDetail", "IMAGE_DETAIL_LOW"),
        "IMAGE_DETAIL_HIGH": ("ImageDetail", "IMAGE_DETAIL_HIGH"),
        "IMAGE_DETAIL_AUTO": ("ImageDetail", "IMAGE_DETAIL_AUTO"),
        "TOOL_CHOICE_MODE_AUTO": ("ToolChoiceMode", "TOOL_CHOICE_MODE_AUTO"),
        "TOOL_CHOICE_MODE_REQUIRED": ("ToolChoiceMode", "TOOL_CHOICE_MODE_REQUIRED"),
        "TOOL_CHOICE_MODE_NONE": ("ToolChoiceMode", "TOOL_CHOICE_MODE_NONE"),
        "AUDIO_ENCODING_PCM16": ("AudioEncoding", "AUDIO_ENCODING_PCM16"),
        "AUDIO_ENCODING_OPUS": ("AudioEncoding", "AUDIO_ENCODING_OPUS"),
        "AUDIO_ENCODING_MP3": ("AudioEncoding", "AUDIO_ENCODING_MP3"),
        "AUDIO_ENCODING_AAC": ("AudioEncoding", "AUDIO_ENCODING_AAC"),
    }
    for attr, (enum_name, value_name) in enum_names.items():
        setattr(ns, attr, enum_value(enum_name, value_name))
    return ns


def _pb(pb: Any | None = None) -> Any:
    return pb if pb is not None else proto_module()


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


# ─── Generic protobuf helpers ────────────────────────────────────────


def _set_wrapper(wrapper: Any, value: Any) -> None:
    wrapper.value = value


def _wrapper_value(msg: Any, field: str) -> Any | None:
    return getattr(msg, field).value if msg.HasField(field) else None


def _copy_struct(target: Any, value: dict[str, Any] | None) -> None:
    if value is not None:
        ParseDict(value, target)


def _struct_value(msg: Any, field: str) -> dict[str, Any] | None:
    if not msg.HasField(field):
        return None
    return MessageToDict(getattr(msg, field), preserving_proto_field_name=True)


# ─── Enum maps ───────────────────────────────────────────────────────


def _maps(pb):
    role_to_proto = {
        "user": pb.ROLE_USER,
        "assistant": pb.ROLE_ASSISTANT,
        "tool": pb.ROLE_TOOL,
        "developer": pb.ROLE_DEVELOPER,
    }
    finish_to_proto = {
        "stop": pb.FINISH_REASON_STOP,
        "length": pb.FINISH_REASON_LENGTH,
        "tool_call": pb.FINISH_REASON_TOOL_CALL,
        "content_filter": pb.FINISH_REASON_CONTENT_FILTER,
        "error": pb.FINISH_REASON_ERROR,
    }
    effort_to_proto = {
        "off": pb.REASONING_EFFORT_OFF,
        "adaptive": pb.REASONING_EFFORT_ADAPTIVE,
        "minimal": pb.REASONING_EFFORT_MINIMAL,
        "low": pb.REASONING_EFFORT_LOW,
        "medium": pb.REASONING_EFFORT_MEDIUM,
        "high": pb.REASONING_EFFORT_HIGH,
        "xhigh": pb.REASONING_EFFORT_XHIGH,
    }
    error_to_proto = {
        "auth": pb.ERROR_CODE_AUTH,
        "billing": pb.ERROR_CODE_BILLING,
        "rate_limit": pb.ERROR_CODE_RATE_LIMIT,
        "invalid_request": pb.ERROR_CODE_INVALID_REQUEST,
        "context_length": pb.ERROR_CODE_CONTEXT_LENGTH,
        "timeout": pb.ERROR_CODE_TIMEOUT,
        "server": pb.ERROR_CODE_SERVER,
        "provider": pb.ERROR_CODE_PROVIDER,
        "unsupported_model": pb.ERROR_CODE_UNSUPPORTED_MODEL,
        "unsupported_feature": pb.ERROR_CODE_UNSUPPORTED_FEATURE,
        "not_configured": pb.ERROR_CODE_NOT_CONFIGURED,
        "transport": pb.ERROR_CODE_TRANSPORT,
    }
    detail_to_proto = {
        "low": pb.IMAGE_DETAIL_LOW,
        "high": pb.IMAGE_DETAIL_HIGH,
        "auto": pb.IMAGE_DETAIL_AUTO,
    }
    mode_to_proto = {
        "auto": pb.TOOL_CHOICE_MODE_AUTO,
        "required": pb.TOOL_CHOICE_MODE_REQUIRED,
        "none": pb.TOOL_CHOICE_MODE_NONE,
    }
    encoding_to_proto = {
        "pcm16": pb.AUDIO_ENCODING_PCM16,
        "opus": pb.AUDIO_ENCODING_OPUS,
        "mp3": pb.AUDIO_ENCODING_MP3,
        "aac": pb.AUDIO_ENCODING_AAC,
    }
    return {
        "role": (role_to_proto, {v: k for k, v in role_to_proto.items()}),
        "finish": (finish_to_proto, {v: k for k, v in finish_to_proto.items()}),
        "effort": (effort_to_proto, {v: k for k, v in effort_to_proto.items()}),
        "error": (error_to_proto, {v: k for k, v in error_to_proto.items()}),
        "detail": (detail_to_proto, {v: k for k, v in detail_to_proto.items()}),
        "mode": (mode_to_proto, {v: k for k, v in mode_to_proto.items()}),
        "encoding": (encoding_to_proto, {v: k for k, v in encoding_to_proto.items()}),
    }


# ─── Python -> protobuf ──────────────────────────────────────────────


def _media_source_to_proto(pb, part: ImagePart | AudioPart | VideoPart | DocumentPart):
    out = pb.MediaSource(media_type=part.media_type)
    if part.data is not None or part.path is not None:
        out.data = part.bytes
    elif part.url is not None:
        out.url = part.url
    elif part.file_id is not None:
        out.file_id = part.file_id
    return out


def _part_to_proto(pb, part: Part):
    maps = _maps(pb)
    out = pb.Part()
    if isinstance(part, TextPart):
        out.text.text = part.text
    elif isinstance(part, ImagePart):
        out.image.source.CopyFrom(_media_source_to_proto(pb, part))
        if part.detail is not None:
            out.image.detail = maps["detail"][0][part.detail]
    elif isinstance(part, AudioPart):
        out.audio.source.CopyFrom(_media_source_to_proto(pb, part))
    elif isinstance(part, VideoPart):
        out.video.source.CopyFrom(_media_source_to_proto(pb, part))
    elif isinstance(part, DocumentPart):
        out.document.source.CopyFrom(_media_source_to_proto(pb, part))
    elif isinstance(part, ToolCallPart):
        out.tool_call.id = part.id
        out.tool_call.name = part.name
        _copy_struct(out.tool_call.input, part.input)
    elif isinstance(part, ToolResultPart):
        out.tool_result.id = part.id
        out.tool_result.content.extend(_part_to_proto(pb, p) for p in part.content)
        if part.name is not None:
            _set_wrapper(out.tool_result.name, part.name)
        out.tool_result.is_error = part.is_error
    elif isinstance(part, ThinkingPart):
        out.thinking.text = part.text
        out.thinking.redacted = part.redacted
    elif isinstance(part, RefusalPart):
        out.refusal.text = part.text
    elif isinstance(part, CitationPart):
        if part.url is not None:
            _set_wrapper(out.citation.url, part.url)
        if part.title is not None:
            _set_wrapper(out.citation.title, part.title)
        if part.text is not None:
            _set_wrapper(out.citation.text, part.text)
    else:
        raise TypeError(type(part))
    return out


def _message_to_proto(pb, message: Message):
    out = pb.Message(role=_maps(pb)["role"][0][message.role])
    out.parts.extend(_part_to_proto(pb, p) for p in message.parts)
    return out


def _system_to_proto(pb, system: str | tuple[Part, ...] | None):
    if system is None:
        return None
    out = pb.SystemContent()
    if isinstance(system, str):
        out.text = system
    else:
        out.parts.parts.extend(_part_to_proto(pb, p) for p in system)
    return out


def _tool_to_proto(pb, tool: Tool):
    out = pb.Tool()
    if isinstance(tool, FunctionTool):
        out.function.name = tool.name
        if tool.description is not None:
            _set_wrapper(out.function.description, tool.description)
        _copy_struct(out.function.parameters, tool.parameters)
    elif isinstance(tool, BuiltinTool):
        out.builtin.name = tool.name
        _copy_struct(out.builtin.config, tool.config)
    else:
        raise TypeError(type(tool))
    return out


def _tool_choice_to_proto(pb, choice: ToolChoice):
    out = pb.ToolChoice(mode=_maps(pb)["mode"][0][choice.mode])
    out.allowed.extend(choice.allowed)
    if choice.parallel is not None:
        _set_wrapper(out.parallel, choice.parallel)
    return out


def _reasoning_to_proto(pb, reasoning: Reasoning):
    out = pb.Reasoning(effort=_maps(pb)["effort"][0][reasoning.effort])
    if reasoning.thinking_budget is not None:
        _set_wrapper(out.thinking_budget, reasoning.thinking_budget)
    if reasoning.total_budget is not None:
        _set_wrapper(out.total_budget, reasoning.total_budget)
    if reasoning.summary is not None:
        _set_wrapper(out.summary, reasoning.summary)
    return out


def _config_to_proto(pb, config: Config):
    out = pb.Config()
    if config.max_tokens is not None:
        _set_wrapper(out.max_tokens, config.max_tokens)
    if config.temperature is not None:
        _set_wrapper(out.temperature, config.temperature)
    if config.top_p is not None:
        _set_wrapper(out.top_p, config.top_p)
    if config.top_k is not None:
        _set_wrapper(out.top_k, config.top_k)
    out.stop.extend(config.stop)
    _copy_struct(out.response_format, config.response_format)
    if config.tool_choice is not None:
        out.tool_choice.CopyFrom(_tool_choice_to_proto(pb, config.tool_choice))
    if config.reasoning is not None:
        out.reasoning.CopyFrom(_reasoning_to_proto(pb, config.reasoning))
    _copy_struct(out.extensions, config.extensions)
    return out


def _usage_to_proto(pb, usage: Usage):
    out = pb.Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )
    for field in (
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "input_audio_tokens",
        "output_audio_tokens",
    ):
        value = getattr(usage, field)
        if value is not None:
            _set_wrapper(getattr(out, field), value)
    return out


def _error_to_proto(pb, error: ErrorDetail):
    out = pb.ErrorDetail(
        code=_maps(pb)["error"][0][error.code],
        message=error.message,
    )
    if error.provider_code is not None:
        _set_wrapper(out.provider_code, error.provider_code)
    return out


def _delta_to_proto(pb, delta):
    out = pb.Delta()
    if isinstance(delta, TextDelta):
        out.text.text = delta.text
        out.text.part_index = delta.part_index
    elif isinstance(delta, ThinkingDelta):
        out.thinking.text = delta.text
        out.thinking.part_index = delta.part_index
    elif isinstance(delta, AudioDelta):
        out.audio.part_index = delta.part_index
        if delta.media_type is not None:
            _set_wrapper(out.audio.media_type, delta.media_type)
        if delta.data is not None:
            out.audio.data = delta.data
        elif delta.url is not None:
            out.audio.url = delta.url
        elif delta.file_id is not None:
            out.audio.file_id = delta.file_id
    elif isinstance(delta, ImageDelta):
        out.image.part_index = delta.part_index
        if delta.media_type is not None:
            _set_wrapper(out.image.media_type, delta.media_type)
        if delta.data is not None:
            out.image.data = delta.data
        elif delta.url is not None:
            out.image.url = delta.url
        elif delta.file_id is not None:
            out.image.file_id = delta.file_id
    elif isinstance(delta, ToolCallDelta):
        out.tool_call.input = delta.input
        out.tool_call.part_index = delta.part_index
        if delta.id is not None:
            _set_wrapper(out.tool_call.id, delta.id)
        if delta.name is not None:
            _set_wrapper(out.tool_call.name, delta.name)
    elif isinstance(delta, CitationDelta):
        if delta.text is not None:
            _set_wrapper(out.citation.text, delta.text)
        if delta.url is not None:
            _set_wrapper(out.citation.url, delta.url)
        if delta.title is not None:
            _set_wrapper(out.citation.title, delta.title)
        out.citation.part_index = delta.part_index
    else:
        raise TypeError(type(delta))
    return out


def _stream_event_to_proto(pb, event: StreamEvent):
    out = pb.StreamEvent()
    if event.type == "start":
        out.start.SetInParent()
        if event.id is not None:
            _set_wrapper(out.start.id, event.id)
        if event.model is not None:
            _set_wrapper(out.start.model, event.model)
    elif event.type == "delta":
        assert event.delta is not None
        out.delta.delta.CopyFrom(_delta_to_proto(pb, event.delta))
    elif event.type == "end":
        out.end.SetInParent()
        if event.finish_reason is not None:
            out.end.finish_reason = _maps(pb)["finish"][0][event.finish_reason]
        if event.usage is not None:
            out.end.usage.CopyFrom(_usage_to_proto(pb, event.usage))
        _copy_struct(out.end.provider_data, event.provider_data)
    elif event.type == "error":
        assert event.error is not None
        out.error.error.CopyFrom(_error_to_proto(pb, event.error))
    else:
        raise ValueError(event.type)
    return out


def _request_to_proto(pb, request: Request):
    out = pb.Request(model=request.model)
    out.messages.extend(_message_to_proto(pb, m) for m in request.messages)
    system = _system_to_proto(pb, request.system)
    if system is not None:
        out.system.CopyFrom(system)
    out.tools.extend(_tool_to_proto(pb, t) for t in request.tools)
    out.config.CopyFrom(_config_to_proto(pb, request.config))
    return out


def _response_to_proto(pb, response: Response):
    out = pb.Response(
        id=response.id or "",
        model=response.model,
        finish_reason=_maps(pb)["finish"][0][response.finish_reason],
    )
    out.message.CopyFrom(_message_to_proto(pb, response.message))
    out.usage.CopyFrom(_usage_to_proto(pb, response.usage))
    _copy_struct(out.provider_data, response.provider_data)
    return out


def _audio_format_to_proto(pb, audio_format: AudioFormat):
    return pb.AudioFormat(
        encoding=_maps(pb)["encoding"][0][audio_format.encoding],
        sample_rate=audio_format.sample_rate,
        channels=audio_format.channels,
    )


def _live_config_to_proto(pb, config: LiveConfig):
    out = pb.LiveConfig(model=config.model)
    system = _system_to_proto(pb, config.system)
    if system is not None:
        out.system.CopyFrom(system)
    out.tools.extend(_tool_to_proto(pb, t) for t in config.tools)
    if config.voice is not None:
        _set_wrapper(out.voice, config.voice)
    if config.input_format is not None:
        out.input_format.CopyFrom(_audio_format_to_proto(pb, config.input_format))
    if config.output_format is not None:
        out.output_format.CopyFrom(_audio_format_to_proto(pb, config.output_format))
    _copy_struct(out.extensions, config.extensions)
    return out


def _live_client_event_to_proto(pb, event: LiveClientEvent):
    out = pb.LiveClientEvent()
    if event.type == "audio":
        out.audio.data = base64.b64decode(event.data or "")
    elif event.type == "video":
        out.video.data = base64.b64decode(event.data or "")
    elif event.type == "text":
        out.text.text = event.text or ""
    elif event.type == "tool_result":
        out.tool_result.id = event.id or ""
        out.tool_result.content.extend(_part_to_proto(pb, p) for p in event.content)
    elif event.type == "interrupt":
        out.interrupt.SetInParent()
    elif event.type == "end_audio":
        out.end_audio.SetInParent()
    else:
        raise ValueError(event.type)
    return out


def _live_server_event_to_proto(pb, event: LiveServerEvent):
    out = pb.LiveServerEvent()
    if event.type == "audio":
        out.audio.data = base64.b64decode(event.data or "")
    elif event.type == "text":
        out.text.text = event.text or ""
    elif event.type == "tool_call":
        out.tool_call.id = event.id or ""
        out.tool_call.name = event.name or ""
        _copy_struct(out.tool_call.input, event.input)
    elif event.type == "tool_call_delta":
        out.tool_call_delta.input_delta = event.input_delta or ""
        if event.id is not None:
            _set_wrapper(out.tool_call_delta.id, event.id)
        if event.name is not None:
            _set_wrapper(out.tool_call_delta.name, event.name)
    elif event.type == "interrupted":
        out.interrupted.SetInParent()
    elif event.type == "turn_end":
        assert event.usage is not None
        out.turn_end.usage.CopyFrom(_usage_to_proto(pb, event.usage))
    elif event.type == "error":
        assert event.error is not None
        out.error.error.CopyFrom(_error_to_proto(pb, event.error))
    else:
        raise ValueError(event.type)
    return out


# ─── protobuf -> Python ──────────────────────────────────────────────


def _media_source_from_proto(msg):
    source = msg.WhichOneof("source")
    kwargs: dict[str, Any] = {"media_type": msg.media_type}
    if source == "data":
        kwargs["data"] = _b64(msg.data)
    elif source == "url":
        kwargs["url"] = msg.url
    elif source == "file_id":
        kwargs["file_id"] = msg.file_id
    else:
        raise ValueError("media source missing")
    return kwargs


def _part_from_proto(pb, msg) -> Part:
    maps = _maps(pb)
    kind = msg.WhichOneof("kind")
    if kind == "text":
        return TextPart(text=msg.text.text)
    if kind == "image":
        detail = maps["detail"][1].get(msg.image.detail)
        return ImagePart(**_media_source_from_proto(msg.image.source), detail=detail)
    if kind == "audio":
        return AudioPart(**_media_source_from_proto(msg.audio.source))
    if kind == "video":
        return VideoPart(**_media_source_from_proto(msg.video.source))
    if kind == "document":
        return DocumentPart(**_media_source_from_proto(msg.document.source))
    if kind == "tool_call":
        return ToolCallPart(
            id=msg.tool_call.id,
            name=msg.tool_call.name,
            input=_struct_value(msg.tool_call, "input") or {},
        )
    if kind == "tool_result":
        return ToolResultPart(
            id=msg.tool_result.id,
            content=tuple(_part_from_proto(pb, p) for p in msg.tool_result.content),
            name=_wrapper_value(msg.tool_result, "name"),
            is_error=msg.tool_result.is_error,
        )
    if kind == "thinking":
        return ThinkingPart(text=msg.thinking.text, redacted=msg.thinking.redacted)
    if kind == "refusal":
        return RefusalPart(text=msg.refusal.text)
    if kind == "citation":
        return CitationPart(
            url=_wrapper_value(msg.citation, "url"),
            title=_wrapper_value(msg.citation, "title"),
            text=_wrapper_value(msg.citation, "text"),
        )
    raise ValueError("part kind missing")


def _message_from_proto(pb, msg):
    return Message(
        role=_maps(pb)["role"][1][msg.role],
        parts=tuple(_part_from_proto(pb, p) for p in msg.parts),
    )


def _system_from_proto(pb, msg) -> str | tuple[Part, ...] | None:
    kind = msg.WhichOneof("kind")
    if kind is None:
        return None
    if kind == "text":
        return msg.text
    return tuple(_part_from_proto(pb, p) for p in msg.parts.parts)


def _tool_from_proto(pb, msg):
    kind = msg.WhichOneof("kind")
    if kind == "function":
        return FunctionTool(
            name=msg.function.name,
            description=_wrapper_value(msg.function, "description"),
            parameters=_struct_value(msg.function, "parameters") or {},
        )
    if kind == "builtin":
        return BuiltinTool(
            name=msg.builtin.name,
            config=_struct_value(msg.builtin, "config"),
        )
    raise ValueError("tool kind missing")


def _tool_choice_from_proto(pb, msg):
    return ToolChoice(
        mode=_maps(pb)["mode"][1][msg.mode],
        allowed=tuple(msg.allowed),
        parallel=_wrapper_value(msg, "parallel"),
    )


def _reasoning_from_proto(pb, msg):
    return Reasoning(
        effort=_maps(pb)["effort"][1][msg.effort],
        thinking_budget=_wrapper_value(msg, "thinking_budget"),
        total_budget=_wrapper_value(msg, "total_budget"),
        summary=_wrapper_value(msg, "summary"),
    )


def _config_from_proto(pb, msg):
    return Config(
        max_tokens=_wrapper_value(msg, "max_tokens"),
        temperature=_wrapper_value(msg, "temperature"),
        top_p=_wrapper_value(msg, "top_p"),
        top_k=_wrapper_value(msg, "top_k"),
        stop=tuple(msg.stop),
        response_format=_struct_value(msg, "response_format"),
        tool_choice=_tool_choice_from_proto(pb, msg.tool_choice)
        if msg.HasField("tool_choice")
        else None,
        reasoning=_reasoning_from_proto(pb, msg.reasoning)
        if msg.HasField("reasoning")
        else None,
        extensions=_struct_value(msg, "extensions"),
    )


def _usage_from_proto(pb, msg):
    return Usage(
        input_tokens=msg.input_tokens,
        output_tokens=msg.output_tokens,
        total_tokens=msg.total_tokens,
        cache_read_tokens=_wrapper_value(msg, "cache_read_tokens"),
        cache_write_tokens=_wrapper_value(msg, "cache_write_tokens"),
        reasoning_tokens=_wrapper_value(msg, "reasoning_tokens"),
        input_audio_tokens=_wrapper_value(msg, "input_audio_tokens"),
        output_audio_tokens=_wrapper_value(msg, "output_audio_tokens"),
    )


def _error_from_proto(pb, msg):
    return ErrorDetail(
        code=_maps(pb)["error"][1][msg.code],
        message=msg.message,
        provider_code=_wrapper_value(msg, "provider_code"),
    )


def _delta_from_proto(pb, msg):
    kind = msg.WhichOneof("kind")
    if kind == "text":
        return TextDelta(text=msg.text.text, part_index=msg.text.part_index)
    if kind == "thinking":
        return ThinkingDelta(text=msg.thinking.text, part_index=msg.thinking.part_index)
    if kind == "audio":
        source = msg.audio.WhichOneof("source")
        kwargs = {
            "part_index": msg.audio.part_index,
            "media_type": _wrapper_value(msg.audio, "media_type"),
        }
        if source == "data":
            kwargs["data"] = msg.audio.data
        elif source == "url":
            kwargs["url"] = msg.audio.url
        elif source == "file_id":
            kwargs["file_id"] = msg.audio.file_id
        return AudioDelta(**kwargs)
    if kind == "image":
        source = msg.image.WhichOneof("source")
        kwargs = {
            "part_index": msg.image.part_index,
            "media_type": _wrapper_value(msg.image, "media_type"),
        }
        if source == "data":
            kwargs["data"] = msg.image.data
        elif source == "url":
            kwargs["url"] = msg.image.url
        elif source == "file_id":
            kwargs["file_id"] = msg.image.file_id
        return ImageDelta(**kwargs)
    if kind == "tool_call":
        return ToolCallDelta(
            input=msg.tool_call.input,
            part_index=msg.tool_call.part_index,
            id=_wrapper_value(msg.tool_call, "id"),
            name=_wrapper_value(msg.tool_call, "name"),
        )
    if kind == "citation":
        return CitationDelta(
            text=_wrapper_value(msg.citation, "text"),
            url=_wrapper_value(msg.citation, "url"),
            title=_wrapper_value(msg.citation, "title"),
            part_index=msg.citation.part_index,
        )
    raise ValueError("delta kind missing")


def _stream_event_from_proto(pb, msg):
    kind = msg.WhichOneof("event")
    if kind == "start":
        return StreamStartEvent(
            id=_wrapper_value(msg.start, "id"),
            model=_wrapper_value(msg.start, "model"),
        )
    if kind == "delta":
        return StreamDeltaEvent(delta=_delta_from_proto(pb, msg.delta.delta))
    if kind == "end":
        finish_reason = _maps(pb)["finish"][1].get(msg.end.finish_reason)
        return StreamEndEvent(
            finish_reason=finish_reason,
            usage=_usage_from_proto(pb, msg.end.usage) if msg.end.HasField("usage") else None,
            provider_data=_struct_value(msg.end, "provider_data"),
        )
    if kind == "error":
        return StreamErrorEvent(error=_error_from_proto(pb, msg.error.error))
    raise ValueError("event kind missing")


def _request_from_proto(pb, msg):
    return Request(
        model=msg.model,
        messages=tuple(_message_from_proto(pb, m) for m in msg.messages),
        system=_system_from_proto(pb, msg.system) if msg.HasField("system") else None,
        tools=tuple(_tool_from_proto(pb, t) for t in msg.tools),
        config=_config_from_proto(pb, msg.config) if msg.HasField("config") else Config(),
    )


def _response_from_proto(pb, msg):
    return Response(
        id=msg.id or None,
        model=msg.model,
        message=_message_from_proto(pb, msg.message),
        finish_reason=_maps(pb)["finish"][1][msg.finish_reason],
        usage=_usage_from_proto(pb, msg.usage),
        provider_data=_struct_value(msg, "provider_data"),
    )


def _audio_format_from_proto(pb, msg):
    return AudioFormat(
        encoding=_maps(pb)["encoding"][1][msg.encoding],
        sample_rate=msg.sample_rate,
        channels=msg.channels,
    )


def _live_config_from_proto(pb, msg):
    return LiveConfig(
        model=msg.model,
        system=_system_from_proto(pb, msg.system) if msg.HasField("system") else None,
        tools=tuple(_tool_from_proto(pb, t) for t in msg.tools),
        voice=_wrapper_value(msg, "voice"),
        input_format=_audio_format_from_proto(pb, msg.input_format)
        if msg.HasField("input_format")
        else None,
        output_format=_audio_format_from_proto(pb, msg.output_format)
        if msg.HasField("output_format")
        else None,
        extensions=_struct_value(msg, "extensions"),
    )


def _live_client_event_from_proto(pb, msg):
    kind = msg.WhichOneof("event")
    if kind == "audio":
        return LiveClientAudioEvent(data=_b64(msg.audio.data))
    if kind == "video":
        return LiveClientVideoEvent(data=_b64(msg.video.data))
    if kind == "text":
        return LiveClientTextEvent(text=msg.text.text)
    if kind == "tool_result":
        return LiveClientToolResultEvent(
            id=msg.tool_result.id,
            content=tuple(_part_from_proto(pb, p) for p in msg.tool_result.content),
        )
    if kind == "interrupt":
        return LiveClientInterruptEvent()
    if kind == "end_audio":
        return LiveClientEndAudioEvent()
    raise ValueError("live client event missing")


def _live_server_event_from_proto(pb, msg):
    kind = msg.WhichOneof("event")
    if kind == "audio":
        return LiveServerAudioEvent(data=_b64(msg.audio.data))
    if kind == "text":
        return LiveServerTextEvent(text=msg.text.text)
    if kind == "tool_call":
        return LiveServerToolCallEvent(
            id=msg.tool_call.id,
            name=msg.tool_call.name,
            input=_struct_value(msg.tool_call, "input") or {},
        )
    if kind == "tool_call_delta":
        return LiveServerToolCallDeltaEvent(
            input_delta=msg.tool_call_delta.input_delta,
            id=_wrapper_value(msg.tool_call_delta, "id"),
            name=_wrapper_value(msg.tool_call_delta, "name"),
        )
    if kind == "interrupted":
        return LiveServerInterruptedEvent()
    if kind == "turn_end":
        return LiveServerTurnEndEvent(usage=_usage_from_proto(pb, msg.turn_end.usage))
    if kind == "error":
        return LiveServerErrorEvent(error=_error_from_proto(pb, msg.error.error))
    raise ValueError("live server event missing")



# Auxiliary endpoint converters.


def _endpoint_request_to_proto(pb, value):
    out = pb.EndpointRequest()
    if isinstance(value, Request):
        out.request.CopyFrom(_request_to_proto(pb, value))
    elif isinstance(value, EmbeddingRequest):
        out.embedding_request.CopyFrom(_embedding_request_to_proto(pb, value))
    elif isinstance(value, FileUploadRequest):
        out.file_upload_request.CopyFrom(_file_upload_request_to_proto(pb, value))
    elif isinstance(value, BatchRequest):
        out.batch_request.CopyFrom(_batch_request_to_proto(pb, value))
    elif isinstance(value, ImageGenerationRequest):
        out.image_generation_request.CopyFrom(_image_generation_request_to_proto(pb, value))
    elif isinstance(value, AudioGenerationRequest):
        out.audio_generation_request.CopyFrom(_audio_generation_request_to_proto(pb, value))
    else:
        raise TypeError(type(value))
    return out


def _endpoint_request_from_proto(pb, msg):
    kind = msg.WhichOneof("kind")
    if kind == "request":
        return _request_from_proto(pb, msg.request)
    if kind == "embedding_request":
        return _embedding_request_from_proto(pb, msg.embedding_request)
    if kind == "file_upload_request":
        return _file_upload_request_from_proto(pb, msg.file_upload_request)
    if kind == "batch_request":
        return _batch_request_from_proto(pb, msg.batch_request)
    if kind == "image_generation_request":
        return _image_generation_request_from_proto(pb, msg.image_generation_request)
    if kind == "audio_generation_request":
        return _audio_generation_request_from_proto(pb, msg.audio_generation_request)
    raise ValueError("endpoint request kind missing")


def _endpoint_response_to_proto(pb, value):
    out = pb.EndpointResponse()
    if isinstance(value, Response):
        out.response.CopyFrom(_response_to_proto(pb, value))
    elif isinstance(value, EmbeddingResponse):
        out.embedding_response.CopyFrom(_embedding_response_to_proto(pb, value))
    elif isinstance(value, FileUploadResponse):
        out.file_upload_response.CopyFrom(_file_upload_response_to_proto(pb, value))
    elif isinstance(value, BatchResponse):
        out.batch_response.CopyFrom(_batch_response_to_proto(pb, value))
    elif isinstance(value, ImageGenerationResponse):
        out.image_generation_response.CopyFrom(_image_generation_response_to_proto(pb, value))
    elif isinstance(value, AudioGenerationResponse):
        out.audio_generation_response.CopyFrom(_audio_generation_response_to_proto(pb, value))
    else:
        raise TypeError(type(value))
    return out


def _endpoint_response_from_proto(pb, msg):
    kind = msg.WhichOneof("kind")
    if kind == "response":
        return _response_from_proto(pb, msg.response)
    if kind == "embedding_response":
        return _embedding_response_from_proto(pb, msg.embedding_response)
    if kind == "file_upload_response":
        return _file_upload_response_from_proto(pb, msg.file_upload_response)
    if kind == "batch_response":
        return _batch_response_from_proto(pb, msg.batch_response)
    if kind == "image_generation_response":
        return _image_generation_response_from_proto(pb, msg.image_generation_response)
    if kind == "audio_generation_response":
        return _audio_generation_response_from_proto(pb, msg.audio_generation_response)
    raise ValueError("endpoint response kind missing")


def _embedding_request_to_proto(pb, value: EmbeddingRequest):
    out = pb.EmbeddingRequest(model=value.model)
    out.inputs.extend(value.inputs)
    _copy_struct(out.extensions, value.extensions)
    return out


def _embedding_request_from_proto(pb, msg):
    return EmbeddingRequest(
        model=msg.model,
        inputs=tuple(msg.inputs),
        extensions=_struct_value(msg, "extensions"),
    )


def _embedding_response_to_proto(pb, value: EmbeddingResponse):
    out = pb.EmbeddingResponse(model=value.model)
    for vector in value.vectors:
        v = out.vectors.add()
        v.values.extend(vector)
    out.usage.CopyFrom(_usage_to_proto(pb, value.usage))
    _copy_struct(out.provider_data, value.provider_data)
    return out


def _embedding_response_from_proto(pb, msg):
    return EmbeddingResponse(
        model=msg.model,
        vectors=tuple(tuple(v.values) for v in msg.vectors),
        usage=_usage_from_proto(pb, msg.usage) if msg.HasField("usage") else Usage(),
        provider_data=_struct_value(msg, "provider_data"),
    )


def _file_upload_request_to_proto(pb, value: FileUploadRequest):
    out = pb.FileUploadRequest(
        filename=value.filename,
        bytes_data=value.bytes,
        media_type=value.media_type,
    )
    if value.model is not None:
        _set_wrapper(out.model, value.model)
    _copy_struct(out.extensions, value.extensions)
    return out


def _file_upload_request_from_proto(pb, msg):
    return FileUploadRequest(
        model=_wrapper_value(msg, "model"),
        filename=msg.filename,
        bytes_data=msg.bytes_data,
        media_type=msg.media_type,
        extensions=_struct_value(msg, "extensions"),
    )


def _file_upload_response_to_proto(pb, value: FileUploadResponse):
    out = pb.FileUploadResponse(id=value.id)
    _copy_struct(out.provider_data, value.provider_data)
    return out


def _file_upload_response_from_proto(pb, msg):
    return FileUploadResponse(id=msg.id, provider_data=_struct_value(msg, "provider_data"))


def _batch_request_to_proto(pb, value: BatchRequest):
    out = pb.BatchRequest()
    if value.model is not None:
        _set_wrapper(out.model, value.model)
    out.requests.extend(_request_to_proto(pb, r) for r in value.requests)
    _copy_struct(out.extensions, value.extensions)
    return out


def _batch_request_from_proto(pb, msg):
    return BatchRequest(
        model=_wrapper_value(msg, "model"),
        requests=tuple(_request_from_proto(pb, r) for r in msg.requests),
        extensions=_struct_value(msg, "extensions"),
    )


def _batch_response_to_proto(pb, value: BatchResponse):
    out = pb.BatchResponse(id=value.id, status=value.status)
    _copy_struct(out.provider_data, value.provider_data)
    return out


def _batch_response_from_proto(pb, msg):
    return BatchResponse(
        id=msg.id,
        status=msg.status,
        provider_data=_struct_value(msg, "provider_data"),
    )


def _image_generation_request_to_proto(pb, value: ImageGenerationRequest):
    out = pb.ImageGenerationRequest(model=value.model, prompt=value.prompt)
    if value.size is not None:
        _set_wrapper(out.size, value.size)
    _copy_struct(out.extensions, value.extensions)
    return out


def _image_generation_request_from_proto(pb, msg):
    return ImageGenerationRequest(
        model=msg.model,
        prompt=msg.prompt,
        size=_wrapper_value(msg, "size"),
        extensions=_struct_value(msg, "extensions"),
    )


def _image_generation_response_to_proto(pb, value: ImageGenerationResponse):
    out = pb.ImageGenerationResponse()
    for image in value.images:
        out.images.add().CopyFrom(_part_to_proto(pb, image).image)
    _copy_struct(out.provider_data, value.provider_data)
    if value.id is not None:
        _set_wrapper(out.id, value.id)
    if value.model is not None:
        _set_wrapper(out.model, value.model)
    out.usage.CopyFrom(_usage_to_proto(pb, value.usage))
    return out


def _image_generation_response_from_proto(pb, msg):
    return ImageGenerationResponse(
        images=tuple(
            ImagePart(
                **_media_source_from_proto(img.source),
                detail=_maps(pb)["detail"][1].get(img.detail),
            )
            for img in msg.images
        ),
        id=_wrapper_value(msg, "id"),
        model=_wrapper_value(msg, "model"),
        usage=_usage_from_proto(pb, msg.usage),
        provider_data=_struct_value(msg, "provider_data"),
    )


def _audio_generation_request_to_proto(pb, value: AudioGenerationRequest):
    out = pb.AudioGenerationRequest(model=value.model, prompt=value.prompt)
    if value.voice is not None:
        _set_wrapper(out.voice, value.voice)
    if value.format is not None:
        _set_wrapper(out.format, value.format)
    _copy_struct(out.extensions, value.extensions)
    return out


def _audio_generation_request_from_proto(pb, msg):
    return AudioGenerationRequest(
        model=msg.model,
        prompt=msg.prompt,
        voice=_wrapper_value(msg, "voice"),
        format=_wrapper_value(msg, "format"),
        extensions=_struct_value(msg, "extensions"),
    )


def _audio_generation_response_to_proto(pb, value: AudioGenerationResponse):
    out = pb.AudioGenerationResponse()
    out.audio.CopyFrom(_part_to_proto(pb, value.audio).audio)
    _copy_struct(out.provider_data, value.provider_data)
    if value.id is not None:
        _set_wrapper(out.id, value.id)
    if value.model is not None:
        _set_wrapper(out.model, value.model)
    out.usage.CopyFrom(_usage_to_proto(pb, value.usage))
    return out


def _audio_generation_response_from_proto(pb, msg):
    return AudioGenerationResponse(
        audio=AudioPart(**_media_source_from_proto(msg.audio.source)),
        id=_wrapper_value(msg, "id"),
        model=_wrapper_value(msg, "model"),
        usage=_usage_from_proto(pb, msg.usage),
        provider_data=_struct_value(msg, "provider_data"),
    )

# ─── Public conversion API ────────────────────────────────────────────


def part_to_proto(part: Part, pb: Any | None = None) -> Any:
    return _part_to_proto(_pb(pb), part)


def part_from_proto(message: Any, pb: Any | None = None) -> Part:
    return _part_from_proto(_pb(pb), message)


def message_to_proto(message: Message, pb: Any | None = None) -> Any:
    return _message_to_proto(_pb(pb), message)


def message_from_proto(message: Any, pb: Any | None = None) -> Message:
    return _message_from_proto(_pb(pb), message)


def delta_to_proto(delta: Any, pb: Any | None = None) -> Any:
    return _delta_to_proto(_pb(pb), delta)


def delta_from_proto(message: Any, pb: Any | None = None) -> Any:
    return _delta_from_proto(_pb(pb), message)


def stream_event_to_proto(event: StreamEvent, pb: Any | None = None) -> Any:
    return _stream_event_to_proto(_pb(pb), event)


def stream_event_from_proto(message: Any, pb: Any | None = None) -> StreamEvent:
    return _stream_event_from_proto(_pb(pb), message)


def tool_to_proto(tool: Tool, pb: Any | None = None) -> Any:
    return _tool_to_proto(_pb(pb), tool)


def tool_from_proto(message: Any, pb: Any | None = None) -> Tool:
    return _tool_from_proto(_pb(pb), message)


def tool_choice_to_proto(choice: ToolChoice, pb: Any | None = None) -> Any:
    return _tool_choice_to_proto(_pb(pb), choice)


def tool_choice_from_proto(message: Any, pb: Any | None = None) -> ToolChoice:
    return _tool_choice_from_proto(_pb(pb), message)


def reasoning_to_proto(reasoning: Reasoning, pb: Any | None = None) -> Any:
    return _reasoning_to_proto(_pb(pb), reasoning)


def reasoning_from_proto(message: Any, pb: Any | None = None) -> Reasoning:
    return _reasoning_from_proto(_pb(pb), message)


def config_to_proto(config: Config, pb: Any | None = None) -> Any:
    return _config_to_proto(_pb(pb), config)


def config_from_proto(message: Any, pb: Any | None = None) -> Config:
    return _config_from_proto(_pb(pb), message)


def usage_to_proto(usage: Usage, pb: Any | None = None) -> Any:
    return _usage_to_proto(_pb(pb), usage)


def usage_from_proto(message: Any, pb: Any | None = None) -> Usage:
    return _usage_from_proto(_pb(pb), message)


def error_detail_to_proto(error: ErrorDetail, pb: Any | None = None) -> Any:
    return _error_to_proto(_pb(pb), error)


def error_detail_from_proto(message: Any, pb: Any | None = None) -> ErrorDetail:
    return _error_from_proto(_pb(pb), message)


def audio_format_to_proto(audio_format: AudioFormat, pb: Any | None = None) -> Any:
    return _audio_format_to_proto(_pb(pb), audio_format)


def audio_format_from_proto(message: Any, pb: Any | None = None) -> AudioFormat:
    return _audio_format_from_proto(_pb(pb), message)


def request_to_proto(request: Request, pb: Any | None = None) -> Any:
    return _request_to_proto(_pb(pb), request)


def request_from_proto(message: Any, pb: Any | None = None) -> Request:
    return _request_from_proto(_pb(pb), message)


def response_to_proto(response: Response, pb: Any | None = None) -> Any:
    return _response_to_proto(_pb(pb), response)


def response_from_proto(message: Any, pb: Any | None = None) -> Response:
    return _response_from_proto(_pb(pb), message)


def endpoint_request_to_proto(value: Any, pb: Any | None = None) -> Any:
    return _endpoint_request_to_proto(_pb(pb), value)


def endpoint_request_from_proto(message: Any, pb: Any | None = None) -> Any:
    return _endpoint_request_from_proto(_pb(pb), message)


def endpoint_response_to_proto(value: Any, pb: Any | None = None) -> Any:
    return _endpoint_response_to_proto(_pb(pb), value)


def endpoint_response_from_proto(message: Any, pb: Any | None = None) -> Any:
    return _endpoint_response_from_proto(_pb(pb), message)


def live_config_to_proto(config: LiveConfig, pb: Any | None = None) -> Any:
    return _live_config_to_proto(_pb(pb), config)


def live_config_from_proto(message: Any, pb: Any | None = None) -> LiveConfig:
    return _live_config_from_proto(_pb(pb), message)


def live_client_event_to_proto(event: LiveClientEvent, pb: Any | None = None) -> Any:
    return _live_client_event_to_proto(_pb(pb), event)


def live_client_event_from_proto(message: Any, pb: Any | None = None) -> LiveClientEvent:
    return _live_client_event_from_proto(_pb(pb), message)


def live_server_event_to_proto(event: LiveServerEvent, pb: Any | None = None) -> Any:
    return _live_server_event_to_proto(_pb(pb), event)


def live_server_event_from_proto(message: Any, pb: Any | None = None) -> LiveServerEvent:
    return _live_server_event_from_proto(_pb(pb), message)


def tool_call_info_to_proto(info: ToolCallInfo, pb: Any | None = None) -> Any:
    pb = _pb(pb)
    out = pb.ToolCallInfo(id=info.id, name=info.name)
    _copy_struct(out.input, info.input)
    return out


def tool_call_info_from_proto(message: Any, pb: Any | None = None) -> ToolCallInfo:
    return ToolCallInfo(
        id=message.id,
        name=message.name,
        input=_struct_value(message, "input") or {},
    )


def to_proto(value: Any, pb: Any | None = None) -> Any:
    """Convert any supported lm15 Python object to its protobuf message."""
    pb = _pb(pb)
    if isinstance(value, (TextPart, ImagePart, AudioPart, VideoPart, DocumentPart,
                          ToolCallPart, ToolResultPart, ThinkingPart, RefusalPart,
                          CitationPart)):
        return _part_to_proto(pb, value)
    if isinstance(value, Message):
        return _message_to_proto(pb, value)
    if isinstance(value, (TextDelta, ThinkingDelta, AudioDelta, ImageDelta,
                          ToolCallDelta, CitationDelta)):
        return _delta_to_proto(pb, value)
    if isinstance(value, StreamEvent):
        return _stream_event_to_proto(pb, value)
    if isinstance(value, (FunctionTool, BuiltinTool)):
        return _tool_to_proto(pb, value)
    if isinstance(value, ToolChoice):
        return _tool_choice_to_proto(pb, value)
    if isinstance(value, Reasoning):
        return _reasoning_to_proto(pb, value)
    if isinstance(value, Config):
        return _config_to_proto(pb, value)
    if isinstance(value, Usage):
        return _usage_to_proto(pb, value)
    if isinstance(value, ErrorDetail):
        return _error_to_proto(pb, value)
    if isinstance(value, Request):
        return _request_to_proto(pb, value)
    if isinstance(value, Response):
        return _response_to_proto(pb, value)
    if isinstance(value, EmbeddingRequest):
        return _embedding_request_to_proto(pb, value)
    if isinstance(value, EmbeddingResponse):
        return _embedding_response_to_proto(pb, value)
    if isinstance(value, FileUploadRequest):
        return _file_upload_request_to_proto(pb, value)
    if isinstance(value, FileUploadResponse):
        return _file_upload_response_to_proto(pb, value)
    if isinstance(value, BatchRequest):
        return _batch_request_to_proto(pb, value)
    if isinstance(value, BatchResponse):
        return _batch_response_to_proto(pb, value)
    if isinstance(value, ImageGenerationRequest):
        return _image_generation_request_to_proto(pb, value)
    if isinstance(value, ImageGenerationResponse):
        return _image_generation_response_to_proto(pb, value)
    if isinstance(value, AudioGenerationRequest):
        return _audio_generation_request_to_proto(pb, value)
    if isinstance(value, AudioGenerationResponse):
        return _audio_generation_response_to_proto(pb, value)
    if isinstance(value, LiveConfig):
        return _live_config_to_proto(pb, value)
    if isinstance(value, LiveClientEvent):
        return _live_client_event_to_proto(pb, value)
    if isinstance(value, LiveServerEvent):
        return _live_server_event_to_proto(pb, value)
    if isinstance(value, ToolCallInfo):
        return tool_call_info_to_proto(value, pb)
    if isinstance(value, AudioFormat):
        return _audio_format_to_proto(pb, value)
    raise TypeError(f"unsupported lm15 object: {type(value)!r}")


def from_proto(message: Any, pb: Any | None = None) -> Any:
    """Convert any supported lm15 protobuf message to its Python object."""
    pb = _pb(pb)
    name = message.DESCRIPTOR.full_name
    if name == "lm15.v1.Part":
        return _part_from_proto(pb, message)
    if name == "lm15.v1.Message":
        return _message_from_proto(pb, message)
    if name == "lm15.v1.Delta":
        return _delta_from_proto(pb, message)
    if name == "lm15.v1.StreamEvent":
        return _stream_event_from_proto(pb, message)
    if name == "lm15.v1.Tool":
        return _tool_from_proto(pb, message)
    if name == "lm15.v1.ToolChoice":
        return _tool_choice_from_proto(pb, message)
    if name == "lm15.v1.Reasoning":
        return _reasoning_from_proto(pb, message)
    if name == "lm15.v1.Config":
        return _config_from_proto(pb, message)
    if name == "lm15.v1.Usage":
        return _usage_from_proto(pb, message)
    if name == "lm15.v1.ErrorDetail":
        return _error_from_proto(pb, message)
    if name == "lm15.v1.Request":
        return _request_from_proto(pb, message)
    if name == "lm15.v1.Response":
        return _response_from_proto(pb, message)
    if name == "lm15.v1.EndpointRequest":
        return _endpoint_request_from_proto(pb, message)
    if name == "lm15.v1.EndpointResponse":
        return _endpoint_response_from_proto(pb, message)
    if name == "lm15.v1.EmbeddingRequest":
        return _embedding_request_from_proto(pb, message)
    if name == "lm15.v1.EmbeddingResponse":
        return _embedding_response_from_proto(pb, message)
    if name == "lm15.v1.FileUploadRequest":
        return _file_upload_request_from_proto(pb, message)
    if name == "lm15.v1.FileUploadResponse":
        return _file_upload_response_from_proto(pb, message)
    if name == "lm15.v1.BatchRequest":
        return _batch_request_from_proto(pb, message)
    if name == "lm15.v1.BatchResponse":
        return _batch_response_from_proto(pb, message)
    if name == "lm15.v1.ImageGenerationRequest":
        return _image_generation_request_from_proto(pb, message)
    if name == "lm15.v1.ImageGenerationResponse":
        return _image_generation_response_from_proto(pb, message)
    if name == "lm15.v1.AudioGenerationRequest":
        return _audio_generation_request_from_proto(pb, message)
    if name == "lm15.v1.AudioGenerationResponse":
        return _audio_generation_response_from_proto(pb, message)
    if name == "lm15.v1.LiveConfig":
        return _live_config_from_proto(pb, message)
    if name == "lm15.v1.LiveClientEvent":
        return _live_client_event_from_proto(pb, message)
    if name == "lm15.v1.LiveServerEvent":
        return _live_server_event_from_proto(pb, message)
    if name == "lm15.v1.ToolCallInfo":
        return tool_call_info_from_proto(message, pb)
    if name == "lm15.v1.AudioFormat":
        return _audio_format_from_proto(pb, message)
    raise TypeError(f"unsupported lm15 proto message: {name}")


def to_proto_bytes(value: Any, pb: Any | None = None) -> bytes:
    """Serialize any supported lm15 Python object to protobuf binary."""
    return to_proto(value, pb).SerializeToString()


def from_proto_bytes(message_name: str, data: bytes, pb: Any | None = None) -> Any:
    """Parse protobuf binary by lm15 message name and convert it to Python.

    ``message_name`` can be either a short name like ``"Request"`` or a full
    protobuf name like ``"lm15.v1.Request"``.
    """
    pb = _pb(pb)
    short_name = message_name.rsplit(".", 1)[-1]
    cls = getattr(pb, short_name)
    msg = cls()
    msg.ParseFromString(data)
    return from_proto(msg, pb)
