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
    LiveClientEvent,
    LiveConfig,
    LiveServerEvent,
    Message,
    Part,
    Reasoning,
    RefusalPart,
    Request,
    Response,
    StreamEvent,
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
CuUFChxnb29nbGUvcHJvdG9idWYvc3RydWN0LnByb3RvEg9nb29nbGUucHJvdG9idWYimAEKBlN0cnVjdBI7CgZm
aWVsZHMYASADKAsyIy5nb29nbGUucHJvdG9idWYuU3RydWN0LkZpZWxkc0VudHJ5UgZmaWVsZHMaUQoLRmllbGRz
RW50cnkSEAoDa2V5GAEgASgJUgNrZXkSLAoFdmFsdWUYAiABKAsyFi5nb29nbGUucHJvdG9idWYuVmFsdWVSBXZh
bHVlOgI4ASKyAgoFVmFsdWUSOwoKbnVsbF92YWx1ZRgBIAEoDjIaLmdvb2dsZS5wcm90b2J1Zi5OdWxsVmFsdWVI
AFIJbnVsbFZhbHVlEiMKDG51bWJlcl92YWx1ZRgCIAEoAUgAUgtudW1iZXJWYWx1ZRIjCgxzdHJpbmdfdmFsdWUY
AyABKAlIAFILc3RyaW5nVmFsdWUSHwoKYm9vbF92YWx1ZRgEIAEoCEgAUglib29sVmFsdWUSPAoMc3RydWN0X3Zh
bHVlGAUgASgLMhcuZ29vZ2xlLnByb3RvYnVmLlN0cnVjdEgAUgtzdHJ1Y3RWYWx1ZRI7CgpsaXN0X3ZhbHVlGAYg
ASgLMhouZ29vZ2xlLnByb3RvYnVmLkxpc3RWYWx1ZUgAUglsaXN0VmFsdWVCBgoEa2luZCI7CglMaXN0VmFsdWUS
LgoGdmFsdWVzGAEgAygLMhYuZ29vZ2xlLnByb3RvYnVmLlZhbHVlUgZ2YWx1ZXMqGwoJTnVsbFZhbHVlEg4KCk5V
TExfVkFMVUUQAEKBAQoTY29tLmdvb2dsZS5wcm90b2J1ZkILU3RydWN0UHJvdG9QAVoxZ2l0aHViLmNvbS9nb2xh
bmcvcHJvdG9idWYvcHR5cGVzL3N0cnVjdDtzdHJ1Y3RwYvgBAaICA0dQQqoCHkdvb2dsZS5Qcm90b2J1Zi5XZWxs
S25vd25UeXBlc2IGcHJvdG8zCv4DCh5nb29nbGUvcHJvdG9idWYvd3JhcHBlcnMucHJvdG8SD2dvb2dsZS5wcm90
b2J1ZiIjCgtEb3VibGVWYWx1ZRIUCgV2YWx1ZRgBIAEoAVIFdmFsdWUiIgoKRmxvYXRWYWx1ZRIUCgV2YWx1ZRgB
IAEoAlIFdmFsdWUiIgoKSW50NjRWYWx1ZRIUCgV2YWx1ZRgBIAEoA1IFdmFsdWUiIwoLVUludDY0VmFsdWUSFAoF
dmFsdWUYASABKARSBXZhbHVlIiIKCkludDMyVmFsdWUSFAoFdmFsdWUYASABKAVSBXZhbHVlIiMKC1VJbnQzMlZh
bHVlEhQKBXZhbHVlGAEgASgNUgV2YWx1ZSIhCglCb29sVmFsdWUSFAoFdmFsdWUYASABKAhSBXZhbHVlIiMKC1N0
cmluZ1ZhbHVlEhQKBXZhbHVlGAEgASgJUgV2YWx1ZSIiCgpCeXRlc1ZhbHVlEhQKBXZhbHVlGAEgASgMUgV2YWx1
ZUJ8ChNjb20uZ29vZ2xlLnByb3RvYnVmQg1XcmFwcGVyc1Byb3RvUAFaKmdpdGh1Yi5jb20vZ29sYW5nL3Byb3Rv
YnVmL3B0eXBlcy93cmFwcGVyc/gBAaICA0dQQqoCHkdvb2dsZS5Qcm90b2J1Zi5XZWxsS25vd25UeXBlc2IGcHJv
dG8zCphaChJsbTE1L3YxL2xtMTUucHJvdG8SB2xtMTUudjEaHGdvb2dsZS9wcm90b2J1Zi9zdHJ1Y3QucHJvdG8a
Hmdvb2dsZS9wcm90b2J1Zi93cmFwcGVycy5wcm90byL+AwoEUGFydBInCgR0ZXh0GAEgASgLMhEubG0xNS52MS5U
ZXh0UGFydEgAUgR0ZXh0EioKBWltYWdlGAIgASgLMhIubG0xNS52MS5JbWFnZVBhcnRIAFIFaW1hZ2USKgoFYXVk
aW8YAyABKAsyEi5sbTE1LnYxLkF1ZGlvUGFydEgAUgVhdWRpbxIqCgV2aWRlbxgEIAEoCzISLmxtMTUudjEuVmlk
ZW9QYXJ0SABSBXZpZGVvEjMKCGRvY3VtZW50GAUgASgLMhUubG0xNS52MS5Eb2N1bWVudFBhcnRIAFIIZG9jdW1l
bnQSNAoJdG9vbF9jYWxsGAYgASgLMhUubG0xNS52MS5Ub29sQ2FsbFBhcnRIAFIIdG9vbENhbGwSOgoLdG9vbF9y
ZXN1bHQYByABKAsyFy5sbTE1LnYxLlRvb2xSZXN1bHRQYXJ0SABSCnRvb2xSZXN1bHQSMwoIdGhpbmtpbmcYCCAB
KAsyFS5sbTE1LnYxLlRoaW5raW5nUGFydEgAUgh0aGlua2luZxIwCgdyZWZ1c2FsGAkgASgLMhQubG0xNS52MS5S
ZWZ1c2FsUGFydEgAUgdyZWZ1c2FsEjMKCGNpdGF0aW9uGAogASgLMhUubG0xNS52MS5DaXRhdGlvblBhcnRIAFII
Y2l0YXRpb25CBgoEa2luZCIvCghQYXJ0TGlzdBIjCgVwYXJ0cxgBIAMoCzINLmxtMTUudjEuUGFydFIFcGFydHMi
HgoIVGV4dFBhcnQSEgoEdGV4dBgBIAEoCVIEdGV4dCJ7CgtNZWRpYVNvdXJjZRIdCgptZWRpYV90eXBlGAEgASgJ
UgltZWRpYVR5cGUSFAoEZGF0YRgCIAEoDEgAUgRkYXRhEhIKA3VybBgDIAEoCUgAUgN1cmwSGQoHZmlsZV9pZBgE
IAEoCUgAUgZmaWxlSWRCCAoGc291cmNlImcKCUltYWdlUGFydBIsCgZzb3VyY2UYASABKAsyFC5sbTE1LnYxLk1l
ZGlhU291cmNlUgZzb3VyY2USLAoGZGV0YWlsGAIgASgOMhQubG0xNS52MS5JbWFnZURldGFpbFIGZGV0YWlsIjkK
CUF1ZGlvUGFydBIsCgZzb3VyY2UYASABKAsyFC5sbTE1LnYxLk1lZGlhU291cmNlUgZzb3VyY2UiOQoJVmlkZW9Q
YXJ0EiwKBnNvdXJjZRgBIAEoCzIULmxtMTUudjEuTWVkaWFTb3VyY2VSBnNvdXJjZSI8CgxEb2N1bWVudFBhcnQS
LAoGc291cmNlGAEgASgLMhQubG0xNS52MS5NZWRpYVNvdXJjZVIGc291cmNlImEKDFRvb2xDYWxsUGFydBIOCgJp
ZBgBIAEoCVICaWQSEgoEbmFtZRgCIAEoCVIEbmFtZRItCgVpbnB1dBgDIAEoCzIXLmdvb2dsZS5wcm90b2J1Zi5T
dHJ1Y3RSBWlucHV0IpYBCg5Ub29sUmVzdWx0UGFydBIOCgJpZBgBIAEoCVICaWQSJwoHY29udGVudBgCIAMoCzIN
LmxtMTUudjEuUGFydFIHY29udGVudBIwCgRuYW1lGAMgASgLMhwuZ29vZ2xlLnByb3RvYnVmLlN0cmluZ1ZhbHVl
UgRuYW1lEhkKCGlzX2Vycm9yGAQgASgIUgdpc0Vycm9yIj4KDFRoaW5raW5nUGFydBISCgR0ZXh0GAEgASgJUgR0
ZXh0EhoKCHJlZGFjdGVkGAIgASgIUghyZWRhY3RlZCIhCgtSZWZ1c2FsUGFydBISCgR0ZXh0GAEgASgJUgR0ZXh0
IqQBCgxDaXRhdGlvblBhcnQSLgoDdXJsGAEgASgLMhwuZ29vZ2xlLnByb3RvYnVmLlN0cmluZ1ZhbHVlUgN1cmwS
MgoFdGl0bGUYAiABKAsyHC5nb29nbGUucHJvdG9idWYuU3RyaW5nVmFsdWVSBXRpdGxlEjAKBHRleHQYAyABKAsy
HC5nb29nbGUucHJvdG9idWYuU3RyaW5nVmFsdWVSBHRleHQiUQoHTWVzc2FnZRIhCgRyb2xlGAEgASgOMg0ubG0x
NS52MS5Sb2xlUgRyb2xlEiMKBXBhcnRzGAIgAygLMg0ubG0xNS52MS5QYXJ0UgVwYXJ0cyJYCg1TeXN0ZW1Db250
ZW50EhQKBHRleHQYASABKAlIAFIEdGV4dBIpCgVwYXJ0cxgCIAEoCzIRLmxtMTUudjEuUGFydExpc3RIAFIFcGFy
dHNCBgoEa2luZCK2AgoFRGVsdGESKAoEdGV4dBgBIAEoCzISLmxtMTUudjEuVGV4dERlbHRhSABSBHRleHQSNAoI
dGhpbmtpbmcYAiABKAsyFi5sbTE1LnYxLlRoaW5raW5nRGVsdGFIAFIIdGhpbmtpbmcSKwoFYXVkaW8YAyABKAsy
Ey5sbTE1LnYxLkF1ZGlvRGVsdGFIAFIFYXVkaW8SKwoFaW1hZ2UYBCABKAsyEy5sbTE1LnYxLkltYWdlRGVsdGFI
AFIFaW1hZ2USNQoJdG9vbF9jYWxsGAUgASgLMhYubG0xNS52MS5Ub29sQ2FsbERlbHRhSABSCHRvb2xDYWxsEjQK
CGNpdGF0aW9uGAYgASgLMhYubG0xNS52MS5DaXRhdGlvbkRlbHRhSABSCGNpdGF0aW9uQgYKBGtpbmQiPgoJVGV4
dERlbHRhEhIKBHRleHQYASABKAlSBHRleHQSHQoKcGFydF9pbmRleBgCIAEoBVIJcGFydEluZGV4IkIKDVRoaW5r
aW5nRGVsdGESEgoEdGV4dBgBIAEoCVIEdGV4dBIdCgpwYXJ0X2luZGV4GAIgASgFUglwYXJ0SW5kZXgifAoKQXVk
aW9EZWx0YRISCgRkYXRhGAEgASgMUgRkYXRhEh0KCnBhcnRfaW5kZXgYAiABKAVSCXBhcnRJbmRleBI7CgptZWRp
YV90eXBlGAMgASgLMhwuZ29vZ2xlLnByb3RvYnVmLlN0cmluZ1ZhbHVlUgltZWRpYVR5cGUitwEKCkltYWdlRGVs
dGESHQoKcGFydF9pbmRleBgBIAEoBVIJcGFydEluZGV4EjsKCm1lZGlhX3R5cGUYAiABKAsyHC5nb29nbGUucHJv
dG9idWYuU3RyaW5nVmFsdWVSCW1lZGlhVHlwZRIUCgRkYXRhGAMgASgMSABSBGRhdGESEgoDdXJsGAQgASgJSABS
A3VybBIZCgdmaWxlX2lkGAUgASgJSABSBmZpbGVJZEIICgZzb3VyY2UipAEKDVRvb2xDYWxsRGVsdGESFAoFaW5w
dXQYASABKAlSBWlucHV0Eh0KCnBhcnRfaW5kZXgYAiABKAVSCXBhcnRJbmRleBIsCgJpZBgDIAEoCzIcLmdvb2ds
ZS5wcm90b2J1Zi5TdHJpbmdWYWx1ZVICaWQSMAoEbmFtZRgEIAEoCzIcLmdvb2dsZS5wcm90b2J1Zi5TdHJpbmdW
YWx1ZVIEbmFtZSLEAQoNQ2l0YXRpb25EZWx0YRIwCgR0ZXh0GAEgASgLMhwuZ29vZ2xlLnByb3RvYnVmLlN0cmlu
Z1ZhbHVlUgR0ZXh0Ei4KA3VybBgCIAEoCzIcLmdvb2dsZS5wcm90b2J1Zi5TdHJpbmdWYWx1ZVIDdXJsEjIKBXRp
dGxlGAMgASgLMhwuZ29vZ2xlLnByb3RvYnVmLlN0cmluZ1ZhbHVlUgV0aXRsZRIdCgpwYXJ0X2luZGV4GAQgASgF
UglwYXJ0SW5kZXgixAEKC1N0cmVhbUV2ZW50EisKBXN0YXJ0GAEgASgLMhMubG0xNS52MS5TdGFydEV2ZW50SABS
BXN0YXJ0EisKBWRlbHRhGAIgASgLMhMubG0xNS52MS5EZWx0YUV2ZW50SABSBWRlbHRhEiUKA2VuZBgDIAEoCzIR
LmxtMTUudjEuRW5kRXZlbnRIAFIDZW5kEisKBWVycm9yGAQgASgLMhMubG0xNS52MS5FcnJvckV2ZW50SABSBWVy
cm9yQgcKBWV2ZW50Im4KClN0YXJ0RXZlbnQSLAoCaWQYASABKAsyHC5nb29nbGUucHJvdG9idWYuU3RyaW5nVmFs
dWVSAmlkEjIKBW1vZGVsGAIgASgLMhwuZ29vZ2xlLnByb3RvYnVmLlN0cmluZ1ZhbHVlUgVtb2RlbCIyCgpEZWx0
YUV2ZW50EiQKBWRlbHRhGAEgASgLMg4ubG0xNS52MS5EZWx0YVIFZGVsdGEiqgEKCEVuZEV2ZW50EjoKDWZpbmlz
aF9yZWFzb24YASABKA4yFS5sbTE1LnYxLkZpbmlzaFJlYXNvblIMZmluaXNoUmVhc29uEiQKBXVzYWdlGAIgASgL
Mg4ubG0xNS52MS5Vc2FnZVIFdXNhZ2USPAoNcHJvdmlkZXJfZGF0YRgDIAEoCzIXLmdvb2dsZS5wcm90b2J1Zi5T
dHJ1Y3RSDHByb3ZpZGVyRGF0YSI4CgpFcnJvckV2ZW50EioKBWVycm9yGAEgASgLMhQubG0xNS52MS5FcnJvckRl
dGFpbFIFZXJyb3IikgEKC0Vycm9yRGV0YWlsEiYKBGNvZGUYASABKA4yEi5sbTE1LnYxLkVycm9yQ29kZVIEY29k
ZRIYCgdtZXNzYWdlGAIgASgJUgdtZXNzYWdlEkEKDXByb3ZpZGVyX2NvZGUYAyABKAsyHC5nb29nbGUucHJvdG9i
dWYuU3RyaW5nVmFsdWVSDHByb3ZpZGVyQ29kZSJ1CgRUb29sEjMKCGZ1bmN0aW9uGAEgASgLMhUubG0xNS52MS5G
dW5jdGlvblRvb2xIAFIIZnVuY3Rpb24SMAoHYnVpbHRpbhgCIAEoCzIULmxtMTUudjEuQnVpbHRpblRvb2xIAFIH
YnVpbHRpbkIGCgRraW5kIpsBCgxGdW5jdGlvblRvb2wSEgoEbmFtZRgBIAEoCVIEbmFtZRI+CgtkZXNjcmlwdGlv
bhgCIAEoCzIcLmdvb2dsZS5wcm90b2J1Zi5TdHJpbmdWYWx1ZVILZGVzY3JpcHRpb24SNwoKcGFyYW1ldGVycxgD
IAEoCzIXLmdvb2dsZS5wcm90b2J1Zi5TdHJ1Y3RSCnBhcmFtZXRlcnMiUgoLQnVpbHRpblRvb2wSEgoEbmFtZRgB
IAEoCVIEbmFtZRIvCgZjb25maWcYAiABKAsyFy5nb29nbGUucHJvdG9idWYuU3RydWN0UgZjb25maWciYQoMVG9v
bENhbGxJbmZvEg4KAmlkGAEgASgJUgJpZBISCgRuYW1lGAIgASgJUgRuYW1lEi0KBWlucHV0GAMgASgLMhcuZ29v
Z2xlLnByb3RvYnVmLlN0cnVjdFIFaW5wdXQiwwEKCVJlYXNvbmluZxIwCgZlZmZvcnQYASABKA4yGC5sbTE1LnYx
LlJlYXNvbmluZ0VmZm9ydFIGZWZmb3J0EkQKD3RoaW5raW5nX2J1ZGdldBgCIAEoCzIbLmdvb2dsZS5wcm90b2J1
Zi5JbnQzMlZhbHVlUg50aGlua2luZ0J1ZGdldBI+Cgx0b3RhbF9idWRnZXQYAyABKAsyGy5nb29nbGUucHJvdG9i
dWYuSW50MzJWYWx1ZVILdG90YWxCdWRnZXQiiwEKClRvb2xDaG9pY2USKwoEbW9kZRgBIAEoDjIXLmxtMTUudjEu
VG9vbENob2ljZU1vZGVSBG1vZGUSGAoHYWxsb3dlZBgCIAMoCVIHYWxsb3dlZBI2CghwYXJhbGxlbBgDIAEoCzIa
Lmdvb2dsZS5wcm90b2J1Zi5Cb29sVmFsdWVSCHBhcmFsbGVsIuADCgZDb25maWcSOgoKbWF4X3Rva2VucxgBIAEo
CzIbLmdvb2dsZS5wcm90b2J1Zi5JbnQzMlZhbHVlUgltYXhUb2tlbnMSPgoLdGVtcGVyYXR1cmUYAiABKAsyHC5n
b29nbGUucHJvdG9idWYuRG91YmxlVmFsdWVSC3RlbXBlcmF0dXJlEjEKBXRvcF9wGAMgASgLMhwuZ29vZ2xlLnBy
b3RvYnVmLkRvdWJsZVZhbHVlUgR0b3BQEjAKBXRvcF9rGAQgASgLMhsuZ29vZ2xlLnByb3RvYnVmLkludDMyVmFs
dWVSBHRvcEsSEgoEc3RvcBgFIAMoCVIEc3RvcBJACg9yZXNwb25zZV9mb3JtYXQYBiABKAsyFy5nb29nbGUucHJv
dG9idWYuU3RydWN0Ug5yZXNwb25zZUZvcm1hdBI0Cgt0b29sX2Nob2ljZRgHIAEoCzITLmxtMTUudjEuVG9vbENo
b2ljZVIKdG9vbENob2ljZRIwCglyZWFzb25pbmcYCCABKAsyEi5sbTE1LnYxLlJlYXNvbmluZ1IJcmVhc29uaW5n
EjcKCmV4dGVuc2lvbnMYCSABKAsyFy5nb29nbGUucHJvdG9idWYuU3RydWN0UgpleHRlbnNpb25zIv0BCgdSZXF1
ZXN0EhQKBW1vZGVsGAEgASgJUgVtb2RlbBIsCghtZXNzYWdlcxgCIAMoCzIQLmxtMTUudjEuTWVzc2FnZVIIbWVz
c2FnZXMSLgoGc3lzdGVtGAMgASgLMhYubG0xNS52MS5TeXN0ZW1Db250ZW50UgZzeXN0ZW0SIwoFdG9vbHMYBCAD
KAsyDS5sbTE1LnYxLlRvb2xSBXRvb2xzEicKBmNvbmZpZxgFIAEoCzIPLmxtMTUudjEuQ29uZmlnUgZjb25maWcS
MAoFY2FjaGUYBiABKAsyGi5nb29nbGUucHJvdG9idWYuQm9vbFZhbHVlUgVjYWNoZSLmAwoFVXNhZ2USIQoMaW5w
dXRfdG9rZW5zGAEgASgDUgtpbnB1dFRva2VucxIjCg1vdXRwdXRfdG9rZW5zGAIgASgDUgxvdXRwdXRUb2tlbnMS
IQoMdG90YWxfdG9rZW5zGAMgASgDUgt0b3RhbFRva2VucxJHChFjYWNoZV9yZWFkX3Rva2VucxgEIAEoCzIbLmdv
b2dsZS5wcm90b2J1Zi5JbnQ2NFZhbHVlUg9jYWNoZVJlYWRUb2tlbnMSSQoSY2FjaGVfd3JpdGVfdG9rZW5zGAUg
ASgLMhsuZ29vZ2xlLnByb3RvYnVmLkludDY0VmFsdWVSEGNhY2hlV3JpdGVUb2tlbnMSRgoQcmVhc29uaW5nX3Rv
a2VucxgGIAEoCzIbLmdvb2dsZS5wcm90b2J1Zi5JbnQ2NFZhbHVlUg9yZWFzb25pbmdUb2tlbnMSSQoSaW5wdXRf
YXVkaW9fdG9rZW5zGAcgASgLMhsuZ29vZ2xlLnByb3RvYnVmLkludDY0VmFsdWVSEGlucHV0QXVkaW9Ub2tlbnMS
SwoTb3V0cHV0X2F1ZGlvX3Rva2VucxgIIAEoCzIbLmdvb2dsZS5wcm90b2J1Zi5JbnQ2NFZhbHVlUhFvdXRwdXRB
dWRpb1Rva2VucyL8AQoIUmVzcG9uc2USDgoCaWQYASABKAlSAmlkEhQKBW1vZGVsGAIgASgJUgVtb2RlbBIqCgdt
ZXNzYWdlGAMgASgLMhAubG0xNS52MS5NZXNzYWdlUgdtZXNzYWdlEjoKDWZpbmlzaF9yZWFzb24YBCABKA4yFS5s
bTE1LnYxLkZpbmlzaFJlYXNvblIMZmluaXNoUmVhc29uEiQKBXVzYWdlGAUgASgLMg4ubG0xNS52MS5Vc2FnZVIF
dXNhZ2USPAoNcHJvdmlkZXJfZGF0YRgGIAEoCzIXLmdvb2dsZS5wcm90b2J1Zi5TdHJ1Y3RSDHByb3ZpZGVyRGF0
YSJ5ChBFbWJlZGRpbmdSZXF1ZXN0EhQKBW1vZGVsGAEgASgJUgVtb2RlbBIWCgZpbnB1dHMYAiADKAlSBmlucHV0
cxI3CgpleHRlbnNpb25zGAMgASgLMhcuZ29vZ2xlLnByb3RvYnVmLlN0cnVjdFIKZXh0ZW5zaW9ucyK4AQoRRW1i
ZWRkaW5nUmVzcG9uc2USFAoFbW9kZWwYASABKAlSBW1vZGVsEikKB3ZlY3RvcnMYAiADKAsyDy5sbTE1LnYxLlZl
Y3RvclIHdmVjdG9ycxIkCgV1c2FnZRgDIAEoCzIOLmxtMTUudjEuVXNhZ2VSBXVzYWdlEjwKDXByb3ZpZGVyX2Rh
dGEYBCABKAsyFy5nb29nbGUucHJvdG9idWYuU3RydWN0Ugxwcm92aWRlckRhdGEiIAoGVmVjdG9yEhYKBnZhbHVl
cxgBIAMoAVIGdmFsdWVzItoBChFGaWxlVXBsb2FkUmVxdWVzdBIyCgVtb2RlbBgBIAEoCzIcLmdvb2dsZS5wcm90
b2J1Zi5TdHJpbmdWYWx1ZVIFbW9kZWwSGgoIZmlsZW5hbWUYAiABKAlSCGZpbGVuYW1lEh0KCmJ5dGVzX2RhdGEY
AyABKAxSCWJ5dGVzRGF0YRIdCgptZWRpYV90eXBlGAQgASgJUgltZWRpYVR5cGUSNwoKZXh0ZW5zaW9ucxgFIAEo
CzIXLmdvb2dsZS5wcm90b2J1Zi5TdHJ1Y3RSCmV4dGVuc2lvbnMiYgoSRmlsZVVwbG9hZFJlc3BvbnNlEg4KAmlk
GAEgASgJUgJpZBI8Cg1wcm92aWRlcl9kYXRhGAIgASgLMhcuZ29vZ2xlLnByb3RvYnVmLlN0cnVjdFIMcHJvdmlk
ZXJEYXRhIosBCgxCYXRjaFJlcXVlc3QSFAoFbW9kZWwYASABKAlSBW1vZGVsEiwKCHJlcXVlc3RzGAIgAygLMhAu
bG0xNS52MS5SZXF1ZXN0UghyZXF1ZXN0cxI3CgpleHRlbnNpb25zGAMgASgLMhcuZ29vZ2xlLnByb3RvYnVmLlN0
cnVjdFIKZXh0ZW5zaW9ucyJ1Cg1CYXRjaFJlc3BvbnNlEg4KAmlkGAEgASgJUgJpZBIWCgZzdGF0dXMYAiABKAlS
BnN0YXR1cxI8Cg1wcm92aWRlcl9kYXRhGAMgASgLMhcuZ29vZ2xlLnByb3RvYnVmLlN0cnVjdFIMcHJvdmlkZXJE
YXRhIrEBChZJbWFnZUdlbmVyYXRpb25SZXF1ZXN0EhQKBW1vZGVsGAEgASgJUgVtb2RlbBIWCgZwcm9tcHQYAiAB
KAlSBnByb21wdBIwCgRzaXplGAMgASgLMhwuZ29vZ2xlLnByb3RvYnVmLlN0cmluZ1ZhbHVlUgRzaXplEjcKCmV4
dGVuc2lvbnMYBCABKAsyFy5nb29nbGUucHJvdG9idWYuU3RydWN0UgpleHRlbnNpb25zIoMBChdJbWFnZUdlbmVy
YXRpb25SZXNwb25zZRIqCgZpbWFnZXMYASADKAsyEi5sbTE1LnYxLkltYWdlUGFydFIGaW1hZ2VzEjwKDXByb3Zp
ZGVyX2RhdGEYAiABKAsyFy5nb29nbGUucHJvdG9idWYuU3RydWN0Ugxwcm92aWRlckRhdGEi6QEKFkF1ZGlvR2Vu
ZXJhdGlvblJlcXVlc3QSFAoFbW9kZWwYASABKAlSBW1vZGVsEhYKBnByb21wdBgCIAEoCVIGcHJvbXB0EjIKBXZv
aWNlGAMgASgLMhwuZ29vZ2xlLnByb3RvYnVmLlN0cmluZ1ZhbHVlUgV2b2ljZRI0CgZmb3JtYXQYBCABKAsyHC5n
b29nbGUucHJvdG9idWYuU3RyaW5nVmFsdWVSBmZvcm1hdBI3CgpleHRlbnNpb25zGAUgASgLMhcuZ29vZ2xlLnBy
b3RvYnVmLlN0cnVjdFIKZXh0ZW5zaW9ucyKBAQoXQXVkaW9HZW5lcmF0aW9uUmVzcG9uc2USKAoFYXVkaW8YASAB
KAsyEi5sbTE1LnYxLkF1ZGlvUGFydFIFYXVkaW8SPAoNcHJvdmlkZXJfZGF0YRgCIAEoCzIXLmdvb2dsZS5wcm90
b2J1Zi5TdHJ1Y3RSDHByb3ZpZGVyRGF0YSLXAwoPRW5kcG9pbnRSZXF1ZXN0EiwKB3JlcXVlc3QYASABKAsyEC5s
bTE1LnYxLlJlcXVlc3RIAFIHcmVxdWVzdBJIChFlbWJlZGRpbmdfcmVxdWVzdBgCIAEoCzIZLmxtMTUudjEuRW1i
ZWRkaW5nUmVxdWVzdEgAUhBlbWJlZGRpbmdSZXF1ZXN0EkwKE2ZpbGVfdXBsb2FkX3JlcXVlc3QYAyABKAsyGi5s
bTE1LnYxLkZpbGVVcGxvYWRSZXF1ZXN0SABSEWZpbGVVcGxvYWRSZXF1ZXN0EjwKDWJhdGNoX3JlcXVlc3QYBCAB
KAsyFS5sbTE1LnYxLkJhdGNoUmVxdWVzdEgAUgxiYXRjaFJlcXVlc3QSWwoYaW1hZ2VfZ2VuZXJhdGlvbl9yZXF1
ZXN0GAUgASgLMh8ubG0xNS52MS5JbWFnZUdlbmVyYXRpb25SZXF1ZXN0SABSFmltYWdlR2VuZXJhdGlvblJlcXVl
c3QSWwoYYXVkaW9fZ2VuZXJhdGlvbl9yZXF1ZXN0GAYgASgLMh8ubG0xNS52MS5BdWRpb0dlbmVyYXRpb25SZXF1
ZXN0SABSFmF1ZGlvR2VuZXJhdGlvblJlcXVlc3RCBgoEa2luZCLqAwoQRW5kcG9pbnRSZXNwb25zZRIvCghyZXNw
b25zZRgBIAEoCzIRLmxtMTUudjEuUmVzcG9uc2VIAFIIcmVzcG9uc2USSwoSZW1iZWRkaW5nX3Jlc3BvbnNlGAIg
ASgLMhoubG0xNS52MS5FbWJlZGRpbmdSZXNwb25zZUgAUhFlbWJlZGRpbmdSZXNwb25zZRJPChRmaWxlX3VwbG9h
ZF9yZXNwb25zZRgDIAEoCzIbLmxtMTUudjEuRmlsZVVwbG9hZFJlc3BvbnNlSABSEmZpbGVVcGxvYWRSZXNwb25z
ZRI/Cg5iYXRjaF9yZXNwb25zZRgEIAEoCzIWLmxtMTUudjEuQmF0Y2hSZXNwb25zZUgAUg1iYXRjaFJlc3BvbnNl
El4KGWltYWdlX2dlbmVyYXRpb25fcmVzcG9uc2UYBSABKAsyIC5sbTE1LnYxLkltYWdlR2VuZXJhdGlvblJlc3Bv
bnNlSABSF2ltYWdlR2VuZXJhdGlvblJlc3BvbnNlEl4KGWF1ZGlvX2dlbmVyYXRpb25fcmVzcG9uc2UYBiABKAsy
IC5sbTE1LnYxLkF1ZGlvR2VuZXJhdGlvblJlc3BvbnNlSABSF2F1ZGlvR2VuZXJhdGlvblJlc3BvbnNlQgYKBGtp
bmQifgoLQXVkaW9Gb3JtYXQSMgoIZW5jb2RpbmcYASABKA4yFi5sbTE1LnYxLkF1ZGlvRW5jb2RpbmdSCGVuY29k
aW5nEh8KC3NhbXBsZV9yYXRlGAIgASgFUgpzYW1wbGVSYXRlEhoKCGNoYW5uZWxzGAMgASgFUghjaGFubmVscyLY
AgoKTGl2ZUNvbmZpZxIUCgVtb2RlbBgBIAEoCVIFbW9kZWwSLgoGc3lzdGVtGAIgASgLMhYubG0xNS52MS5TeXN0
ZW1Db250ZW50UgZzeXN0ZW0SIwoFdG9vbHMYAyADKAsyDS5sbTE1LnYxLlRvb2xSBXRvb2xzEjIKBXZvaWNlGAQg
ASgLMhwuZ29vZ2xlLnByb3RvYnVmLlN0cmluZ1ZhbHVlUgV2b2ljZRI3CgxpbnB1dF9mb3JtYXQYBSABKAsyFC5s
bTE1LnYxLkF1ZGlvRm9ybWF0UgtpbnB1dEZvcm1hdBI5Cg1vdXRwdXRfZm9ybWF0GAYgASgLMhQubG0xNS52MS5B
dWRpb0Zvcm1hdFIMb3V0cHV0Rm9ybWF0EjcKCmV4dGVuc2lvbnMYByABKAsyFy5nb29nbGUucHJvdG9idWYuU3Ry
dWN0UgpleHRlbnNpb25zIukCCg9MaXZlQ2xpZW50RXZlbnQSMAoFYXVkaW8YASABKAsyGC5sbTE1LnYxLkxpdmVD
bGllbnRBdWRpb0gAUgVhdWRpbxIwCgV2aWRlbxgCIAEoCzIYLmxtMTUudjEuTGl2ZUNsaWVudFZpZGVvSABSBXZp
ZGVvEi0KBHRleHQYAyABKAsyFy5sbTE1LnYxLkxpdmVDbGllbnRUZXh0SABSBHRleHQSQAoLdG9vbF9yZXN1bHQY
BCABKAsyHS5sbTE1LnYxLkxpdmVDbGllbnRUb29sUmVzdWx0SABSCnRvb2xSZXN1bHQSPAoJaW50ZXJydXB0GAUg
ASgLMhwubG0xNS52MS5MaXZlQ2xpZW50SW50ZXJydXB0SABSCWludGVycnVwdBI6CgllbmRfYXVkaW8YBiABKAsy
Gy5sbTE1LnYxLkxpdmVDbGllbnRFbmRBdWRpb0gAUghlbmRBdWRpb0IHCgVldmVudCIlCg9MaXZlQ2xpZW50QXVk
aW8SEgoEZGF0YRgBIAEoDFIEZGF0YSIlCg9MaXZlQ2xpZW50VmlkZW8SEgoEZGF0YRgBIAEoDFIEZGF0YSIkCg5M
aXZlQ2xpZW50VGV4dBISCgR0ZXh0GAEgASgJUgR0ZXh0Ik8KFExpdmVDbGllbnRUb29sUmVzdWx0Eg4KAmlkGAEg
ASgJUgJpZBInCgdjb250ZW50GAIgAygLMg0ubG0xNS52MS5QYXJ0Ugdjb250ZW50IhUKE0xpdmVDbGllbnRJbnRl
cnJ1cHQiFAoSTGl2ZUNsaWVudEVuZEF1ZGlvIuYCCg9MaXZlU2VydmVyRXZlbnQSMAoFYXVkaW8YASABKAsyGC5s
bTE1LnYxLkxpdmVTZXJ2ZXJBdWRpb0gAUgVhdWRpbxItCgR0ZXh0GAIgASgLMhcubG0xNS52MS5MaXZlU2VydmVy
VGV4dEgAUgR0ZXh0EjoKCXRvb2xfY2FsbBgDIAEoCzIbLmxtMTUudjEuTGl2ZVNlcnZlclRvb2xDYWxsSABSCHRv
b2xDYWxsEkIKC2ludGVycnVwdGVkGAQgASgLMh4ubG0xNS52MS5MaXZlU2VydmVySW50ZXJydXB0ZWRIAFILaW50
ZXJydXB0ZWQSNwoIdHVybl9lbmQYBSABKAsyGi5sbTE1LnYxLkxpdmVTZXJ2ZXJUdXJuRW5kSABSB3R1cm5FbmQS
MAoFZXJyb3IYBiABKAsyGC5sbTE1LnYxLkxpdmVTZXJ2ZXJFcnJvckgAUgVlcnJvckIHCgVldmVudCIlCg9MaXZl
U2VydmVyQXVkaW8SEgoEZGF0YRgBIAEoDFIEZGF0YSIkCg5MaXZlU2VydmVyVGV4dBISCgR0ZXh0GAEgASgJUgR0
ZXh0ImcKEkxpdmVTZXJ2ZXJUb29sQ2FsbBIOCgJpZBgBIAEoCVICaWQSEgoEbmFtZRgCIAEoCVIEbmFtZRItCgVp
bnB1dBgDIAEoCzIXLmdvb2dsZS5wcm90b2J1Zi5TdHJ1Y3RSBWlucHV0IhcKFUxpdmVTZXJ2ZXJJbnRlcnJ1cHRl
ZCI5ChFMaXZlU2VydmVyVHVybkVuZBIkCgV1c2FnZRgBIAEoCzIOLmxtMTUudjEuVXNhZ2VSBXVzYWdlIj0KD0xp
dmVTZXJ2ZXJFcnJvchIqCgVlcnJvchgBIAEoCzIULmxtMTUudjEuRXJyb3JEZXRhaWxSBWVycm9yKmIKBFJvbGUS
FAoQUk9MRV9VTlNQRUNJRklFRBAAEg0KCVJPTEVfVVNFUhABEhIKDlJPTEVfQVNTSVNUQU5UEAISDQoJUk9MRV9U
T09MEAMSEgoOUk9MRV9ERVZFTE9QRVIQBCq3AQoMRmluaXNoUmVhc29uEh0KGUZJTklTSF9SRUFTT05fVU5TUEVD
SUZJRUQQABIWChJGSU5JU0hfUkVBU09OX1NUT1AQARIYChRGSU5JU0hfUkVBU09OX0xFTkdUSBACEhsKF0ZJTklT
SF9SRUFTT05fVE9PTF9DQUxMEAMSIAocRklOSVNIX1JFQVNPTl9DT05URU5UX0ZJTFRFUhAEEhcKE0ZJTklTSF9S
RUFTT05fRVJST1IQBSr4AQoPUmVhc29uaW5nRWZmb3J0EiAKHFJFQVNPTklOR19FRkZPUlRfVU5TUEVDSUZJRUQQ
ABIYChRSRUFTT05JTkdfRUZGT1JUX09GRhABEh0KGVJFQVNPTklOR19FRkZPUlRfQURBUFRJVkUQAhIcChhSRUFT
T05JTkdfRUZGT1JUX01JTklNQUwQAxIYChRSRUFTT05JTkdfRUZGT1JUX0xPVxAEEhsKF1JFQVNPTklOR19FRkZP
UlRfTUVESVVNEAUSGQoVUkVBU09OSU5HX0VGRk9SVF9ISUdIEAYSGgoWUkVBU09OSU5HX0VGRk9SVF9YSElHSBAH
KvYBCglFcnJvckNvZGUSGgoWRVJST1JfQ09ERV9VTlNQRUNJRklFRBAAEhMKD0VSUk9SX0NPREVfQVVUSBABEhYK
EkVSUk9SX0NPREVfQklMTElORxACEhkKFUVSUk9SX0NPREVfUkFURV9MSU1JVBADEh4KGkVSUk9SX0NPREVfSU5W
QUxJRF9SRVFVRVNUEAQSHQoZRVJST1JfQ09ERV9DT05URVhUX0xFTkdUSBAFEhYKEkVSUk9SX0NPREVfVElNRU9V
VBAGEhUKEUVSUk9SX0NPREVfU0VSVkVSEAcSFwoTRVJST1JfQ09ERV9QUk9WSURFUhAIKm8KC0ltYWdlRGV0YWls
EhwKGElNQUdFX0RFVEFJTF9VTlNQRUNJRklFRBAAEhQKEElNQUdFX0RFVEFJTF9MT1cQARIVChFJTUFHRV9ERVRB
SUxfSElHSBACEhUKEUlNQUdFX0RFVEFJTF9BVVRPEAMqhwEKDlRvb2xDaG9pY2VNb2RlEiAKHFRPT0xfQ0hPSUNF
X01PREVfVU5TUEVDSUZJRUQQABIZChVUT09MX0NIT0lDRV9NT0RFX0FVVE8QARIdChlUT09MX0NIT0lDRV9NT0RF
X1JFUVVJUkVEEAISGQoVVE9PTF9DSE9JQ0VfTU9ERV9OT05FEAMqkgEKDUF1ZGlvRW5jb2RpbmcSHgoaQVVESU9f
RU5DT0RJTkdfVU5TUEVDSUZJRUQQABIYChRBVURJT19FTkNPRElOR19QQ00xNhABEhcKE0FVRElPX0VOQ09ESU5H
X09QVVMQAhIWChJBVURJT19FTkNPRElOR19NUDMQAxIWChJBVURJT19FTkNPRElOR19BQUMQBEJDCgtkZXYubG0x
NS52MUIJTG0xNVByb3RvUAFaJ2dpdGh1Yi5jb20vbG0xNS9sbTE1L2dlbi9sbTE1L3YxO2xtMTV2MWIGcHJvdG8z
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
    if part.data is not None:
        out.data = base64.b64decode(part.data)
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
        out.audio.data = base64.b64decode(delta.data)
        out.audio.part_index = delta.part_index
        if delta.media_type is not None:
            _set_wrapper(out.audio.media_type, delta.media_type)
    elif isinstance(delta, ImageDelta):
        out.image.part_index = delta.part_index
        if delta.media_type is not None:
            _set_wrapper(out.image.media_type, delta.media_type)
        if delta.data is not None:
            out.image.data = base64.b64decode(delta.data)
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
    _set_wrapper(out.cache, request.cache)
    return out


def _response_to_proto(pb, response: Response):
    out = pb.Response(
        id=response.id,
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
        return AudioDelta(
            data=_b64(msg.audio.data),
            part_index=msg.audio.part_index,
            media_type=_wrapper_value(msg.audio, "media_type"),
        )
    if kind == "image":
        source = msg.image.WhichOneof("source")
        kwargs = {
            "part_index": msg.image.part_index,
            "media_type": _wrapper_value(msg.image, "media_type"),
        }
        if source == "data":
            kwargs["data"] = _b64(msg.image.data)
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
        return StreamEvent(
            type="start",
            id=_wrapper_value(msg.start, "id"),
            model=_wrapper_value(msg.start, "model"),
        )
    if kind == "delta":
        return StreamEvent(type="delta", delta=_delta_from_proto(pb, msg.delta.delta))
    if kind == "end":
        finish_reason = _maps(pb)["finish"][1].get(msg.end.finish_reason)
        return StreamEvent(
            type="end",
            finish_reason=finish_reason,
            usage=_usage_from_proto(pb, msg.end.usage) if msg.end.HasField("usage") else None,
            provider_data=_struct_value(msg.end, "provider_data"),
        )
    if kind == "error":
        return StreamEvent(type="error", error=_error_from_proto(pb, msg.error.error))
    raise ValueError("event kind missing")


def _request_from_proto(pb, msg):
    return Request(
        model=msg.model,
        messages=tuple(_message_from_proto(pb, m) for m in msg.messages),
        system=_system_from_proto(pb, msg.system) if msg.HasField("system") else None,
        tools=tuple(_tool_from_proto(pb, t) for t in msg.tools),
        config=_config_from_proto(pb, msg.config) if msg.HasField("config") else Config(),
        cache=_wrapper_value(msg, "cache") if msg.HasField("cache") else True,
    )


def _response_from_proto(pb, msg):
    return Response(
        id=msg.id,
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
        return LiveClientEvent(type="audio", data=_b64(msg.audio.data))
    if kind == "video":
        return LiveClientEvent(type="video", data=_b64(msg.video.data))
    if kind == "text":
        return LiveClientEvent(type="text", text=msg.text.text)
    if kind == "tool_result":
        return LiveClientEvent(
            type="tool_result",
            id=msg.tool_result.id,
            content=tuple(_part_from_proto(pb, p) for p in msg.tool_result.content),
        )
    if kind == "interrupt":
        return LiveClientEvent(type="interrupt")
    if kind == "end_audio":
        return LiveClientEvent(type="end_audio")
    raise ValueError("live client event missing")


def _live_server_event_from_proto(pb, msg):
    kind = msg.WhichOneof("event")
    if kind == "audio":
        return LiveServerEvent(type="audio", data=_b64(msg.audio.data))
    if kind == "text":
        return LiveServerEvent(type="text", text=msg.text.text)
    if kind == "tool_call":
        return LiveServerEvent(
            type="tool_call",
            id=msg.tool_call.id,
            name=msg.tool_call.name,
            input=_struct_value(msg.tool_call, "input") or {},
        )
    if kind == "interrupted":
        return LiveServerEvent(type="interrupted")
    if kind == "turn_end":
        return LiveServerEvent(type="turn_end", usage=_usage_from_proto(pb, msg.turn_end.usage))
    if kind == "error":
        return LiveServerEvent(type="error", error=_error_from_proto(pb, msg.error.error))
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
        bytes_data=value.bytes_data,
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
    out = pb.BatchRequest(model=value.model)
    out.requests.extend(_request_to_proto(pb, r) for r in value.requests)
    _copy_struct(out.extensions, value.extensions)
    return out


def _batch_request_from_proto(pb, msg):
    return BatchRequest(
        model=msg.model,
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
    return out


def _audio_generation_response_from_proto(pb, msg):
    return AudioGenerationResponse(
        audio=AudioPart(**_media_source_from_proto(msg.audio.source)),
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
