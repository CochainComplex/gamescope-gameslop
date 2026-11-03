"""SDK-free cold negotiation; bindings retain original backend callables.

The same .def used by the C/C++ header supplies constants and profiles. No SDK,
NumPy, process, thread, tensor conversion or runtime registry is imported here.
"""
from pathlib import Path
import re
import struct
from typing import NamedTuple


class Contract(NamedTuple):
    size: int
    abi: int
    operation: int
    execution: int
    payload: int
    model_version: int
    semantics: int
    state: int
    transport: int
    features: int
    max_inflight: int
    reserved: int

    def as_dict(self):
        return self._asdict()


class Requirements(NamedTuple):
    operation: int
    payload: int
    execution: int
    model_version: int
    semantics: int
    state: int
    transport: int
    features: int = 0


def _declarations():
    values, profiles = {}, {}
    source = Path(__file__).resolve().parents[1] / 'src/framegen/backend_contract.def'
    for line in source.read_text().splitlines():
        line = line.split('//', 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r'GSC_CONSTANT\((\w+),\s*(\d+)\)', line)
        if match:
            name, value = match.groups()
            if name in values:
                raise RuntimeError('duplicate accelerator constant')
            values[name] = int(value)
            continue
        match = re.fullmatch(r'GSC_PROFILE\((.*)\)', line)
        if not match:
            raise RuntimeError('invalid accelerator declaration')
        fields = [x.strip() for x in match.group(1).split(',')]
        if len(fields) != 10 or fields[0] in profiles:
            raise RuntimeError('invalid accelerator profile')
        numbers = []
        for expression in fields[1:]:
            number = 0
            for token in map(str.strip, expression.split('|')):
                number |= int(token) if token.isdecimal() else values[token]
            numbers.append(number)
        profiles[fields[0]] = Contract(struct.calcsize('<12I'), values['GSC_BACKEND_ABI'], *numbers, 0)
    return values, profiles


VALUES, PROFILES = _declarations()
_SCHEMAS = {c.payload: c for c in PROFILES.values()}


def requirements_for(contract, features=None):
    return Requirements(contract.operation, contract.payload, contract.execution, contract.model_version,
                        contract.semantics, contract.state, contract.transport,
                        contract.features if features is None else features)


def _record(value, record):
    # Tools intentionally load isolated script modules. Wire compatibility must
    # not depend on which copy created an immutable named tuple.
    if (not isinstance(value, tuple) or getattr(type(value), '_fields', None) != record._fields
            or len(value) != len(record._fields)):
        return None
    if any(type(x) is not int or not 0 <= x <= 0xffffffff for x in value):
        return None
    # Use the validated words, never potentially inconsistent foreign attributes.
    return value if type(value) is record else record(*value)


def _valid(contract):
    if contract is None:
        return False
    schema = _SCHEMAS.get(contract.payload)
    if schema is None or contract.size != 48 or contract.abi != VALUES['GSC_BACKEND_ABI'] or contract.reserved:
        return False
    fields = ('operation', 'execution', 'model_version', 'semantics', 'state', 'transport', 'max_inflight')
    return (all(getattr(contract, f) == getattr(schema, f) for f in fields)
            and not contract.features & ~(schema.features | VALUES['GSC_FEATURE_EXACT_DEVICE']))


def require_contract(contract, required):
    contract, required = _record(contract, Contract), _record(required, Requirements)
    if (not _valid(contract) or required is None
            or any(getattr(contract, f) != getattr(required, f)
                   for f in ('operation', 'payload', 'execution', 'model_version', 'semantics', 'state'))
            or contract.transport & required.transport != required.transport
            or contract.features & required.features != required.features):
        raise ValueError('accelerator contract does not satisfy the typed operation')


class Binding:
    __slots__ = ('owner', 'contract', 'execute', 'submit', 'poll', 'collect', 'checkpoint',
                 'discard', 'retire', 'reset', 'close', 'evidence')


def bind_backend(owner, contract, required, **callbacks):
    """Negotiate once; result attributes are the original bound callables.

    Async C callbacks retain their native signatures, including context/error
    arguments. This is a typed binding, not a generic per-job dispatcher.
    """
    contract = _record(contract, Contract)
    require_contract(contract, required)
    names = Binding.__slots__[2:]
    if any(name not in names or (value is not None and not callable(value)) for name, value in callbacks.items()):
        raise ValueError('invalid accelerator binding callback')
    needed = {'submit', 'poll', 'collect'} if contract.execution == VALUES['GSC_EXEC_ASYNC'] else {'execute'}
    if contract.execution == VALUES['GSC_EXEC_ASYNC'] and callbacks.get('execute') is not None:
        raise ValueError('async operation cannot silently use blocking execution')
    if contract.execution == VALUES['GSC_EXEC_BLOCKING'] and any(callbacks.get(n) is not None for n in ('submit', 'poll', 'collect', 'discard', 'retire')):
        raise ValueError('blocking operation cannot manufacture async tickets')
    for feature, methods in (('GSC_FEATURE_CHECKPOINT', ('checkpoint',)),
                             ('GSC_FEATURE_DISCARD_RESULT', ('discard', 'retire')),
                             ('GSC_FEATURE_EXECUTION_EVIDENCE', ('evidence',))):
        if contract.features & VALUES[feature]:
            needed.update(methods)
        elif any(callbacks.get(n) is not None for n in methods):
            raise ValueError('callback is not advertised by the accelerator contract')
    if any(not callable(callbacks.get(name)) for name in needed):
        raise ValueError('required accelerator callback is missing')
    binding = Binding()
    binding.owner, binding.contract = owner, contract
    for name in names:
        setattr(binding, name, callbacks.get(name))
    return binding
