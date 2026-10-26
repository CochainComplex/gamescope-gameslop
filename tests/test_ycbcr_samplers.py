#!/usr/bin/env python3
"""Check converted samplers in the compositor's actual compiled SPIR-V headers."""
from pathlib import Path
import re
import struct
import sys


SHADERS = ('cs_composite_blit', 'cs_composite_blur', 'cs_composite_blur_cond',
           'cs_composite_rcas', 'cs_composite_rcas_1px', 'cs_gaussian_blur_horizontal')


def read_words(path):
    data = path.read_bytes()
    if path.suffix == '.h':
        text = data.decode('ascii')
        data = b''.join(struct.pack('<I', int(word, 16)) for word in
                        re.findall(r'0x[0-9a-fA-F]+', text[text.index('{'):text.rindex('}')]))
    if len(data) % 4:
        raise ValueError('unaligned SPIR-V module')
    return list(struct.unpack(f'<{len(data)//4}I', data))


def instructions(words):
    if len(words) < 5 or words[0] != 0x07230203 or words[4] != 0:
        raise ValueError('invalid SPIR-V header')
    cursor, function = 5, None
    while cursor < len(words):
        size, opcode = words[cursor] >> 16, words[cursor] & 0xffff
        if not size or cursor + size > len(words):
            raise ValueError('truncated SPIR-V instruction')
        args = words[cursor+1:cursor+size]
        if opcode == 54:  # OpFunction
            function = args[1]
        yield opcode, args, function
        if opcode == 56:  # OpFunctionEnd
            function = None
        cursor += size


def collect_metadata(ops):
    """Resolve descriptor roots, integer constants and result function owners."""
    decorations, integers, constants, functions = {}, {}, {}, {}
    for opcode, args, function in ops:
        if opcode == 71 and args[1] in (33, 34):  # Binding, DescriptorSet
            decorations.setdefault(args[0], {})[args[1]] = args[2]
        elif opcode == 21:  # OpTypeInt
            integers[args[0]] = (args[1], args[2])
        elif opcode == 43:  # OpConstant, not OpSpecConstant/OpConstantComposite
            constants[args[1]] = (args[0], args[2:])
        if opcode in (59, 61, 65, 66, 67, 70, 83, 86, 169, 245):
            functions[args[1]] = function
    roots = {item for item, values in decorations.items() if values.get(33) == 4 and values.get(34) == 0}
    return roots, integers, constants, functions


def opaque_ancestry(ops, roots):
    """Reach a fixed point over pointer/opaque aliases from descriptor roots."""
    # Each link retains opaque/pointer ancestry. Numeric image-sample results
    # deliberately end propagation, so normal vec4 helper calls remain legal.
    parents = {root: set() for root in roots}
    changed = True
    while changed:
        changed = False
        for opcode, args, _ in ops:
            result, sources = None, []
            if opcode in (61, 65, 66, 67, 70, 83):
                result, sources = args[1], [args[2]]
            elif opcode == 86:  # OpSampledImage
                result, sources = args[1], args[2:4]
            elif opcode == 169:  # OpSelect
                result, sources = args[1], args[3:5]
            elif opcode == 245:  # OpPhi (ignore predecessor label operands)
                result, sources = args[1], args[2::2]
            elif opcode == 62:  # OpStore, including function-local opaque aliases
                result, sources = args[0], [args[1]]
            linked = {item for item in sources if item in parents}
            if linked and not linked.issubset(parents.get(result, set())):
                parents.setdefault(result, set()).update(linked)
                changed = True
    return parents


def array_index(args, roots, integers, constants):
    """Validate access indices and decode a direct descriptor-array slot."""
    # The descriptor is a one-dimensional sampler array. Pointer-chain
    # element/index operands must all be ordinary integral constants.
    for index in args[3:]:
        if index not in constants or constants[index][0] not in integers:
            raise ValueError('converted sampler access uses a nonconstant integral index')
    if args[2] not in roots:
        return None
    if len(args[3:]) != 1:
        raise ValueError('unexpected converted sampler array shape')
    type_id, literal = constants[args[3]]
    width, signed = integers[type_id]
    value = sum(word << (32*i) for i, word in enumerate(literal))
    if width not in (32, 64) or len(literal) != width//32 or (signed and value >> (width-1)):
        raise ValueError('invalid converted sampler index literal')
    return value


def check_module(words, all_slots=True):
    """Binding 4/set 0 only; propagate opaque resources, never sampled vec4s."""
    ops = list(instructions(words))
    roots, integers, constants, functions = collect_metadata(ops)
    if len(roots) != 1:
        raise ValueError('expected one converted sampler descriptor at set 0 binding 4')
    parents = opaque_ancestry(ops, roots)
    slots, loads, sampled = set(), set(), set()
    for opcode, args, function in ops:
        if opcode in (65, 66, 67, 70) and args[2] in parents:
            slot = array_index(args, roots, integers, constants)
            if slot is not None:
                slots.add(slot)
        elif opcode == 57 and any(item in parents for item in args[3:]):
            raise ValueError('converted sampler pointer/image passed to OpFunctionCall')
        elif opcode == 254 and args[0] in parents:  # OpReturnValue
            raise ValueError('converted sampler pointer/image returned across a function boundary')
        elif opcode == 61 and args[1] in parents:
            loads.add(args[1])
        elif opcode == 60 and args[2] in parents:  # OpImageTexelPointer
            raise ValueError('converted sampler used by forbidden image operation 60')
        elif 87 <= opcode <= 107 and args[0 if opcode == 99 else 2] in parents:
            # Project converted sampling uses textureLod only. Explicitly reject
            # fetch/gather/query operations and OpImage extraction (100), even
            # if this same object also feeds an allowed sample elsewhere.
            if opcode != 88:  # OpImageSampleExplicitLod
                raise ValueError(f'converted sampler used by forbidden image operation {opcode}')
            if function is None or functions.get(args[2]) != function:
                raise ValueError('converted sampler is not fetched locally')
            pending = [args[2]]
            while pending:
                item = pending.pop()
                if item not in sampled:
                    sampled.add(item)
                    pending.extend(parents[item])
    if not slots or not loads or not loads.issubset(sampled):
        raise ValueError('converted sampler loads must feed local image sampling')
    if all_slots and slots != set(range(8)):
        raise ValueError(f'expected all eight compositor layer slots, got {sorted(slots)}')
    return sorted(slots)


def check_forbidden_uses(words):
    """Opcode-only parser policy negatives; NOT valid executable GPU modules."""
    for forbidden in (95, 96, 100, 103):  # Fetch, Gather, Image, QuerySizeLod
        mutated = words[:5]
        cursor, next_id, changed = 5, words[3], False
        for opcode, _, _ in instructions(words):
            size = words[cursor] >> 16
            instruction = words[cursor:cursor+size]
            mutated.extend(instruction)
            if opcode == 88:
                # Keep the allowed sample too: an extra forbidden use must fail
                # even when the very same loaded object was sampled locally.
                instruction[0] = (instruction[0] & 0xffff0000) | forbidden
                instruction[2] = next_id
                next_id += 1
                mutated.extend(instruction)
                changed = True
            cursor += size
        mutated[3] = next_id
        if not changed:
            raise ValueError('policy negative has no sample operation to mutate')
        try:
            check_module(mutated)
        except ValueError as error:
            if 'forbidden image operation' not in str(error):
                raise
        else:
            raise ValueError(f'policy negative accepted forbidden image opcode {forbidden}')


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: test_ycbcr_samplers.py BUILD/src/gamescope.p')
    directory = Path(sys.argv[1])
    for name in SHADERS:
        path = directory / f'{name}.h'
        try:
            words = read_words(path)
            slots = check_module(words)
            if name == SHADERS[0]:
                check_forbidden_uses(words)
        except (OSError, ValueError, IndexError) as error:
            raise SystemExit(f'{path}: {error}') from error
        print(f'{name}: converted sampler slots {slots}, constant indices and local fetches PASS')


if __name__ == '__main__':
    main()
