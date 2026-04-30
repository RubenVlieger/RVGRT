#!/usr/bin/env python3
"""
Post-processes Slang-generated CUDA .cu files for per-kernel compilation.

Slang's CUDA backend packs all shader resources and uniforms into a single
__constant__ 'SLANG_globalParams' struct. When multiple kernels are compiled
together this causes duplicate-symbol errors. This script:

1. Renames each kernel's GlobalParams struct to a unique name
2. Renames SLANG_globalParams to a unique symbol per kernel
3. Changes 'extern' to an actual definition so storage is allocated
4. Appends a host-side Launch_*() helper function directly into each .cu file
5. Generates a .cuh header with Launch_* declarations for CudaRenderer.cu

Each resulting .cu file is compiled independently by nvcc.
"""
import argparse
import os
import re
import sys

# Map Slang-generated field types to C++ wrapper parameter types
FIELD_TYPE_MAP = {
    'CUsurfObject': 'cudaSurfaceObject_t',
    'CUtexObject': 'cudaTextureObject_t',
    'SamplerState': None,  # ignored in CUDA
}

# Types that we pass by const reference from host code
POINTER_TYPE_ALIASES = {
    'CameraData_natural_0': 'CameraData',
    'FrameData_0': 'FrameData',
    'ExposureData_0': 'ExposureData',
    'CharacterGPUData_natural_0': 'CharacterGPUData',
    'SectorInfo_0': 'SectorInfo',
    'SectorWorkItem_0': 'SectorWorkItem',
    'BrickWorkItem_0': 'BrickWorkItem',
    'GlyphInstance_0': 'GlyphInstance',
    'TextOverlayData_0': 'TextOverlayData',
}

SCALAR_TYPES = {'int', 'uint', 'float', 'int32_t', 'uint32_t', 'int64_t', 'uint64_t'}


def parse_struct_fields(text):
    """Extract fields from a GlobalParams struct body."""
    fields = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
        # Match: Type fieldName;
        # Handles: CUsurfObject, CUtexObject, SamplerState, StructuredBuffer<T>, T*
        m = re.match(r'(\w+(?:<[^>]+>)?(?:\s*\*)?)\s+(\w+)\s*;', line)
        if m:
            fields.append((m.group(1), m.group(2)))
    return fields


def map_field_to_param(field_type, field_name):
    """Return (cpp_type, param_name) for a wrapper parameter, or None to skip."""
    base_type = field_type.replace('*', '').strip()

    # Direct mappings (surfaces, textures, samplers)
    if base_type in FIELD_TYPE_MAP:
        cpp_type = FIELD_TYPE_MAP[base_type]
        if cpp_type is None:
            return None  # skip samplers
        return (cpp_type, field_name.replace('_0', ''))

    # Pointer types (charData)
    if field_type.endswith('*'):
        return ('void*', field_name.replace('_0', ''))

    # StructuredBuffer / RWStructuredBuffer
    if base_type.startswith('StructuredBuffer<') or base_type.startswith('RWStructuredBuffer<'):
        return ('void*', field_name.replace('_0', ''))

    # Scalar types passed by value
    if base_type in SCALAR_TYPES:
        alias = POINTER_TYPE_ALIASES.get(base_type, base_type)
        return (alias, field_name.replace('_0', ''))

    # Vector / small struct types passed by const reference
    alias = POINTER_TYPE_ALIASES.get(base_type, base_type)
    return (f'const {alias}&', field_name.replace('_0', ''))


def generate_wrapper(kernel_name, struct_name, fields):
    """Generate a host-side Launch_* function for this kernel."""
    params = []
    assignments = []

    for field_type, field_name in fields:
        mapping = map_field_to_param(field_type, field_name)
        if mapping is None:
            # SamplerState - set to nullptr in wrapper
            assignments.append(f'    gp.{field_name} = nullptr;')
            continue

        cpp_type, param_name = mapping
        params.append(f'{cpp_type} {param_name}')

        if field_type.endswith('*'):
            assignments.append(f'    gp.{field_name} = ({field_type.strip()}){param_name};')
        else:
            base_type = field_type.replace('*', '').strip()
            if base_type.startswith('StructuredBuffer<') or base_type.startswith('RWStructuredBuffer<'):
                inner = re.search(r'<([^>]+)>', base_type).group(1)
                assignments.append(f'    gp.{field_name}.data = ({inner}*){param_name};')
                assignments.append(f'    gp.{field_name}.count = 0;')
            elif base_type in POINTER_TYPE_ALIASES:
                assignments.append(f'    memcpy(&gp.{field_name}, &{param_name}, sizeof(gp.{field_name}));')
            else:
                assignments.append(f'    gp.{field_name} = {param_name};')

    param_list = ', '.join(params)
    decl_list = ', '.join(params)

    wrapper = f'''
extern "C" void Launch_{kernel_name}(
    cudaStream_t stream, dim3 grid, dim3 block,
    {param_list}
) {{
    {struct_name} gp = {{}};
{chr(10).join(assignments)}
    cudaMemcpyToSymbolAsync(SLANG_gp_{kernel_name}, &gp, sizeof(gp), 0, cudaMemcpyHostToDevice, stream);
    {kernel_name}<<<grid, block, 0, stream>>>();
}}
'''
    header = f'''extern "C" void Launch_{kernel_name}(
    cudaStream_t stream, dim3 grid, dim3 block,
    {decl_list}
);
'''
    return wrapper, header


def process_file(input_path, output_path, kernel_name):
    with open(input_path, 'r') as f:
        content = f.read()

    # Find the GlobalParams struct name
    struct_match = re.search(r'struct\s+(GlobalParams_\w+)\s*\{', content)
    if not struct_match:
        print(f'Warning: No GlobalParams struct found in {input_path}', file=sys.stderr)
        return None
    old_struct_name = struct_match.group(1)
    new_struct_name = f'GlobalParams_{kernel_name}'

    # Extract struct body
    start = struct_match.end()
    brace_depth = 1
    end = start
    for i in range(start, len(content)):
        if content[i] == '{':
            brace_depth += 1
        elif content[i] == '}':
            brace_depth -= 1
            if brace_depth == 0:
                end = i
                break
    struct_body = content[start:end]
    fields = parse_struct_fields(struct_body)

    # Post-process content
    content = content.replace(f'struct {old_struct_name}', f'struct {new_struct_name}')
    content = content.replace(f'extern "C" __constant__ {old_struct_name} SLANG_globalParams;',
                              f'__constant__ {new_struct_name} SLANG_gp_{kernel_name};')
    content = content.replace('#define globalParams_0 (&SLANG_globalParams)',
                              f'#define globalParams_0 (&SLANG_gp_{kernel_name})')

    wrapper, header = generate_wrapper(kernel_name, new_struct_name, fields)

    with open(output_path, 'w') as f:
        f.write(content)
        f.write(wrapper)

    return header


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Directory with Slang-generated .cu files')
    parser.add_argument('output_dir', help='Directory for post-processed .cu files')
    parser.add_argument('output_cuh', help='Header with Launch_* declarations')
    args = parser.parse_args()

    headers = []
    headers.append('// Auto-generated by gen_cuda_wrappers.py')
    headers.append('#pragma once')
    headers.append('#include <cuda_runtime.h>')
    headers.append('#include "renderer/ShaderTypes.h"')
    headers.append('')

    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.cu')])
    for filename in files:
        kernel_name = filename[:-3]  # strip .cu
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, f'{kernel_name}.cu')
        header = process_file(input_path, output_path, kernel_name)
        if header:
            headers.append(header)

    os.makedirs(os.path.dirname(args.output_cuh) or '.', exist_ok=True)
    with open(args.output_cuh, 'w') as f:
        f.write('\n'.join(headers))

    print(f'Post-processed {len(files)} kernels into {args.output_dir}')
    print(f'Generated header: {args.output_cuh}')


if __name__ == '__main__':
    main()
