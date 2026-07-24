#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cprq_root="$(cd "${script_dir}/../.." && pwd)"
embdc_root="$(cd "${cprq_root}/.." && pwd)"
build_root="${cprq_root}/build_cuda_cprq_imprv"
nvcomp_root="${embdc_root}/venv/lib/python3.10/site-packages/nvidia/libnvcomp"
cutensor_root="${cprq_root}/tensorCodec/third_party/cutensor/lib/12"

g++ \
    -std=c++17 \
    -O2 \
    -I"${cprq_root}/include" \
    "${script_dir}/export_nvcomp_package_raw.cpp" \
    -L"${build_root}" \
    -L"${cutensor_root}" \
    -Wl,--start-group \
    -lcuda_cprq_imprv \
    "${build_root}/tensorCodec/libTZ.a" \
    -Wl,--end-group \
    "${nvcomp_root}/lib64/libnvcomp.so.5" \
    -lcudart \
    -lcusolver \
    -lcublas \
    -lcusparse \
    -lcurand \
    -lcutensor \
    -lgomp \
    -lpthread \
    -ldl \
    -Wl,-rpath,"${nvcomp_root}/lib64:${cutensor_root}" \
    -o "${script_dir}/export_nvcomp_package_raw"

echo "Built ${script_dir}/export_nvcomp_package_raw"
