#!/bin/bash
# prog2 自动测试：扫描 VECTOR_WIDTH = 2/4/8/16
#   - 每个宽度跑 ./myexp -s 3（测不整除尾巴的正确性）
#   - 每个宽度跑 ./myexp -s 10000（记录 Vector Utilization / Total Vector Instructions）
# 用法: ./run_sweep.sh [输出文件]   默认 results_prog2.md
set -u
cd "$(dirname "$0")"

OUT="${1:-results_prog2.md}"
HEADER="CS149intrin.h"
ORIG_WIDTH=$(grep -oP '#define\s+VECTOR_WIDTH\s+\K\d+' "$HEADER")

{
    echo "# prog2 向量利用率扫描"
    echo
    echo "- 测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- 命令: \`./myexp -s 3\`（正确性/尾巴）与 \`./myexp -s 10000\`（统计）"
    echo
    echo "| VECTOR_WIDTH | -s 3 验证 | -s 10000 验证 | Total Vector Instructions | Vector Utilization |"
    echo "|:------------:|:---------:|:-------------:|--------------------------:|-------------------:|"
} > "$OUT"

for w in 2 4 8 16; do
    sed -i "s/^#define VECTOR_WIDTH .*/#define VECTOR_WIDTH $w/" "$HEADER"
    make clean >/dev/null 2>&1
    if ! make >/dev/null 2>&1; then
        echo "| $w | 编译失败 | - | - | - |" >> "$OUT"
        echo "VECTOR_WIDTH=$w 编译失败"
        continue
    fi

    out3=$(./myexp -s 3 2>&1)
    if echo "$out3" | grep -q '@@@ Failed'; then r3="❌ 失败"; else r3="✅ 通过"; fi

    out10k=$(./myexp -s 10000 2>&1)
    if echo "$out10k" | grep -q '@@@ Failed'; then r10k="❌ 失败"; else r10k="✅ 通过"; fi
    instr=$(echo "$out10k" | grep 'Total Vector Instructions' | grep -oP '\d+')
    util=$(echo "$out10k" | grep 'Vector Utilization' | grep -oP '[\d.]+%')

    echo "| $w | $r3 | $r10k | ${instr:--} | ${util:--} |" >> "$OUT"
    echo "VECTOR_WIDTH=$w 完成: -s3 $r3 / -s10000 $r10k / 指令数 ${instr:-?} / 利用率 ${util:-?}"
done

# 恢复原来的 VECTOR_WIDTH 并重编
sed -i "s/^#define VECTOR_WIDTH .*/#define VECTOR_WIDTH $ORIG_WIDTH/" "$HEADER"
make clean >/dev/null 2>&1 && make >/dev/null 2>&1

echo
echo "结果已写入 $OUT（VECTOR_WIDTH 已恢复为 $ORIG_WIDTH）"
