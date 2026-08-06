#!/bin/bash
# prog3 任务数扫描：对每个任务数 N，自动修改 mandelbrot.ispc 的 rowsPerTask/launch[N]，
# 重编译后跑 view 1 和 view 2（--tasks），记录 ISPC 加速比与 task ISPC 加速比。
# 注意：N 必须整除 1200（height），否则末尾余数行不会被计算。
# 用法: ./run_sweep.sh [输出文件]   默认 results_prog3.md
set -u
cd "$(dirname "$0")"

OUT="${1:-results_prog3.md}"
ISPC_FILE="mandelbrot.ispc"
BACKUP="${ISPC_FILE}.sweep_bak"
TASK_COUNTS="2 4 8 16 20 40"

# ispc 编译器：优先 PATH，否则用 lab1 自带的
if command -v ispc >/dev/null 2>&1; then
    ISPC_BIN="ispc"
else
    ISPC_BIN="$(cd .. && pwd)/ispc-v1.28.1-linux/bin/ispc"
fi

cp "$ISPC_FILE" "$BACKUP"

restore() {
    cp "$BACKUP" "$ISPC_FILE"
    rm -f "$BACKUP"
    make clean >/dev/null 2>&1
    make ISPC="$ISPC_BIN" >/dev/null 2>&1
}
trap restore EXIT

{
    echo "# prog3 ISPC 任务数扫描（--tasks）"
    echo
    echo "- 测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- 机器: $(nproc) 核 CPU，ISPC target: avx2-i32x8（8 路 SIMD）"
    echo
    echo "| 任务数 | 每任务行数 | view1 ISPC 加速 | view1 task 耗时(ms) | view1 task 加速 | view2 ISPC 加速 | view2 task 耗时(ms) | view2 task 加速 |"
    echo "|:------:|:----------:|:---------------:|--------------------:|:---------------:|:---------------:|--------------------:|:---------------:|"
} > "$OUT"

for n in $TASK_COUNTS; do
    sed -i "s/rowsPerTask = height \/ [0-9]*/rowsPerTask = height \/ $n/" "$ISPC_FILE"
    sed -i "s/launch\[[0-9]*\]/launch[$n]/" "$ISPC_FILE"
    if ! (make clean >/dev/null 2>&1 && make ISPC="$ISPC_BIN" >/dev/null 2>&1); then
        echo "| $n | 编译失败 | - | - | - | - | - | - |" >> "$OUT"
        echo "任务数 $n 编译失败"
        continue
    fi

    out1=$(./mandelbrot_ispc --tasks 2>&1)
    out2=$(./mandelbrot_ispc --view 2 --tasks 2>&1)

    ispc1=$(echo "$out1" | grep -oP '\(\K[0-9.]+(?=x speedup from ISPC\))')
    task1=$(echo "$out1" | grep 'multicore ispc' | grep -oP '\[\K[0-9.]+(?=\] ms)')
    tacc1=$(echo "$out1" | grep -oP '\(\K[0-9.]+(?=x speedup from task ISPC\))')
    ispc2=$(echo "$out2" | grep -oP '\(\K[0-9.]+(?=x speedup from ISPC\))')
    task2=$(echo "$out2" | grep 'multicore ispc' | grep -oP '\[\K[0-9.]+(?=\] ms)')
    tacc2=$(echo "$out2" | grep -oP '\(\K[0-9.]+(?=x speedup from task ISPC\))')

    rows=$((800 / n))
    echo "| $n | $rows | ${ispc1:-?}x | ${task1:-?} | ${tacc1:-?}x | ${ispc2:-?}x | ${task2:-?} | ${tacc2:-?}x |" >> "$OUT"
    echo "任务数 $n 完成: view1 task ${tacc1:-?}x / view2 task ${tacc2:-?}x"
done

echo
echo "结果已写入 $OUT（mandelbrot.ispc 已恢复原样）"
