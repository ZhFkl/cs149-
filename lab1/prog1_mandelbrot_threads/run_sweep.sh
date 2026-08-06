#!/bin/bash
# 自动测试 mandelbrot 在 view 1 / view 2 下、线程数 2..8 的耗时与加速比
# 用法: ./run_sweep.sh [输出文件]   默认输出到 results.md
set -u
cd "$(dirname "$0")"

OUT="${1:-results.md}"

if [ ! -x ./mandelbrot ]; then
    echo "mandelbrot 不存在，先编译..."
    make
fi

{
    echo "# Mandelbrot 多线程加速比测试"
    echo
    echo "- 测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "- 机器: $(nproc) 核 CPU"
    echo
    echo "| View | 线程数 | 串行耗时 (ms) | 多线程耗时 (ms) | 加速比 |"
    echo "|:----:|:------:|--------------:|----------------:|-------:|"
} > "$OUT"

for view in 1 2; do
    for t in 2 3 4 5 6 7 8; do
        result=$(./mandelbrot --view "$view" -t "$t" 2>&1)
        serial=$(echo "$result" | grep 'mandelbrot serial' | grep -oP '\[\K[0-9.]+(?=\] ms)')
        thread=$(echo "$result" | grep 'mandelbrot thread' | grep -oP '\[\K[0-9.]+(?=\] ms)')
        speedup=$(echo "$result" | grep -oP '\(\K[0-9.]+(?=x speedup)')
        if [ -z "$speedup" ]; then
            echo "| $view | $t | 运行失败 | - | - |" >> "$OUT"
        else
            echo "| $view | $t | $serial | $thread | ${speedup}x |" >> "$OUT"
        fi
        echo "view $view / $t 线程 完成: 串行 ${serial:-?} ms, 多线程 ${thread:-?} ms, 加速比 ${speedup:-失败}x"
    done
done

echo
echo "结果已写入 $OUT"
