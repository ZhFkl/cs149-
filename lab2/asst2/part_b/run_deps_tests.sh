#!/bin/bash
# Part B 依赖图测试脚本：逐个运行 5 个依赖测试，报告通过/失败
# 用法: ./run_deps_tests.sh [线程数]   （默认 2 线程）
cd "$(dirname "$0")"

N_THREADS=${1:-2}

if [ ! -x ./runtasks ]; then
    echo "未找到 ./runtasks，先执行 make ..."
    make || exit 1
fi

TESTS=(
    simple_run_deps_test
    strict_diamond_deps_async
    strict_graph_deps_small_async
    strict_graph_deps_med_async
    strict_graph_deps_large_async
)

pass=0
fail=0
for t in "${TESTS[@]}"; do
    echo "===== $t (threads=$N_THREADS) ====="
    # simple_run_deps_test 的 SleepTask 会让 4 个实现各串行睡 3 轮计时，完整要 ~30 分钟；
    # 调试期建议改用独立驱动 /tmp/sleep_bench（只测 Sleep 实现，~80s）
    tmo=300
    [ "$t" = "simple_run_deps_test" ] && tmo=2100
    out=$(timeout $tmo ./runtasks -n "$N_THREADS" "$t" 2>&1)
    rc=$?
    echo "$out"
    if [ $rc -eq 124 ]; then
        echo ">>> FAIL: 超时 ${tmo}s（疑似死锁）"
        fail=$((fail+1))
    elif [ $rc -ne 0 ]; then
        echo ">>> FAIL: 进程异常退出 (rc=$rc，可能崩溃)"
        fail=$((fail+1))
    elif echo "$out" | grep -q "ERROR"; then
        echo ">>> FAIL: 正确性校验未通过"
        fail=$((fail+1))
    else
        echo ">>> PASS"
        pass=$((pass+1))
    fi
    echo
done

echo "=========================================="
echo "通过 $pass / ${#TESTS[@]}，失败 $fail"
if [ $fail -eq 0 ]; then
    echo "依赖测试全部通过"
else
    exit 1
fi
