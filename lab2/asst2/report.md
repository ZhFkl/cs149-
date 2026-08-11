请向 Gradescope 上的 *Assignment 2 (Write-up)* 作业提交一份简短的书面报告，回答以下内容：

 1. 描述你的任务系统实现（1 页即可）。除了对其工作方式的一般性描述外，请确保回答以下问题：
  * 你是如何决定管理线程的？（例如，你是否实现了线程池？）

        这里我是是用了线程池，在a和b的实验中，在构造函数中此时就生成对应的一个线程，然后让他们自选或这sleep来跑线程，等到有任务的时候在唤醒
        或者让他们推出自选，就能够接任务，任务完成之后从新回到睡眠状态或者自选状态


  * 你的系统如何将任务分配给工作线程？你使用的是静态分配还是动态分配？

        这里都有用到，在parallelSpwan的这个类中我是通过静态分配来给线程安排工作的， 就是通过算来来给每个线程分配好工作然后让他们跑。

        void TaskSystemParallelSpawn::work_task(IRunnable* runnable, int thread_id, int num_threads, int num_total_tasks){
            for (int i = thread_id; i < num_total_tasks; i += num_threads) {
                runnable->runTask(i, num_total_tasks);
            }
        }

        然后后续的spain和对应的sleep是通过动态分配来让线程工作， 此是是所有线程回去竞争任务，去强任务然后进行执行，不预先分配好。

        while(!stop.load()){
            if(seen.load() == epoch.load()) continue;
            int task_id  = next_task_id.fetch_add(1);
            if(runnable_ != nullptr && task_id < num_total_tasks_){
                // zhe li jin xing zi xuan ran hou qiang ren wu 
                // ci shi ru guo qiang dao le zen 
                runnable_->runTask(task_id,num_total_tasks_);
                complete.fetch_add(1);
            }
        }

            while(true){
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock,[&]{return epoch != seen || stop.load();});
            if(stop.load()){
                return ;
            }
            lock.unlock();
            while(true){
                int task_id = next_task_id.fetch_add(1);
                if(task_id >= num_total_tasks_)break;
                runnable_->runTask(task_id,num_total_tasks_);
                if(complete.fetch_add(1)  == num_total_tasks_ -1){
                    std::lock_guard<std::mutex> g(mtx);
                    done_cv.notify_one();
                }
            }
            lock.lock();
        }

  * 在 B 部分中，你是如何跟踪依赖关系以确保任务图正确执行的？

        struct Lanuch{
            TaskID id;
            IRunnable* runnable_;
            int num_total_tasks_;
            std::atomic<int> next_task_id{0};
            std::atomic<int> complete{0};
            std::vector<Lanuch*> dependents;
            int deps_remaining = 0;
            bool finish = false;
        };
        int n_threads;
        std::deque<Lanuch*> ready_queue;
        std::map<TaskID,Lanuch*> lanuches;


        这里是同过lanuch这个数据结构，首先这里有一个dependents，和 deps_remaining这个之，如果此时deps_remaining=0, 那么此时代表他之前的以来已经完成，然后现在可以执行这个lanuch了然后把他加入到ready_queue中，之后执行这个lanuch，完成之后通过dependents这个vector去更新别的lanuch的deps_remaining用来提醒那些以来已经完成了。 
        这样按照顺序的话就能保证此时的任务图正确的完成


 2. 在 A 部分中，你可能已经注意到，较简单的任务系统实现（例如，完全串行的实现，或每次启动都派生线程的实现）的性能与更高级的实现相当，有时甚至更好。请解释为什么会出现这种情况，并引用某些测试作为例子。例如，在什么情况下串行任务系统实现表现最好？为什么？在什么情况下每次启动都派生线程的实现与使用线程池的更高级并行实现表现相当？什么时候又不是这样？

        三种实现的差距不在“计算”，而在“每轮 launch 的固定开销”。总时间可以建模为：

            总时间 ≈ 有效计算 / 并行度 + 每轮固定开销 × launch 轮数

        三种实现的每轮固定开销对比：

        | 实现 | 每轮 launch 的固定开销 |
        |---|---|
        | Serial | 0（但没有并行加速） |
        | Always Spawn | N 次线程创建 + 销毁（Linux 上每线程几十 µs） |
        | 线程池（Spin/Sleep） | 几次原子操作 / 一次 notify（ns~µs 级） |

        串行何时最好：任务极小、launch 频繁时。以 super_super_light 为例（400 轮 launch × 64 个任务，
        base_iters=0，每个任务只有几条指令），整个测试的有效计算只有 ~11ms，并行加速没有肉可吃，
        但每个机制都要按轮付固定开销：Serial 零开销 ~11.6ms；Always Spawn 每轮重建线程，400 轮的创建费
        加起来 ~14ms，总耗时 ~26ms，反而比串行慢一倍多；睡眠线程池把每轮开销压到 µs 级（~11.6ms），
        追平串行。结论：任务粒度趋近“空调用”时，调度固定成本无处分摊，谁开销低谁赢。

        Spawn 何时与线程池相当：任务粒度大、launch 次数少时。以 mandelbrot_chunked 为例（单次重任务
        launch，串行 ~520ms）：Spawn 的线程创建费只有几百 µs，占总时间 ~0.1%，测不出来，于是
        Spawn ~286ms ≈ 线程池 ~286ms，两者相对串行都拿到接近 2 倍加速。结论：每轮计算量远大于线程
        创建费时，创建费被摊薄到噪声级，两者打平。

        何时不相当：轻任务 + 多轮 launch 时。同样是 super_super_light / ping_pong_equal 这种 400 轮的
        测试，Spawn 的创建费按轮收（400 × N × 几十 µs），总量超过有效计算本身；线程池的创建费只在
        构造函数付一次，之后每轮几乎免费。于是 Spawn ~26ms vs 线程池 ~12ms，差距约 2 倍。结论：
        launch 越频繁、每轮任务越轻，Spawn 的“每轮重建”税越致命，线程池“一次建仓、终身复用”的
        优势全部兑现。

        所以“更高级”的实现并不总是更快——胜负由任务粒度与 launch 频率决定，不由实现复杂度决定。

 3. 描述你为本作业实现的一个测试。这个测试做了什么，它旨在检查什么，以及你是如何验证你的作业解决方案在你的测试上表现良好的？你添加的测试结果是否促使你修改了作业实现？