#include "tasksys.h"


IRunnable::~IRunnable() {}

ITaskSystem::ITaskSystem(int num_threads) {}
ITaskSystem::~ITaskSystem() {}

/*
 * ================================================================
 * Serial task system implementation
 * ================================================================
 */

const char* TaskSystemSerial::name() {
    return "Serial";
}

TaskSystemSerial::TaskSystemSerial(int num_threads): ITaskSystem(num_threads) {
}

TaskSystemSerial::~TaskSystemSerial() {}

void TaskSystemSerial::run(IRunnable* runnable, int num_total_tasks) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                          const std::vector<TaskID>& deps) {
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemSerial::sync() {
    return;
}

/*
 * ================================================================
 * Parallel Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelSpawn::name() {
    return "Parallel + Always Spawn";
}

TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
}

TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelSpawn in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Spinning Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSpinning::name() {
    return "Parallel + Thread Pool + Spin";
}

TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    for (int i = 0; i < num_total_tasks; i++) {
        runnable->runTask(i, num_total_tasks);
    }

    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // NOTE: CS149 students are not expected to implement TaskSystemParallelThreadPoolSpinning in Part B.
    return;
}

/*
 * ================================================================
 * Parallel Thread Pool Sleeping Task System Implementation
 * ================================================================
 */

const char* TaskSystemParallelThreadPoolSleeping::name() {
    return "Parallel + Thread Pool + Sleep";
}

void TaskSystemParallelThreadPoolSleeping::Sleepingwork(){
    // ru guo ci shi de ready_queue shi kong de na me ci shi jiu chen shui 
    while(true){
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock,[&]{
            if(stop.load()) return true;
            if(ready_queue.empty()) return false;
            Lanuch* f = ready_queue.front();
            return f->next_task_id.load() < f->num_total_tasks_;
        });
        if(stop.load()) return;
        //ru guo mei you de hua ci shi jiu zhi jie kai shi zhi xing 
        Lanuch* node = ready_queue.front();
        lock.unlock();
        while(true){
            // zhe li shi zhi xing ren wu de luo ji 
            int id = node->next_task_id.fetch_add(1);
            if(id >= node->num_total_tasks_)break;
            node->runnable_->runTask(id,node->num_total_tasks_);
            if(node->complete.fetch_add(1) == node->num_total_tasks_ - 1){
                // ci shi dai biao yi ge lanuch jie shu wan cheng le , ci shi jiu ke yi 
                // geng gai dui ying de shu ju jie gou le 
                std::lock_guard<std::mutex> g(mtx);
                node->finish = true;
                ready_queue.pop_front();
                finish_count++;
                if(finish_count == submit_count) done_cv.notify_one();
                for(auto dep: node->dependents){
                    if(--dep->deps_remaining == 0){
                        ready_queue.push_back(dep);
                    }
                }
                cv.notify_all();
            }
        }
    }
}

TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    workers.resize(num_threads);
    n_threads = num_threads;
    for(int i = 0; i < num_threads; i++){
        workers[i] = std::thread(&TaskSystemParallelThreadPoolSleeping::Sleepingwork,this);
    }

}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping() {
    //
    // TODO: CS149 student implementations may decide to perform cleanup
    // operations (such as thread pool shutdown construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    { 
    std::lock_guard<std::mutex> g(mtx);
    stop = true;
    }
    cv.notify_all();
    for(int i = 0; i < n_threads;i++){
        if(workers[i].joinable()) workers[i].join();
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    runAsyncWithDeps(runnable,num_total_tasks,{});
    sync();


}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //
    std::lock_guard<std::mutex> g(mtx);
    Lanuch* node = new Lanuch();
    node->id = next_task_id++;
    node->runnable_ = runnable;
    node->num_total_tasks_ = num_total_tasks; 

    for(TaskID dep_id : deps){
        Lanuch* dep = lanuches[dep_id];
        if(!dep->finish){
            dep->dependents.push_back(node);
            node->deps_remaining++;
        }
    }

    lanuches[node->id] = node;
    submit_count++;
    if(node->deps_remaining == 0){
        ready_queue.push_back(node);
        cv.notify_all();
    }

    return node->id;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //
    // cong ci shi de 
    std::unique_lock<std::mutex> lock(mtx);
    done_cv.wait(lock,[&]{ return finish_count == submit_count;});
    return;
}
