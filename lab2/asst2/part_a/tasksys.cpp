#include "tasksys.h"
#include <stdio.h>
#include <thread>
#include <mutex>

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
    // You do not need to implement this method.
    return 0;
}


void TaskSystemSerial::sync() {
    // You do not need to implement this method.
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
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    n_threads = num_threads;
}

void TaskSystemParallelSpawn::work_task(IRunnable* runnable, int thread_id, int num_threads, int num_total_tasks){
    for (int i = thread_id; i < num_total_tasks; i += num_threads) {
        runnable->runTask(i, num_total_tasks);
    }
}


TaskSystemParallelSpawn::~TaskSystemParallelSpawn() {}

void TaskSystemParallelSpawn::run(IRunnable* runnable, int num_total_tasks) {


    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (int t = 0; t < n_threads; t++) {
        workers.emplace_back(TaskSystemParallelSpawn::work_task,
                             runnable, t, n_threads, num_total_tasks);
    }
    for (auto& w : workers) {
        w.join();
    }
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                 const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelSpawn::sync() {
    // You do not need to implement this method.
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

void TaskSystemParallelThreadPoolSpinning::Spinningwork(){
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
}


TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    n_threads = num_threads;
    workers.resize(num_threads);
    // zhe li yao chuang jian xian chegn 
    for(int i = 0; i < num_threads ; i++){
        workers[i] = std::thread(&TaskSystemParallelThreadPoolSpinning::Spinningwork,this);
    }
    // zhe li shi ba ci shi de xian cheng chuang jian hao le , zhi hou zen me ban ? 
    // zhe li chuang jian de func shi shen me ? 
    
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning() {
    stop = true;
    for(int i = 0; i < n_threads; i++){
        workers[i].join();
    }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Part A.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    runnable_ = runnable;
    num_total_tasks_ = num_total_tasks;
    next_task_id = 0;
    complete = 0;
    stop = false;
    epoch.fetch_add(1);
    while(complete.load() < num_total_tasks){}
    seen.fetch_add(1);

}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                              const std::vector<TaskID>& deps) {
    // You do not need to implement this method.
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {
    // You do not need to implement this method.
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
    // ci shi suo you de xian cheng hui pao zhe ge han shu 
    // ru guo ci shi mei you gong zuo ci shi jiu ba xian cheng gua qi lai shi ma ? 
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
}


TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(int num_threads): ITaskSystem(num_threads) {
    //
    // TODO: CS149 student implementations may decide to perform setup
    // operations (such as thread pool construction) here.
    // Implementations are free to add new class member variables
    // (requiring changes to tasksys.h).
    //
    n_threads = num_threads;
    workers.resize(num_threads);
    for(int i = 0; i < n_threads; i++){
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
    for(int i = 0 ; i < n_threads; i++){
        if(workers[i].joinable()) workers[i].join();
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable* runnable, int num_total_tasks) {


    //
    // TODO: CS149 students will modify the implementation of this
    // method in Parts A and B.  The implementation provided below runs all
    // tasks sequentially on the calling thread.
    //
    std::unique_lock<std::mutex> lock(mtx);
    runnable_ = runnable;
    stop = false;
    num_total_tasks_ = num_total_tasks;
    complete = 0;
    next_task_id = 0;
    epoch.fetch_add(1);
    lock.unlock();

    cv.notify_all();
    lock.lock();
    done_cv.wait(lock,[&]{return complete.load() == num_total_tasks_;});
    seen.fetch_add(1);
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(IRunnable* runnable, int num_total_tasks,
                                                    const std::vector<TaskID>& deps) {


    //
    // TODO: CS149 students will implement this method in Part B.
    //

    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {

    //
    // TODO: CS149 students will modify the implementation of this method in Part B.
    //

    return;
}
