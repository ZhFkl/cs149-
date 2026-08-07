#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>

#include "CycleTimer.h"

using namespace std;

typedef struct {
  // Control work assignments
  int start, end;      // 质心范围（computeCost 仍在使用）
  int startM, endM;    // 数据点范围（computeAssignments 多线程划分用）

  // Shared by all functions
  double *data;
  double *clusterCentroids;
  int *clusterAssignments;
  double *currCost;
  int M, N, K;
} WorkerArgs;


/**
 * Checks if the algorithm has converged.
 * 
 * @param prevCost Pointer to the K dimensional array containing cluster costs 
 *    from the previous iteration.
 * @param currCost Pointer to the K dimensional array containing cluster costs 
 *    from the current iteration.
 * @param epsilon Predefined hyperparameter which is used to determine when
 *    the algorithm has converged.
 * @param K The number of clusters.
 * 
 * NOTE: DO NOT MODIFY THIS FUNCTION!!!
 */
static bool stoppingConditionMet(double *prevCost, double *currCost,
                                 double epsilon, int K) {
  for (int k = 0; k < K; k++) {
    if (abs(prevCost[k] - currCost[k]) > epsilon)
      return false;
  }
  return true;
}

/**
 * Computes L2 distance between two points of dimension nDim.
 * 
 * @param x Pointer to the beginning of the array representing the first
 *     data point.
 * @param y Poitner to the beginning of the array representing the second
 *     data point.
 * @param nDim The dimensionality (number of elements) in each data point
 *     (must be the same for x and y).
 */
double dist(double *x, double *y, int nDim) {
  double accum = 0.0;
  for (int i = 0; i < nDim; i++) {
    accum += pow((x[i] - y[i]), 2);
  }
  return sqrt(accum);
}

/**
 * Assigns each data point to its "closest" cluster centroid.
 */
void computeAssignments(WorkerArgs *const args) {
  // 循环交换为"点在外、质心在内"：
  //  - 800MB 数据每轮只扫一趟（原来每个质心扫一趟，共 K 趟）
  //  - minDist 从堆数组变成局部变量（寄存器）
  //  - 每个点的分配互相独立，线程各管 [startM, endM) 一段，无需同步
  // 注意：k 按从小到大遍历且用严格 <，并列时取小编号质心，与原实现一致
  for (int m = args->startM; m < args->endM; m++) {
    double minDist = 1e30;
    int best = -1;
    for (int k = args->start; k < args->end; k++) {
      double d = dist(&args->data[m * args->N],
                      &args->clusterCentroids[k * args->N], args->N);
      if (d < minDist) {
        minDist = d;
        best = k;
      }
    }
    args->clusterAssignments[m] = best;
  }
}

/**
 * Given the cluster assignments, computes the new centroid locations for
 * each cluster.
 */
void computeCentroids(WorkerArgs *const args) {
  int *counts = new int[args->K];

  // Zero things out
  for (int k = 0; k < args->K; k++) {
    counts[k] = 0;
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] = 0.0;
    }
  }


  // Sum up contributions from assigned examples
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] +=
          args->data[m * args->N + n];
    }
    counts[k]++;
  }

  // Compute means
  for (int k = 0; k < args->K; k++) {
    counts[k] = max(counts[k], 1); // prevent divide by 0
    for (int n = 0; n < args->N; n++) {
      args->clusterCentroids[k * args->N + n] /= counts[k];
    }
  }
  delete[] counts;
}

/**
 * Computes the per-cluster cost. Used to check if the algorithm has converged.
 */
void computeCost(WorkerArgs *const args) {
  double *accum = new double[args->K];
  // Zero things out
  for (int k = 0; k < args->K; k++) {
    accum[k] = 0.0;
  }

  // Sum cost for all data points assigned to centroid
  for (int m = 0; m < args->M; m++) {
    int k = args->clusterAssignments[m];
    accum[k] += dist(&args->data[m * args->N],
                     &args->clusterCentroids[k * args->N], args->N);
  }

  // Update costs
  for (int k = args->start; k < args->end; k++) {
    args->currCost[k] = accum[k];
  }
  delete[] accum;
}

/**
 * Computes the K-Means algorithm, using std::thread to parallelize the work.
 *
 * @param data Pointer to an array of length M*N representing the M different N 
 *     dimensional data points clustered. The data is layed out in a "data point
 *     major" format, so that data[i*N] is the start of the i'th data point in 
 *     the array. The N values of the i'th datapoint are the N values in the 
 *     range data[i*N] to data[(i+1) * N].
 * @param clusterCentroids Pointer to an array of length K*N representing the K 
 *     different N dimensional cluster centroids. The data is laid out in
 *     the same way as explained above for data.
 * @param clusterAssignments Pointer to an array of length M representing the
 *     cluster assignments of each data point, where clusterAssignments[i] = j
 *     indicates that data point i is closest to cluster centroid j.
 * @param M The number of data points to cluster.
 * @param N The dimensionality of the data points.
 * @param K The number of cluster centroids.
 * @param epsilon The algorithm is said to have converged when
 *     |currCost[i] - prevCost[i]| < epsilon for all i where i = 0, 1, ..., K-1
 */
void kMeansThread(double *data, double *clusterCentroids, int *clusterAssignments,
               int M, int N, int K, double epsilon) {

  // Used to track convergence
  double *prevCost = new double[K];
  double *currCost = new double[K];

  // The WorkerArgs array is used to pass inputs to and return output from
  // functions.
  WorkerArgs args;
  args.data = data;
  args.clusterCentroids = clusterCentroids;
  args.clusterAssignments = clusterAssignments;
  args.currCost = currCost;
  args.M = M;
  args.N = N;
  args.K = K;

  // Initialize arrays to track cost
  for (int k = 0; k < K; k++) {
    prevCost[k] = 1e30;
    currCost[k] = 0.0;
  }

  /* Main K-Means Algorithm Loop */
  int iter = 0;
  // 临时计时：统计三个函数各自的总耗时（跨迭代累加）
  double tAssign = 0.0, tCentroids = 0.0, tCost = 0.0;
  while (!stoppingConditionMet(prevCost, currCost, epsilon, K)) {
    // Update cost arrays (for checking convergence criteria)
    for (int k = 0; k < K; k++) {
      prevCost[k] = currCost[k];
    }

    // Setup args struct：start/end 维持质心范围（computeCost 使用）
    args.start = 0;
    args.end = K;

    // 多线程并行 computeAssignments：按点连续分块（各点独立，无需同步）
    const int T = 2;
    std::thread workers[T];
    WorkerArgs targs[T];
    int per = (M + T - 1) / T;
    for (int t = 0; t < T; t++) {
      targs[t] = args;
      targs[t].startM = t * per;
      targs[t].endM = std::min(M, (t + 1) * per);
    }
    for(int t = 1; t < T; t++){
      workers[t] = std::thread(computeAssignments,&targs[t]);
    }
    double t0 = CycleTimer::currentSeconds();
    computeAssignments(&targs[0]);
    for(int t = 1; t < T; t++)
      workers[t].join();
    double t1 = CycleTimer::currentSeconds();
    computeCentroids(&args);
    double t2 = CycleTimer::currentSeconds();
    computeCost(&args);
    double t3 = CycleTimer::currentSeconds();
    tAssign += t1 - t0;
    tCentroids += t2 - t1;
    tCost += t3 - t2;

    iter++;
  }

  double tSum = tAssign + tCentroids + tCost;
  printf("[timing] 迭代 %d 轮 | assignments %.1f ms (%.1f%%) | centroids %.1f ms (%.1f%%) | cost %.1f ms (%.1f%%) | 合计 %.1f ms\n",
         iter,
         tAssign * 1000, tAssign / tSum * 100,
         tCentroids * 1000, tCentroids / tSum * 100,
         tCost * 1000, tCost / tSum * 100,
         tSum * 1000);

  delete[] currCost;
  delete[] prevCost;
}
