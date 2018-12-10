#ifndef GLOBAL_H
#define GLOBAL_H

#include <bits/stdc++.h>

static float min_weight = 0.000001;

const float qnan = std::numeric_limits<float>::quiet_NaN();
float dist_func(const std::vector<float>& A, const std::vector<float>& B);

std::vector<std::vector<size_t> > rank_matrix_gpu(std::vector<std::vector<float> >& distance_matrix, std::vector<size_t> which_pred);

#endif