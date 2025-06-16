#include <math.h>
#include <algorithm>
#include "loc.h"

const double PI = 3.14159265358979323846;
const double* SCORE_WEIGHT = new double[4]{0.2, 1, 0.1, 0.4};

double* get_parents_score(Points& points) {
    // num of points
    int number = points.nums;

    // get col 2 to 4 of points info
    double** data_subset = new double*[number];
    for (int i = 0; i < number; ++i) {
        data_subset[i] = new double[3];
        data_subset[i][0] = points.data[i][2];
        data_subset[i][1] = points.data[i][3];
        data_subset[i][2] = points.data[i][4];
    }

    double** data_rr = new double*[number];
    double** data_ra = new double*[number];
    int rr_l = 0, ra_l = 0;

    // get data[data[:, 0] == 0] and data[data[:, 0] == 1]
    // update rr_l and ra_l
    for (int i = 0; i < number; ++i) {
        if (data_subset[i][0] == 0) {
            data_rr[rr_l++] = data_subset[i];
        } else if (data_subset[i][0] == 1) {
            data_ra[ra_l++] = data_subset[i];
        }
    }

    double max_r = 0;
    if (rr_l > 0 && ra_l > 0) {
        double max_rr = 0, max_ra = 0;
        for (int i = 0; i < rr_l; ++i) {
            max_rr = std::max(max_rr, std::max(data_rr[i][1], data_rr[i][2]));
        }
        for (int i = 0; i < ra_l; ++i) {
            max_ra = std::max(max_ra, data_ra[i][1]);
        }
        max_r = std::max(max_rr, max_ra);
    } else if (rr_l > 0) {
        for (int i = 0; i < rr_l; ++i) {
            max_r = std::max(max_r, std::max(data_rr[i][1], data_rr[i][2]));
        }
    } else if (ra_l > 0) {
        for (int i = 0; i < ra_l; ++i) {
            max_r = std::max(max_r, data_ra[i][1]);
        }
    }

    double max_a = PI / 3;

    double* score = new double[number];
    for (int i = 0; i < number; ++i) {
        if (data_subset[i][0] == 0) {
            score[i] = (1 - (data_subset[i][1] / (2 * max_r))) * 0.5 + (1 - (data_subset[i][2] / (2 * max_r))) * 0.5;
        } else if (data_subset[i][0] == 1) {
            score[i] = (1 - (data_subset[i][1] / (2 * max_r))) * 0.5 + (1 - (data_subset[i][2] / max_a)) * 0.5;
        } else if (data_subset[i][0] == 2) {
            score[i] = (1 - (data_subset[i][1] / max_a)) * 0.5 + (1 - (data_subset[i][2] / max_a)) * 0.5;
        }
    }

    // Clean up
    for (int i = 0; i < number; ++i) {
        delete[] data_subset[i];
    }
    delete[] data_subset;
    delete[] data_rr;
    delete[] data_ra;

    return score;
}

double* get_density_score(Points& points) {
    int number = points.nums;
    int K = std::min(int(std::sqrt(number * 2)), number - 1);

    // get points.x and points.y
    double** points_loc = new double*[number];
    for (int i = 0; i < number; ++i) {
        points_loc[i] = new double[2];
        points_loc[i][0] = points.data[i][0];
        points_loc[i][1] = points.data[i][1];
    }

    // brute search
    double* dis_all_points = new double[number];
    for (int i = 0; i < number; ++i) {
        double* distances = new double[number];
        int idx = 0;
        for (int j = 0; j < number; ++j) {
            distances[idx++] = std::sqrt(
                std::pow(points_loc[i][0] - points_loc[j][0], 2) + 
                std::pow(points_loc[i][1] - points_loc[j][1], 2));
        }
        // ascending sort
        std::sort(distances, distances + number);
        dis_all_points[i] = 0;
        // sum of dis from first K closest points
        for (int k = 0; k < K + 1; ++k) {
            dis_all_points[i] += distances[k];
        }
        delete[] distances;

        std::cout<<dis_all_points[i]<<std::endl;
    }

    double dis_all_points_sum = 0;
    for (int i = 0; i < number; ++i) {
        dis_all_points_sum += dis_all_points[i];
    }

    // cal score
    double* score = new double[number];
    for (int i = 0; i < number; ++i) {
        score[i] = 1 - dis_all_points[i] / (dis_all_points_sum + 1e-6);
    }

    double mark_alpha = 0.9;
    double* weight = new double[number];
    for (int i = 0; i < number; ++i) {
        weight[i] = std::pow(mark_alpha, points.data[i][2]);
    }

    double* result = new double[number];
    for (int i = 0; i < number; ++i) {
        result[i] = score[i] * weight[i];
    }

    delete[] points_loc;
    delete[] dis_all_points;
    delete[] score;
    delete[] weight;

    return result;
}

double* get_history_score(Points& data, double* history_location) {
    // num of points
    int number = data.nums;

    // no history loc
    if (history_location == nullptr) {
        double* score = new double[number];
        for (int i = 0; i < number; ++i) {
            score[i] = 0;
        }
        return score;
    }

    // get loc(x, y), equal history_location[:2]
    double** points = new double*[number];
    for (int i = 0; i < number; ++i) {
        points[i] = new double[2];
        points[i][0] = data.data[i][0];
        points[i][1] = data.data[i][1];
    }

    // cal dis of current loc and history loc
    double* dis = new double[number];
    for (int i = 0; i < number; ++i) {
        double dx = points[i][0] - history_location[0];
        double dy = points[i][1] - history_location[1];
        // L2 Norm
        dis[i] = std::sqrt(dx * dx + dy * dy);
    }

    // get max(dis)
    double max_dis = 0;
    for (int i = 0; i < number; ++i) {
        if (dis[i] > max_dis) {
            max_dis = dis[i];
        }
    }

    // normalization
    double* score = new double[number];
    for (int i = 0; i < number; ++i) {
        score[i] = 1 - dis[i] / (max_dis + 1e-6);
        if (score[i] > 0.75) {
            score[i] = 0.75;
        }
    }

    // Clean up
    for (int i = 0; i < number; ++i) {
        delete[] points[i];
    }
    delete[] points;
    delete[] dis;

    return score;
}

double* get_time_score(Points& points, double delta_alpha) {
    // num of points
    int number = points.nums;

    // get col 5 to 6 of points info, equal data[:, 5:7]
    double** data_subset = new double*[number];
    for (int i = 0; i < number; ++i) {
        data_subset[i] = new double[2];
        data_subset[i][0] = points.data[i][5];
        data_subset[i][1] = points.data[i][6];
    }

    double* delta_time_score = new double[number];
    for (int i = 0; i < number; ++i) {
        // 1- np.square(data[:,0] - data[:,1])/100
        delta_time_score[i] = 1 - std::pow(data_subset[i][0] - data_subset[i][1], 2) / 100;
        if (delta_time_score[i] < 0) {
            delta_time_score[i] = 0;
        }
    }

    // (data >= 3) & (data < 60)
    double** result = new double*[number
];
    for (int i = 0; i < number; ++i) {
        result[i] = new double[2];
        for (int j = 0; j < 2; ++j) {
            if (data_subset[i][j] < 3) {
                result[i][j] = 1;
            } else if (data_subset[i][j] >= 60) {
                result[i][j] = -1;
            } else {
                result[i][j] = -(data_subset[i][j] / 30) + 1;
            }
        }
    }

    double* score = new double[number];
    for (int i = 0; i < number; ++i) {
        score[i] = (result[i][0] + result[i][1]) / 2;
    }

    double* score2 = new double[number];
    for (int i = 0; i < number; ++i) {
        score2[i] = score[i] * (1 - delta_alpha) + delta_time_score[i] * delta_alpha;
    }

    // Clean up
    for (int i = 0; i < number; ++i) {
        delete[] data_subset[i];
        delete[] result[i];
    }
    delete[] data_subset;
    delete[] result;
    delete[] delta_time_score;
    delete[] score;

    return score2;
}

double* get_score(Points& points, double* history_loc) {

    double* parents_score = get_parents_score(points);
    double* density_score = get_density_score(points);
    double* time_score = get_time_score(points);
    double* history_score = get_history_score(points, history_loc);

    double* score = new double[points.nums];
    for (int i = 0; i < points.nums; ++i ) {
        score[i] = parents_score[i] * SCORE_WEIGHT[0] + density_score[i] * SCORE_WEIGHT[1] + history_score[i] * SCORE_WEIGHT[2] + time_score[i] * SCORE_WEIGHT[3];
        score[i] /= SCORE_WEIGHT[0] + SCORE_WEIGHT[1] + SCORE_WEIGHT[2] + SCORE_WEIGHT[3];
    }

    for (int i = 0; i < points.nums; ++i) {
        std::cout<<density_score[i]<<",";
    }
    std::cout<<std::endl;

    delete[] parents_score;
    delete[] time_score;
    delete[] history_score;

    return score;
}
