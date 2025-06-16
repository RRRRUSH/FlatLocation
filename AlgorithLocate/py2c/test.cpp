#include "loc.h"

void test_scores() {
    int nums = 4;
    int cols = 7;
    Points points(nums, cols);

    // no points no scores
    double test[nums][cols] =  {
        {-2.18036,0.657144,0,35,2,18,27},
        {10.274,43.5806,0,35,24,18,28},
        {-1.92936,-0.795384,0,35,24,18,28},
        {0.23561,-0.520488,0,2,24,27,28},
    };

    // Initial data
    for(int i = 0; i < nums; ++i){
        for(int j = 0; j < 7; ++j){
            points.data[i][j] = test[i][j];
        }
    }
    double* history_loc = new double[2]{-1.92936,-0.795384};

    double* score = get_score(points, history_loc);

    for (int i = 0; i < nums; ++i) {
        std::cout<<score[i]<<" ";
    }
}

void test_points() {
    int num_1 = 3;
    double test_c[num_1][6] = {
        {25.6,23.2,40,-1,5,-1},
        {0,1.6,1,-1,6,-1},
        {-2.4,23.2,-1,6.0912,27,0.191986},
        {29.6,14.4,23,-1,23,-1},
    };
    // 动态分配多维数组
    double** c = new double*[num_1];
    // 将一维数组的元素赋值给多维数组
    for (int i = 0; i < num_1; i++) {
        c[i] = new double[6];
        for (int j = 0; j < 6; j++) {
            c[i][j] = test_c[i][j];
        }
    }

    node* res = get_points(c, num_1);
    while(res != nullptr){
        for(int i = 0; i < 7; i++)
            std::cout<< res->info[i] << " ";
        std::cout << std::endl;
        res = res->next;
    }
}

int main() {

    // test_points();
    test_scores();
    return 0;
}
