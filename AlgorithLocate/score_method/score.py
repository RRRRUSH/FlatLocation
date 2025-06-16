from configs import score_weight
from score_method.angle_radio import get_parents_score
from score_method.density import get_density_score
from score_method.history import get_history_score
from score_method.time import get_time_score

def get_score(data, history_location):

    ra_score = get_parents_score(data)
    density_score = get_density_score(data)
    history_score = get_history_score(data, history_location=history_location)
    time_score = get_time_score(data)
    print(f"ra_score: {ra_score}")
    print(f"density_score: {density_score}")
    print(f"history_score: {history_score}")
    print(f"time_score: {time_score}")
    score = ra_score * score_weight[0] + density_score * score_weight[1] + history_score * score_weight[2] + time_score * score_weight[3]
    return score / sum(score_weight)

if __name__ == '__main__':

    z = [
[43.4997,14.278,1,20,0.191986,17,16,],
[5.66148,21.633,1,20,0.191986,17,16,],
[6.7087,16.6335,0,20,23,17,13,],
[42.969,33.1155,0,20,23,17,13,],
[52.5234,12.524,1,23,0.191986,13,16,],
[7.64306,21.2478,1,23,0.191986,13,16,],
    ]

    y = get_score(data=z, history_location=[5.66148,21.633])
    print(y)