import numpy as np


def get_history_score(data, history_location=None):

    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if history_location is None:
        return np.zeros(data.shape[0],)
    else:
        history_location = history_location[:2]
        if not isinstance(history_location, np.ndarray):
            history_location = np.array(history_location)

    points = data[:,:2]

    dis = points - history_location
    dis = np.linalg.norm(dis,axis=1)
    # print(f"dis:{dis}")
    score = 1 - dis / (np.max(dis)+1e-6)
    score[score > 0.75] = 0.75

    return score

if __name__ == '__main__':

    z = [[0, 0, 2, 4.71, 3.14, 0, 0],
        [0, -1, 1, 1, 4.71, 0, 0],
        [-1, 0, 1, 1, 3.14, 0, 0]]

    y=get_history_score(data=z,history_location=[1,1])
    print(y)