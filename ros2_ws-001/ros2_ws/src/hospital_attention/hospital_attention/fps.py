import numpy as np

def farthest_point_sampling(points, n_samples=1024):

    N = points.shape[0]

    sampled = np.zeros((n_samples, 3))

    first_idx = np.random.randint(N)

    sampled[0] = points[first_idx]

    distances = np.linalg.norm(
        points - sampled[0],
        axis=1
    )

    for i in range(1, n_samples):

        farthest_idx = np.argmax(distances)

        sampled[i] = points[farthest_idx]

        new_distances = np.linalg.norm(
            points - sampled[i],
            axis=1
        )

        distances = np.minimum(
            distances,
            new_distances
        )

    return sampled
