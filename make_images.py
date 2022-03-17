import math
import os
import shutil
from threading import Thread

import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

image_path = "urban grids/10.png"
append = "10-rew"
save_frames = True
base_reward = -10
antenna_reward = -1
radius = 4
sigma = 3

FREE_CELL = 0
OBSTACLE = 1
START = 2
GOAL = 3
BLACK = [0, 0, 0]
WHITE = [255, 255, 255]
BLUE = [0, 0, 255]
GREEN = [0, 255, 0]
YELLOW = [255, 255, 0]
states = {}
cell_types = {}
shape = {}
n_states = {}
next_states = {}
probs = {}
c = {}

moves = [
    lambda x, y: (x, y) if is_out_of_bounds((x - 1, y - 1)) else (x - 1, y - 1),  # Up-Left
    lambda x, y: (np.clip(x - 1, 0, size[0] - 1), y),  # Up
    lambda x, y: (x, y) if is_out_of_bounds((x - 1, y + 1)) else (x - 1, y + 1),  # Up-Right
    lambda x, y: (x, np.clip(y - 1, 0, size[0] - 1)),  # Left
    lambda x, y: (x, y),  # Still
    lambda x, y: (x, np.clip(y + 1, 0, size[0] - 1)),  # Right
    lambda x, y: (x, y) if is_out_of_bounds((x + 1, y - 1)) else (x + 1, y - 1),  # Down-Left
    lambda x, y: (np.clip(x + 1, 0, size[0] - 1), y),  # Down
    lambda x, y: (x, y) if is_out_of_bounds((x + 1, y + 1)) else (x + 1, y + 1),  # Down-Right
]
n_actions = len(moves)
current_step = 0
gamma = 1.0
theta = 1e-6


def is_out_of_bounds(new_index):
    return np.any(np.array(new_index) > (np.array(states.shape) - 1)) or np.any(np.array(new_index) < 0)


def move(x, y, n_action):
    if cell_types[x, y] == GOAL:
        next_state = 0
        reward = 0
    elif cell_types[x, y] == OBSTACLE:
        next_state = 1
        reward = -100
    else:
        new_index = moves[n_action](x, y)
        next_state = states[new_index] if cell_types[new_index] != OBSTACLE else states[x, y]
        reward = -100 if cell_types[new_index] == OBSTACLE else c[
            x, y]  # if n_action in [1, 3, 5, 7] else math.sqrt(2) * \
        #                   c[x, y]
    return int(next_state), reward


def arrowedLine(im, ptA, ptB, width=1, color=(0, 0, 0)):
    """Draw line from ptA to ptB with arrowhead at ptB"""
    # Get drawing context
    draw = ImageDraw.Draw(im)
    # Draw the line without arrows

    # Now work out the arrowhead
    # = it will be a triangle with one vertex at ptB
    # - it will start at 95% of the length of the line
    # - it will extend 8 pixels either side of the line
    x0, y0 = ptA
    x1, y1 = ptB
    # Now we can work out the x,y coordinates of the bottom of the arrowhead triangle
    xb = 0.8 * (x1 - x0) + x0
    yb = 0.8 * (y1 - y0) + y0
    draw.line([ptA, (xb, yb)], width=width, fill=color)
    # Work out the other two vertices of the triangle
    # Check if line is vertical
    if x0 == x1:
        vtx0 = (xb - 5, yb)
        vtx1 = (xb + 5, yb)
    # Check if line is horizontal
    elif y0 == y1:
        vtx0 = (xb, yb + 5)
        vtx1 = (xb, yb - 5)
    else:
        alpha = math.atan2(y1 - y0, x1 - x0) - 90 * math.pi / 180
        a = 8 * math.cos(alpha)
        b = 8 * math.sin(alpha)
        vtx0 = (xb + a, yb + b)
        vtx1 = (xb - a, yb - b)

    # draw.point((xb,yb), fill=(255,0,0))    # DEBUG: draw point of base in red - comment out draw.polygon() below if using this line
    # im.save('DEBUG-base.png')              # DEBUG: save

    # Now draw the arrowhead triangle
    draw.polygon([vtx0, vtx1, ptB], fill=color)
    return im


def center(x, y):
    return np.array((x + 0.5, y + 0.5))


rel_pos = 0.45
diag_pos = rel_pos * math.sin(math.pi / 4)

points = [
    (-diag_pos, -diag_pos),  # Up-Left
    (0.0, -rel_pos),  # Up
    (diag_pos, -diag_pos),  # Up-Right
    (-rel_pos, 0.0),  # Left
    (0.0, 0.0),  # Still
    (rel_pos, 0.0),  # Right
    (-diag_pos, diag_pos),  # Down-Left
    (0.0, rel_pos),  # Down
    (diag_pos, diag_pos),  # Down-Right
]


def pol2(polic, pixel_size=200, font_color=(0, 0, 0)):
    img = Image.new('RGB', (pixel_size * size[0], pixel_size * size[1]), color=(255, 255, 255))
    it = np.nditer(states, flags=['multi_index'])
    canvas = ImageDraw.Draw(img)
    while not it.finished:
        x, y = it.multi_index
        if cell_types[x, y] == OBSTACLE:
            canvas.rectangle(
                    [((y + 1) * pixel_size, (x + 1) * pixel_size),
                     ((y + 1) * pixel_size - pixel_size, (x + 1) * pixel_size - pixel_size)],
                    fill=font_color)
        elif cell_types[x, y] == GOAL:
            canvas.rectangle(
                    [((y + 1) * pixel_size, (x + 1) * pixel_size),
                     ((y + 1) * pixel_size - pixel_size, (x + 1) * pixel_size - pixel_size)],
                    fill=tuple(BLUE))

        else:
            if cell_types[x, y] == START:
                canvas.rectangle(
                        [((y + 1) * pixel_size, (x + 1) * pixel_size),
                         ((y + 1) * pixel_size - pixel_size, (x + 1) * pixel_size - pixel_size)],
                        fill=tuple(GREEN))
            best_actions = np.argwhere(polic[:, x, y] == np.amax(polic[:, x, y]))
            for a in best_actions.flatten().tolist():
                cnt = center(y, x)
                b = center(y, x) + np.array(points[int(a)])

                img = arrowedLine(img, tuple(cnt * pixel_size), tuple(b * pixel_size), width=7,
                                  color=font_color)
        it.iternext()
    return img


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def values_to_V(val):
    vv = np.zeros(n_states)
    it = np.nditer(states, flags=['multi_index'])
    while not it.finished:
        x, y = it.multi_index
        vv[states[x, y]] = val[x, y]
        it.iternext()
    return vv


def save_pol_val(policy, vv, i, output_name):
    pol2(policy).save(f"{output_folder}/{output_name}/policy/{output_name}_policy_{i:04d}.png")
    fig, ax = plt.subplots(figsize=(16, 16))
    cmap = cm.get_cmap('plasma').copy()
    cmap.set_over("black")
    ax.axis('off')

    im = ax.matshow(vv[states] + (1 - obstacle_map), cmap=cmap, vmax=0.01)
    # for (i, j), z in np.ndenumerate(Vs[i+1][states]):
    #     ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, shrink=0.7, ax=ax).ax.tick_params(labelsize=30)
    plt.savefig(f"{output_folder}/{output_name}/values/{output_name}_values_{i:04d}.png",
                # bbox_inches=50,
                dpi=300,
                transparent=True)
    plt.close()


def imgss(policies, policies2, Vs, output_name):
    space = np.linspace(0, policies.shape[0] - 2, 4, dtype=int)
    for i in range(policies.shape[0] - 1):
        if save_frames:
            vv = np.copy(Vs[:-1][i])
            vv[1] = 0
            save_pol_val(policies2[:-1][i], vv, i, output_name)
            if i in space:
                shutil.copy2(f"{output_folder}/{output_name}/values/{output_name}_values_{i:04d}.png",
                             f"{output_folder}/{output_name}/{output_name}_values_{i:04d}.png")
                shutil.copy2(f"{output_folder}/{output_name}/policy/{output_name}_policy_{i:04d}.png",
                             f"{output_folder}/{output_name}/{output_name}_policy_{i:04d}.png")
        else:
            if i in space:
                vv = np.copy(Vs[:-1][i])
                vv[1] = 0
                save_pol_val(policies2[:-1][i], vv, i, output_name)
    path = np.zeros(obstacle_map.shape)
    current_position = np.transpose(np.nonzero(cell_types == START))[0]
    total_rew = 0
    while True:
        x = current_position[0]
        y = current_position[1]
        state = int(states[x, y])
        if path[x, y] == 1 or state == 0:
            path[x, y] = 1
            break
        path[x, y] = 1
        best_a = int(policies[-1][x, y])
        _, rew = move(x, y, best_a)
        total_rew += rew
        current_position = moves[best_a](x, y)
    print(output_name, total_rew)
    map_with_path = obstacle_map + path
    pixel_size = 100
    img = Image.new('RGB', (pixel_size * size[0], pixel_size * size[1]), color=(255, 255, 255))
    canvas = ImageDraw.Draw(img)
    for x, row in enumerate(map_with_path):
        for y, value in enumerate(row):
            if cell_types[x, y] == GOAL:
                canvas.rectangle(
                        [((y + 1) * pixel_size, (x + 1) * pixel_size),
                         ((y + 1) * pixel_size - pixel_size, (x + 1) * pixel_size - pixel_size)],
                        fill=tuple(BLUE))
            elif cell_types[x, y] == START:
                canvas.rectangle(
                        [((y + 1) * pixel_size, (x + 1) * pixel_size),
                         ((y + 1) * pixel_size - pixel_size, (x + 1) * pixel_size - pixel_size)],
                        fill=tuple(GREEN))
            elif int(value) == 0:
                canvas.rectangle(
                        [((y + 1) * pixel_size, (x + 1) * pixel_size),
                         ((y + 1) * pixel_size - pixel_size, (x + 1) * pixel_size - pixel_size)],
                        fill=tuple(BLACK))
            elif int(value) == 1:
                canvas.rectangle(
                        [((y + 1) * pixel_size, (x + 1) * pixel_size),
                         ((y + 1) * pixel_size - pixel_size, (x + 1) * pixel_size - pixel_size)],
                        fill=tuple(WHITE))
            elif int(value) == 2:
                canvas.rectangle(
                        [((y + 1) * pixel_size, (x + 1) * pixel_size),
                         ((y + 1) * pixel_size - pixel_size, (x + 1) * pixel_size - pixel_size)],
                        fill=(0, 150, 0))
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.axis('off')
    ax.imshow(img)
    img.save(f"{output_folder}/{output_name}/{output_name}.png")


def value_iteration(gamma=0.97, theta=1e-6):
    finished = False

    rewards = np.zeros(next_states.shape)
    iter = np.nditer(states, flags=['multi_index'])
    while not iter.finished:
        x, y = iter.multi_index
        rewards[x, y, :, :] = np.ones((3, 3)) * c[x, y]
        iter.iternext()
    values = np.zeros(obstacle_map.shape)
    Vs = np.zeros((1, n_states))
    policies = np.zeros((1,) + obstacle_map.shape)
    policies2 = np.zeros((1, n_actions,) + obstacle_map.shape)
    V = np.zeros(n_states)
    diffs = np.zeros(1)
    while not finished:
        delta = 0.0
        v = np.copy(values)
        Q = np.sum((rewards + gamma * V[next_states]) * probs, (3, 4))
        values = np.max(Q, 0)
        V = values_to_V(values)
        Vs = np.append(Vs, [V], 0)
        # V[0] = 0
        delta = max(delta, np.max(np.abs(v - values)))
        diffs = np.append(diffs, delta)
        policy = np.argmax(Q, 0)
        policy2 = np.copy(Q)
        if policy.shape == obstacle_map.shape:
            policies = np.append(policies, [policy], axis=0)
            policies2 = np.append(policies2, [policy2], axis=0)
        if delta < theta:
            finished = True
    print(f"vi: {V[states][start == 1]}")
    imgss(policies, policies2, Vs, vi_output_name)
    return diffs


def sump(gamma=0.97, theta=1e-6):
    finished = False

    policies = np.zeros((1,) + obstacle_map.shape)
    policies2 = np.zeros((1, n_actions,) + obstacle_map.shape)
    Vs = np.zeros((1, n_states))
    V = np.zeros(n_states)
    diffs = np.zeros(1)

    while not finished:
        delta = 0.0
        Q = np.log(np.sum(np.exp(gamma * V[next_states] + rewards + probs), (3, 4)))
        values = np.log(np.sum(np.exp(Q), 0))
        prev_values = np.copy(V[states])
        prev_values = np.where(prev_values == -np.inf, 0, prev_values)
        V = values_to_V(values)
        # V[0] = 0
        V = V - V.max()
        V[1] = 0
        Vs = np.append(Vs, [V], 0)

        values = V[states]
        delta = max(delta, np.max(np.abs(prev_values - np.where(values == -np.inf, 0, values))))
        diffs = np.append(diffs, delta)
        policy = np.argmax(Q, 0)
        policy2 = np.copy(Q)
        if policy.shape == obstacle_map.shape:
            policies = np.append(policies, [policy], axis=0)
            policies2 = np.append(policies2, [policy2], axis=0)
        if delta < theta:
            finished = True
    print(f"sp: {V[states][start == 1]}")

    imgss(policies, policies2, Vs, sp_output_name)
    return diffs


def gen_apm():
    A = 1 / 18 * np.ones(8)
    A[[0, 2]] = 1 / 6
    A[1] = 1 / 3
    apm = 1 / 18 * np.ones((9, 9))
    apm[4] = 1 / 16 * np.ones(9)
    apm[4, 4] = 1 / 2
    for i in [0, 1, 2, 5, 8, 7, 6, 3]:
        apm[i, [3, 5]] = A[[0, 4]]
        apm[i, :3] = A[1:4]
        apm[i, 6:] = A[5:8][::-1]
        A = np.roll(A, 1)
    return np.reshape(apm, (9, 3, 3))


aps = gen_apm()


def state_trnstn_distr(x, y, action, obstacle_map, aps):
    obstacles = obstacle_map[x - 1:x + 2, y - 1:y + 2]
    ap = np.copy(aps[action])
    n_obs = 9 - np.count_nonzero(obstacles)
    if n_obs:
        prob = np.sum(ap[np.nonzero(obstacles == 0)])
        ap *= obstacles
        # ap[1, 1] += prob
        ap /= (1 - prob)
    return ap


if __name__ == "__main__":
    output_folder = "output/"
    vi_output_name = f"vi-{append}"
    sp_output_name = f"sp-{append}"
    if not os.path.exists(output_folder + vi_output_name + "/policy/") or \
            not os.path.exists(output_folder + sp_output_name + "/values/"):
        os.makedirs(output_folder + vi_output_name + "/policy/")
        os.makedirs(output_folder + vi_output_name + "/values/")
    if not os.path.exists(output_folder + sp_output_name + "/policy/") or \
            not os.path.exists(output_folder + sp_output_name + "/values/"):
        os.makedirs(output_folder + sp_output_name + "/policy/")
        os.makedirs(output_folder + sp_output_name + "/values/")
    image = Image.open(image_path)
    size = image.size
    data = np.asarray(image)
    is_using_distribution = True

    mu = 0
    from scipy.stats import multivariate_normal

    gauss2d = multivariate_normal([mu, mu], [[sigma ** 2, 0], [0, sigma ** 2]])
    X, Y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    antenna_distribution = gauss2d.pdf(pos)
    antenna_distribution -= antenna_distribution.min()
    antenna_distribution *= abs(base_reward - antenna_reward) / antenna_distribution.max()
    plt.imshow(antenna_distribution)
    cell_types = np.zeros(size, dtype=int)
    c = np.zeros(size, dtype=np.float32)
    antenna_indexes = []
    for x in range(size[0]):
        for y in range(size[1]):
            cell = data[x, y, :]
            if np.array_equal(cell, BLACK):
                cell_types[x, y] = OBSTACLE
                c[x, y] = -100
            elif np.array_equal(cell, WHITE):
                cell_types[x, y] = FREE_CELL
                c[x, y] = base_reward
            elif np.array_equal(cell, GREEN):
                cell_types[x, y] = START
                c[x, y] = base_reward
            elif np.array_equal(cell, BLUE):
                cell_types[x, y] = GOAL
                c[x, y] = 1.0
            elif np.array_equal(cell, YELLOW):
                if is_using_distribution:
                    # cell_types[x, y] = OBSTACLE
                    antenna_indexes.append((x, y))
                    # c[x, y] = -100
                    c[x, y] = base_reward
                else:
                    c[x, y] = antenna_reward
    obstacle_map = np.where(cell_types == OBSTACLE, 0, 1).astype(np.float32)

    start = np.copy(cell_types)
    start = np.where(start != START, 0, start)
    start = np.where(start == START, 1, start).astype(np.float32)

    goal = np.copy(cell_types)
    goal = np.where(goal != GOAL, 0, goal)
    goal = np.where(goal == GOAL, 1, goal).astype(np.float32)

    if len(antenna_indexes) > 0:
        map = np.zeros(c.shape)
        for x, y in antenna_indexes:
            x_min_radius = x if x <= radius else radius
            x_max_radius = radius + 1 if x + radius + 1 < size[0] else size[0] - x
            y_min_radius = y if y <= radius else radius
            y_max_radius = radius + 1 if y + radius + 1 < size[1] else size[1] - y
            distribution = antenna_distribution[radius - x_min_radius:radius + x_max_radius,
                           radius - y_min_radius:radius + y_max_radius]
            mask = obstacle_map[x - x_min_radius:x + x_max_radius, y - y_min_radius:y + y_max_radius]
            c[x - x_min_radius:x + x_max_radius, y - y_min_radius:y + y_max_radius] += mask * distribution
            map[x - x_min_radius:x + x_max_radius, y - y_min_radius:y + y_max_radius] += mask * distribution

    c[goal == 1] = 0
    figure, ax = plt.subplots(figsize=(16, 16))
    cmap = cm.get_cmap('plasma').copy()
    cmap.set_under("black")
    ax.axis('off')
    im = ax.matshow(c, cmap=cmap, vmin=base_reward - 0.2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax).ax.tick_params(labelsize=30)

    plt.savefig(f"{output_folder}/reward-map_{append}.png", bbox_inches='tight', dpi=300, transparent=True)
    states = np.zeros(size, dtype=np.intp)
    cell_types = cell_types.astype(int)
    # states
    it = np.nditer(states, flags=['multi_index'])
    n_states = 1
    while not it.finished:
        x, y = it.multi_index
        if cell_types[x, y] == OBSTACLE:
            states[x, y] = 1
        elif cell_types[x, y] != GOAL:
            n_states += 1
            states[x, y] = n_states
        else:
            states[x, y] = 0

        it.iternext()
    n_states += 1
    next_states = np.zeros((size[0], size[1], 3, 3), dtype=np.intp)
    rewards = np.zeros(next_states.shape)
    probs = np.zeros((n_actions, size[0], size[1], 3, 3))
    iter = np.nditer(states, flags=['multi_index'])
    while not iter.finished:
        x, y = iter.multi_index

        nexts = np.zeros(n_actions)
        rews = np.zeros(n_actions)
        for a in range(n_actions):
            nexts[a], rews[a] = move(x, y, a)

            if x in range(1, size[1] - 1) and y in range(1, size[0] - 1):
                probs[a, x, y, :, :] = state_trnstn_distr(x, y, a, obstacle_map, aps)

        rewards[x, y, :, :] = rews.reshape((3, 3))
        next_states[x, y, :, :] = np.reshape(nexts, (3, 3))
        iter.iternext()
    th1 = ThreadWithReturnValue(target=lambda: sump(gamma=gamma, theta=theta))
    th2 = ThreadWithReturnValue(target=lambda: value_iteration(gamma=gamma, theta=theta))
    th1.start()
    th2.start()

    sump_diff = th1.join()
    value_iteration_diff = th2.join()
    plt.close('all')
    plt.rcParams.update({
        "text.usetex": True})
    figure, axi = plt.subplots(figsize=(16, 16))
    axi.plot(range(1, sump_diff.shape[0]), sump_diff[1:], label='Sum-Product')
    axi.plot(range(1, value_iteration_diff.shape[0]), value_iteration_diff[1:], label='Dynamic Programming')
    axi.legend(fontsize=25)
    axi.set_xlabel("t", fontsize=40)
    axi.set_ylabel("$\Delta$", fontsize=40)
    axi.set_yscale('log')
    axi.xaxis.set_major_locator(MaxNLocator(integer=True))
    axi.grid()
    axi.tick_params(labelsize=30)
    plt.savefig(f"output/deltas_{append}.png", bbox_inches='tight', dpi=300, transparent=True)
    os.system(
            f"ffmpeg -r 10 -y -i {output_folder}/{sp_output_name}/values/{sp_output_name}_values_%04d.png "
            f"-vf \"scale=2000:2000,format=yuv420p\" -codec:v libx264 -b 2M {output_folder}/{sp_output_name}/values/{sp_output_name}_values.mp4")
    os.system(f"ffmpeg -y -i {output_folder}/{sp_output_name}/values/{sp_output_name}_values.mp4"
              f"  -filter:v \"crop=1600:1300:200:350\" {output_folder}/{sp_output_name}/values/{sp_output_name}_values2.mp4")
    os.system(
            f"ffmpeg -r 10 -y -i {output_folder}/{sp_output_name}/policy/{sp_output_name}_policy_%04d.png "
            f"-vf \"scale=2000:2000,format=yuv420p\" -codec:v libx264 -b 2M {output_folder}/{sp_output_name}/policy/{sp_output_name}_policy.mp4")
    os.system(
            f"ffmpeg -r 20 -y -i {output_folder}/{vi_output_name}/values/{vi_output_name}_values_%04d.png "
            f"-vf \"scale=2000:2000,format=yuv420p\" -codec:v libx264 -b 2M {output_folder}/{vi_output_name}/values/{vi_output_name}_values.mp4")
    os.system(f"ffmpeg -y -i {output_folder}/{vi_output_name}/values/{vi_output_name}_values.mp4"
              f"  -filter:v \"crop=1600:1300:200:350\" {output_folder}/{vi_output_name}/values/{vi_output_name}_values2.mp4")
    os.system(
            f"ffmpeg -r 20 -y -i {output_folder}/{vi_output_name}/policy/{vi_output_name}_policy_%04d.png "
            f"-vf \"scale=2000:2000,format=yuv420p\" -codec:v libx264 -b 2M {output_folder}/{vi_output_name}/policy/{vi_output_name}_policy.mp4")
