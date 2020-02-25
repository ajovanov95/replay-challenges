import networkx as nx
import numpy as np
import numba

from dataclasses import dataclass

import tqdm

from typing import Tuple, List

from itertools import groupby
from functools import reduce


@dataclass
class World:
    n_customer_hqs: int
    max_replay_offices: int
    map_size: Tuple[int, int]

    customer_hqs: List[Tuple[int, int, int]]

    terrain: nx.DiGraph

def get_penalty(terrain_str, i, j):
    s = terrain_str[i, j]

    if s == '#':
        return np.inf
    elif s == '~':
        return 800
    elif s == '*':
        return 200
    elif s == '+':
        return 150
    elif s == 'X':
        return 120
    elif s == '_':
        return 100
    elif s == 'H':
        return 70
    elif s == 'T':
        return 50

def read_input(name):
    with open(name + '.txt', 'r') as f:
        N, M, C, R = f.readline().strip('\n').split(' ')
        N, M, C, R = list(map(int, [N, M, C, R]))

        customer_hqs = []

        for c in range(C):
            x, y, r = f.readline().strip('\n').split(' ')
            x, y, r = list(map(int, [x, y, r]))

            customer_hqs.append((x, y, r))

        # Read terrain
        terrain = nx.DiGraph()

        terrain_str = f.read() 
        terrain_str = np.array(list(terrain_str))
        terrain_str = terrain_str[terrain_str != '\n']
        terrain_str = terrain_str.reshape((N, M))

        # Cost based on type of terrain type of next cell
        neighbours = [(+1, 0, 'R'), (-1, 0, 'L'), (0, -1, 'U'), (0, +1, 'D')]

        hqs_coords = [hq[:2] for hq in customer_hqs]

        # i = x = columns
        for i in range(N):
            # j = y = rows
            for j in range(M):
                if terrain_str[i, j] == '#' and (i, j) not in customer_hqs:
                    continue

                for ni, nj, direction in neighbours:
                    if i + ni >= N or i + ni < 0:
                        continue

                    if j + nj >= M or j + nj < 0:
                        continue

                    # was i + ni, j + nj
                    penalty = get_penalty(terrain_str, i, j)

                    if np.isinf(penalty) and (i + ni, j + nj) in hqs_coords and terrain_str[i, j] != '#':
                        terrain.add_edge((i, j), (i + ni, j + nj), 
                        penalty=0, direction=direction)

                    if not np.isinf(penalty):               
                        terrain.add_edge((i, j), (i + ni, j + nj), 
                                        penalty=penalty, 
                                        direction=direction)

        nx.set_node_attributes(terrain, {
            (i, j): terrain_str[i, j] for (i, j) in terrain.nodes()
        }, 'terrain_type')

        # assert np.sum([terrain_str[i, j] == '#' and (i, j) not in hqs_coords
        #                for (i, j) in terrain.nodes()]) == 0

        print([terrain_str[i, j] for (i, j, _) in customer_hqs])

        return World(C, R, (N, M), customer_hqs, terrain)

@dataclass
class Solution:
    offices: List[Tuple[Tuple[int, int], str]]

def find_all_valid_sources(world: World, 
                           hq: Tuple[int, int], 
                           hq_reward: int) -> List[Tuple[Tuple[int, int], int]]:
    penalties = { node: 1e18
                  for node in world.terrain.nodes() }
    penalties[hq] = 0

    # Go backwards from HQ node and find score from all nodes to it
    nodes_to_visit = [hq]

    visited_nodes = {node: False for node in world.terrain.nodes()}

    while len(nodes_to_visit) > 0:
        node = nodes_to_visit.pop()

        for (pred, _, dd) in world.terrain.in_edges([node], data=True):            
            penalty = dd['penalty']

            penalties[pred] = min(penalties[pred], penalties[node] + penalty)

            if not visited_nodes[pred]:
                nodes_to_visit.append(pred)
        
        visited_nodes[node] = True

    scores = list(penalties.items())
    scores = [(sc[0], hq_reward - sc[1]) 
              for sc in scores 
              if sc[0] != hq and hq_reward - sc[1] > 0]#-1e12]

    return scores


def solve(world: World):
    directions = nx.get_edge_attributes(world.terrain, 'direction')
    penalties  = nx.get_edge_attributes(world.terrain, 'penalty')

    wanted_targets = [hq[:2] for hq in world.customer_hqs]
    rewards = {hq[:2]: hq[2] for hq in world.customer_hqs}

    # Part 1 of algorithm - Find all valid paths to all HQs
    paths = []

    for hq in tqdm.tqdm(wanted_targets):
        local_paths = find_all_valid_sources(world, hq, rewards[hq])

        paths.extend([(lp[0], hq, lp[1]) 
                      for lp in local_paths])

    # Part 2 of algorithm - Group by destination, find source with maximal reward - penalty
    choices = paths

    # for dst, grp in tqdm.tqdm(groupby(paths, lambda x: x[1]), total=world.n_customer_hqs):
    #     grp = list(grp)
       
    #     c = sorted(grp, key=lambda x: x[2])[0]

    #     choices.append(c)

    # print(choices)

    # Part 3 of algorithm - Group by source, find all HQs reachable by a source
    # and find total score
    final_choices = []

    for src, grp in groupby(choices, lambda x: x[0]):
        grp = list(grp)

        grp_score = sum([e[2] for e in grp])

        hqs_covered = set([e[1] for e in grp])

        n_hqs_covered = len(hqs_covered)

        # print(src, grp_score, hqs_covered, n_hqs_covered)

        final_choices.append((src, grp_score, n_hqs_covered, hqs_covered))

    # print(final_choices)
    # Part 4 of algorithm - Greedily build offices to maximize HQ coverage
    # TODO: FIXME: How to deal with overlap of HQs covered here
    # Overlap damages score as more than 1 HQ coverage does not count for score
    # Order by number of HQs covered in descending order
    final_choices = sorted(final_choices, key=lambda x: x[2], reverse=True)

    # We are bound by a fixed number of offices we can build.
    hqs_covered = { hq: False for hq in wanted_targets}

    grand_total_score = 0

    offices_built = set()

    output = []

    tt = nx.get_node_attributes(world.terrain, 'terrain_type')

    for (src, grp_score, n_hqs_covered, grp_hqs_covered) in tqdm.tqdm(final_choices):
        for dst in grp_hqs_covered:
            # If not covered, cover by this source
            if not hqs_covered[dst]:
                # If not possible to build a new office or reuse an old one
                if (len(offices_built) == world.max_replay_offices - 1 and src not in offices_built):
                    continue 

                if tt[dst] == '#':
                    continue
                
                # We can't build at a location of HQ
                if src in wanted_targets:
                    continue

                hqs_covered[dst] = True

                sp = nx.dijkstra_path(world.terrain, src, dst, weight='penalty')

                # If a path passes through HQ it is not valid
                if any(map(lambda s: s in wanted_targets, sp[:-1])):
                    continue

                desc = ''.join([directions[node1, node2] 
                               for node1, node2 in zip(sp, sp[1:])])

                # Total penalty to arrive to customer HQ
                total_penalty = np.sum([penalties[node1, node2] 
                                        for node1, node2 in zip(sp, sp[1:])])

                # Reward for arriving
                reward = rewards[dst]

                # Score is reward - penalty
                total_score = reward - total_penalty

                print(src, dst, desc, total_score, ''.join([tt[node] for node in sp]))

                grand_total_score += total_score

                output.append((src, desc))

                offices_built = offices_built.union({src})

    print('Grand total score', grand_total_score)
    print('HQs coverage percent ', np.sum(list(hqs_covered.values())) / world.n_customer_hqs)
    print(f'HQs coverage {np.sum(list(hqs_covered.values()))} out of {world.n_customer_hqs}')
    print(f'Built {len(offices_built)} out of maximum of {world.max_replay_offices}')

    return Solution(output)

def trace_path(world: World, src: Tuple[int, int], desc: str):
    neighbours = [(+1, 0, 'R'), (-1, 0, 'L'), (0, -1, 'U'), (0, +1, 'D')]

    neighbours = { dir_: np.array([dx, dy]) for dx, dy, dir_ in neighbours }

    dirs = np.cumsum([neighbours[s] for s in desc], axis=0)

    return [src] + [(src[0] + s[0], src[1] + s[1]) for s in dirs]

def validate_solution(world: World, solution: Solution):
    hqs_coords = [hq[:2] for hq in world.customer_hqs]

    terrain_type = nx.get_node_attributes(world.terrain, 'terrain_type')

    paths = list(map(lambda o: trace_path(world, o[0], o[1]), solution.offices))

    # Check that all paths end in a HQ
    assert all(map(lambda p: p[-1] in hqs_coords, paths)), 'Not all paths end in HQ'

    # Check that no path crosses a mountain (except for last step if HQ is on mountain)
    assert all(map(lambda p: all(map(lambda s: terrain_type[s] != '#', p[:-1])), paths)), 'Path crosses a mountain'

    # Check the number of offices built is less than max available
    assert len(set([o[0] for o in solution.offices])) < world.max_replay_offices, 'Too many offices built'

    # Check that no paths starts in a HQ
    assert all(map(lambda p: p[0] not in hqs_coords, paths)), 'Building offices in HQ cells is not allowed'

    # Check that no path crosses a HQ?
    assert all(map(lambda p: all(map(lambda s: s not in hqs_coords, p[:-1])), paths))


def write_solution(name, solution: Solution):
    with open(name + '.out', 'w') as f:
        for (x, y), desc in solution.offices:
            f.write(f'{x} {y} {desc}\n')

import os
os.chdir('2019')

name = '1_victoria_lake'

names = ['1_victoria_lake', '4_manhattan', '3_budapest', '2_himalayas']#, '5_oceania']

for name in names:
    print(name)

    world = read_input(name)

    solution = solve(world)

    # print(solution)

    validate_solution(world, solution)

    write_solution(name, solution)
