#!/usr/bin/env python3
import string
import itertools
import random
import argparse
import os
import heapq
import logging
import pickle
import random
import uuid
import time
import numpy as np
from jericho.util import clean
from jericho import *
from visit_counter import *

OUTPUT_DIR = os.environ.get('AMLT_OUTPUT_DIR', '.')

parser = argparse.ArgumentParser()
parser.add_argument("--rom", type=str, required=True,
                    help="Path to ROM to run")
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed to use")
parser.add_argument("--iters", type=int, default=100,
                    help="Iterations of tree search to run")
parser.add_argument("--load", type=str, default=None,
                    help="Load this save file *.qzl")
parser.add_argument("--debug", action='store_true',
                    help="Enable debug mode")
parser.add_argument("--viz", action='store_true',
                    help="Create a graph of objects after searching")

# Globals
hash2node = {}
visit_counter = VisitCounter()


class Node(object):
    def __init__(self, env, act, obs, score, terminal):
        self.act = act
        self.obs = obs
        self.world_state_hash = hash(env.get_world_state_hash())
        self.terminal = terminal
        self.location = env.get_player_location()
        self.inventory = env.get_inventory()
        self.score = score
        self.progress = max(0, score / env.get_max_score())
        self.save_buff = env.get_state()
        self.uid = uuid.uuid4()        
        self.children = []
        self.actions = []
        self.diffs = []

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.score == other.score and self.world_state_hash == other.world_state_hash
        return False      

    def __str__(self):
        return "Obs: {} Loc: {} ({}) Score: {} Inventory: {} uid: {}"\
            .format(clean(self.obs), self.location.num, self.location.name,
                    self.score, self.inv_str(), self.uid)          

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self.world_state_hash

    def inv_str(self):
        return ", ".join(self.inventory_items())

    def inventory_items(self):
        # Returns a list of items in the inventory
        return [i.name for i in self.inventory]

    def short_str(self):
        return "{} {}\nScore{} {}\n[{}]"\
            .format(self.location.name, self.location.num, self.score, str(self.uid)[:8],
                    self.inv_str(), self.uid)        

    def add_child(self, action, diff, child_node):
        # assert child_node != self
        self.children.append(child_node)
        self.actions.append(action)
        self.diffs.append(diff)

    def get_relative_quality(self, other):
        """ Returns the relative quality of this node in relation to the other. 
            Quality in [0, 1] - 0 indicates other is better, 1 indicates this is better 0.5 indicates a tie.
        """
        assert isinstance(self, other.__class__)
        tau = 10
        score_q = np.exp(tau*self.progress) / (np.exp(tau*self.progress) + np.exp(tau*other.progress))
        return score_q

    def __lt__(self, other):
        # Returning True prioritizes this node over the other
        assert isinstance(self, other.__class__)
        # quality = self.get_relative_quality(other)
        # return random.random() < quality

        # Prefer non-terminal nodes
        # if self.terminal != other.terminal:
        #     return other.terminal
        # Prefer high game scores
        if self.score != other.score:
            return self.score > other.score
        # Prefer less visited locations
        # elif visit_counter.visit_count(self) != visit_counter.visit_count(other):
        #     return visit_counter.visit_count(self) < visit_counter.visit_count(other)            
        # Prefer more items in inventory
        elif len(self.inventory) != len(other.inventory):
            return len(self.inventory) > len(other.inventory)
        else:
            return random.random() < .5

    def is_leaf(self):
        return len(self.children) == 0

    def diff(self, other):
        # Returns a string difference between this game state and another
        if not isinstance(self, other.__class__) or self == other:
            return
        diff = ""
        if self.score != other.score:
            diff += "Score Diff {} vs {}; ".format(self.score, other.score)
        if len(self.world) != len(other.world):
            diff += "Diff number world objs {} vs {}; ".format(len(self.world), len(other.world))
        for o1, o2 in zip(self.world, other.world):
            if o1 != o2:
                diff += "Object Diff:\n{}\n{}; ".format(o1,o2)
        return diff


class PrioritySet(object):
    def __init__(self):
        self.heap = []
        self.set = set()

    def push(self, d):
        heapq.heappush(self.heap, d)
        self.set.add(d)

    def pop(self):
        d = heapq.heappop(self.heap)
        return d

    def __contains__(self, key):
        return key in self.set

    def __len__(self):
        return len(self.heap)


def visualize(fname):
    # Visualize the current tree of nodes, from the given root node
    import pydot
    graph = pydot.Dot(graph_type='digraph')
    node2graph = {}
    for hsh, node in hash2node.items():
        graph_node = pydot.Node(node.short_str())
        if node.terminal:
            graph_node.add_style("filled")
        graph.add_node(graph_node)
        node2graph[node] = graph_node
    for hsh, node in hash2node.items():
        graph_node = node2graph[node]
        for child, act in zip(node.children, node.actions):
            graph_child = node2graph[child]
            graph.add_edge(pydot.Edge(graph_node, graph_child, label=act))
    graph.write_pdf(fname)


def save_file(state, fname):
    with open(fname, 'wb') as f:
        pickle.dump(state.save_buff, f)


def load_file(env, fname):
    with open(fname, 'rb') as f:
        save_buff = pickle.load(f)
    env.set_state(save_buff)


def expand(node, env):
    """ Expands a node by trying many possible actions and adding the
        succuessful ones as children. 
        Returns True if game ends with victory, False otherwise.
    """
    env.set_state(node.save_buff)
    assert not node.terminal, "Cannot expand a terminal node!"
    logging.debug("EXPAND {}".format(node))

    valid_actions = env.get_valid_actions(use_object_tree=True, use_ctypes=True)

    for act in valid_actions:
        env.set_state(node.save_buff)
        obs, rew, done, info = env.step(act)
        if env.victory():
            logging.info(f'\nVICTORY! {act} --> {obs}')
            return True
        wsh = hash(env.get_world_state_hash())
        if wsh in hash2node:
            child_node = hash2node[wsh]
        else:
            child_node = Node(env, act, obs, info['score'], done)
        world_diff = env._get_world_diff()
        node.add_child(act, world_diff, child_node)
        hash2node[wsh] = child_node
    return False


def astar_search(env, args):
    """ Performs an A* Search """
    start = time.time()
    env.reset()
    if args.load:
        load_file(env, args.load)
    act = 'look'
    obs, rew, done, info = env.step(act)
    high_score = info['score']
    root = Node(env, act, obs, high_score, done)
    hash2node[root.world_state_hash] = root
    curr_node = root
    heap = PrioritySet()
    for i in range(1, args.iters+1):
        fps = i / (time.time() - start)
        logging.info("Iter {} Heap {} HighScore {} Locs {} FPS {:.1f} {} --> {} {}".format(
            i, len(heap), high_score, len(visit_counter), fps, curr_node.act, curr_node.location.num, curr_node.location.name))
        victory = expand(curr_node, env)
        possible_actions = curr_node.actions
        visit_counter.record_possible_actions(curr_node, possible_actions)
        for child in curr_node.children:
            if child not in heap and not child.terminal:
                heap.push(child)
        if victory or len(heap) <= 0:
            break
        curr_node = heap.pop()
        visit_counter.visit(curr_node)
        score = curr_node.score
        if score > high_score:
            high_score = score
            fname = os.path.join(OUTPUT_DIR, f"{env.bindings['name']}_{score}.pkl")
            save_file(curr_node, fname)
    visit_counter.log_visit_counts(env.get_world_objects())

    if args.viz:
        visualize(fname=os.path.join(args.exp, 'graph.pdf'))


def setup_logging(log_filename=None, debug=False):
    while logging.root.handlers:
        logging.root.removeHandler(logging.root.handlers[0])
    log_level = logging.DEBUG if debug else logging.INFO
    if log_filename:
        logging.basicConfig(format='%(message)s', filename=log_filename, level=log_level)
    else:
        logging.basicConfig(format='%(message)s', level=log_level)


def main():
    args = parser.parse_args()
    setup_logging(debug=args.debug)
    logging.info("RandomSeed {}".format(args.seed))
    random.seed(args.seed)
    env = FrotzEnv(args.rom, seed=args.seed)
    astar_search(env, args)
    env.close()


if __name__ == "__main__":
    main()
