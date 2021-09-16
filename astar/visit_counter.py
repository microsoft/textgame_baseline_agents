import logging

class VisitCounter(object):
    """ Keeps track of the visits to locations. """
    def __init__(self):
        self.visit_dict = {}
        self.taken_act_dict = {}
        self.poss_act_dict = {}
        self.act_cnt = {}

    def __len__(self):
        return len(self.visit_dict)

    def visit_count(self, node):
        loc = node.location.num
        return self.visit_dict[loc] if loc in self.visit_dict else 0

    def visit(self, node):
        loc = node.location.num
        if loc not in self.visit_dict:
            self.visit_dict[loc] = 1
        else:
            self.visit_dict[loc] += 1
        if node.act not in self.act_cnt:
            self.act_cnt[node.act] = 1
        else:
            self.act_cnt[node.act] += 1

    def record_taken_action(self, node, taken_action):
        # Record the taken action and possible actions from a node
        loc = node.location.num
        if loc in self.taken_act_dict:
            self.taken_act_dict[loc].add(taken_action)
        else:
            self.taken_act_dict[loc] = set([taken_action])

    def record_possible_actions(self, node, possible_actions):
        loc = node.location.num
        for act in possible_actions:
            if loc in self.poss_act_dict:
                self.poss_act_dict[loc].add(act)
            else:
                self.poss_act_dict[loc] = set([act])

    def log_visit_counts(self, objs):
        logging.info("\n=======================\n     Visit Counts\n=======================\n")
        s = [(k, self.visit_dict[k]) for k in sorted(
            self.visit_dict, key=self.visit_dict.get, reverse=True)]
        for obj in objs:
            if not obj:
                continue
            loc_num = obj.num
            if loc_num not in self.visit_dict:
                continue
            visit_cnt = self.visit_dict[loc_num]
            loc_name = obj.name
            taken_acts = self.taken_act_dict[loc_num] if loc_num in self.taken_act_dict else set()
            poss_acts = self.poss_act_dict[loc_num] if loc_num in self.poss_act_dict else set()
            unexplored_acts = poss_acts.difference(taken_acts)
            logging.info("Loc {} {} Visits: {}\nTakenActions: {}\nUnexploredActions: {}\n"\
                         .format(loc_num, loc_name, visit_cnt, taken_acts, unexplored_acts))
        logging.info("\n=======================\n     Action Counts\n=======================\n")                         
        for act, cnt in self.act_cnt.items():
            logging.info(f"{act} {cnt}")
