import numpy as np


class GoalNavAgent(object):
    def __init__(self, id, action_space, goal_area, goal_list=None):
        self.id = id
        self.velocity_low = action_space['velocity'][0]
        self.velocity_high = action_space['velocity'][1]
        self.angle_low = action_space['angle'][0]
        self.angle_high = action_space['angle'][1]
        self.goal_area = goal_area
        self.obstacle = None

        self.step_counter = 0
        self.goal_id = 0
        self.max_len = 100
        self.research_times = 10
        self.delta_time = 0.13
        self.goal_list = goal_list

        self.goal = None
        self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
        self.pose_now = []
        self.pose_last = [[], []]

    def act(self):
        self.step_counter += 1
        if len(self.pose_last[0]) == 0:
            self.pose_last[0] = np.array(self.pose_now)
            self.pose_last[1] = np.array(self.pose_now)
            d_moved = 30
        else:
            d_moved = min(np.linalg.norm(np.array(self.pose_last[0]) - np.array(self.pose_now)),
                          np.linalg.norm(np.array(self.pose_last[1]) - np.array(self.pose_now)))
            self.pose_last[0] = np.array(self.pose_last[1])
            self.pose_last[1] = np.array(self.pose_now)
        if self.check_reach(self.goal, self.pose_now) or d_moved < 10 or self.step_counter > self.max_len:
            self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
            self.step_counter = 0
            self.goal = self.generate_goal()

        left_distance = np.linalg.norm(self.goal[:2] - self.pose_now[:2])
        if left_distance > 0:
            delt_unit = (self.goal[:2] - self.pose_now[:2]) / left_distance
        else:
            delt_unit = [0, 0]
        velocity = self.velocity * (1 + 0.2 * np.random.random())
        if velocity * self.delta_time > left_distance:
            self.pose_now = self.goal.copy()
        else:
            self.pose_now[0] += delt_unit[0] * velocity * self.delta_time
            self.pose_now[1] += delt_unit[1] * velocity * self.delta_time
            self.pose_now[0] = np.clip(self.pose_now[0], self.goal_area[0], self.goal_area[1])
            self.pose_now[1] = np.clip(self.pose_now[1], self.goal_area[2], self.goal_area[3])
        return self.pose_now.copy()

    def reset(self, init_pose, obstacle):
        self.step_counter = 0
        self.goal_id = 0
        self.velocity = np.random.randint(self.velocity_low, self.velocity_high)
        self.pose_last = [[], []]
        self.pose_now = init_pose
        self.obstacle = obstacle
        self.goal = self.generate_goal()

    def generate_goal(self):
        x_len = self.goal_area[1] - self.goal_area[0]
        y_len = self.goal_area[3] - self.goal_area[2]
        if self.goal_list and len(self.goal_list) != 0:
            index = self.goal_id % len(self.goal_list)
            goal = np.array(self.goal_list[index])
        else:
            legal_goal = True
            for i in range(self.research_times):
                x = np.random.randint(max(self.goal_area[0], int(self.pose_now[0] - x_len + x_len / self.research_times * i)),
                                      min(self.goal_area[1], int(self.pose_now[0] + x_len - x_len / self.research_times * i)))
                y = np.random.randint(max(self.goal_area[2], int(self.pose_now[1] - y_len + y_len / self.research_times * i)),
                                      min(self.goal_area[3], int(self.pose_now[1] + y_len - y_len / self.research_times * i)))
                goal = np.array([x, y])
                legal_goal = True
                for item in self.obstacle:
                    if self.cross_lines((self.pose_now, goal), ((item[0], item[1]), (item[2], item[3]))):
                        legal_goal = False
                        break
                if legal_goal:
                    break
            if not legal_goal:
                goal = self.pose_now.copy()
        self.goal_id += 1
        return goal

    def check_reach(self, goal, now):
        error = np.array(now[:2]) - np.array(goal[:2])
        distance = np.linalg.norm(error)
        return distance < 5

    def cross_lines(self, la, lb):
        """
        :param la: [(ax1, ay1), (ax2, ay2)]
        :param lb: [(bx1, by1), (bx2, by2)]
        :return: True or False
        """
        if max(la[0][0], la[1][0]) < min(lb[0][0], lb[1][0]) or min(la[0][0], la[1][0]) > max(lb[0][0], lb[1][0]) or \
           max(la[0][1], la[1][1]) < min(lb[0][1], lb[1][1]) or min(la[0][1], la[1][1]) > max(lb[0][1], lb[1][1]):
            return False
        if self.cross_product(self.vector_minus(la[0], lb[0]), self.vector_minus(lb[1], lb[0])) * \
           self.cross_product(self.vector_minus(la[1], lb[0]), self.vector_minus(lb[1], lb[0])) > 0 or \
           self.cross_product(self.vector_minus(lb[0], la[0]), self.vector_minus(la[1], la[0])) * \
           self.cross_product(self.vector_minus(lb[1], la[0]), self.vector_minus(la[1], la[0])) > 0:
            return False
        return True

    @staticmethod
    def vector_minus(va, vb):
        answer = (va[0] - vb[0], va[1] - vb[1])
        return answer

    @staticmethod
    def cross_product(va, vb):
        return va[0] * vb[1] - va[1] * vb[0]
