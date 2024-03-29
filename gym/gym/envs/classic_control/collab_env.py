# -*- coding: utf-8 -*-
"""
@author: H
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

min_x, min_y, min_vx, min_vy = 0.0, 0.0, -2.0, -15.0
max_x, max_y, max_vx, max_vy = 10.0, 8.0, 2.0, 15.0


class Agent(object):
    def __init__(self, x=0.0, y=0.5, length=0.5, height=0.5, agent_number=0):
        self.gravity = 5
        self.mass = 1.0
        self.height = height
        self.length = length  #
        self.agent_number = agent_number

        self.min_x, self.min_y, self.min_vx, self.min_vy = (
            min_x + self.length,
            min_y + self.height,
            min_vx,
            min_vy,
        )
        self.max_x, self.max_y, self.max_vx, self.max_vy = (
            max_x - self.length,
            max_y - self.height,
            max_vx,
            max_vy,
        )

        self.on_ground = True  # = can jump

        self.jump_force = 80.0
        self.force_mag = 10.0
        self.tau = 0.05

        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0

    def up_down_update(self, agents, obstacles, jump=False):
        # handles gravity and up/down movement collisions

        if not (self.on_ground):
            new_vy = max(self.vy - self.gravity * self.tau, self.min_vy)
        if self.on_ground:
            # print str(self.agent_number) + " is on ground"
            if jump:
                new_vy = min(
                    self.vy + (self.jump_force / self.mass - self.gravity) * self.tau,
                    self.max_vy,
                )
                self.on_ground = False
            else:
                new_vy = 0.0

        new_y = self.y + new_vy * self.tau

        current_max = -1000.0
        current_min = 1000.0
        intersect_agent_upwards = False
        intersect_agent_downwards = False
        contact = False
        for agent in agents:
            if not (self.agent_number == agent.agent_number):
                if (
                    (new_y < agent.y + self.height + agent.height)
                    and (self.y > agent.y)
                    and (abs(self.x - agent.x) < self.length + agent.length)
                ):
                    current_max = max(current_max, agent.y + self.height + agent.height)
                    intersect_agent_downwards = True
                if (
                    (new_y > agent.y - self.height - agent.height)
                    and (self.y < agent.y)
                    and (abs(self.x - agent.x) < self.length + agent.length)
                ):
                    current_min = min(current_min, agent.y - self.height - agent.height)
                    intersect_agent_upwards = True
                if (abs(self.y - agent.y) <= self.height + agent.height) and (
                    abs(self.x - agent.x) < (self.length + agent.length)
                ):
                    contact = True

        if not (contact):
            self.on_ground = False

        intersect_obstacle_upwards = False
        intersect_obstacle_downwards = False

        for obstacle in obstacles:
            if (
                (new_y < obstacle.y + self.height + obstacle.height)
                and (self.y > obstacle.y)
                and (abs(self.x - obstacle.x) < self.length + obstacle.length)
            ):
                current_max = max(
                    current_max, obstacle.y + self.height + obstacle.height
                )
                intersect_obstacle_downwards = True
            if (
                (new_y > obstacle.y - self.height - obstacle.height)
                and (self.y < obstacle.y)
                and (abs(self.x - obstacle.x) < self.length + obstacle.length)
            ):
                current_min = min(
                    current_min, obstacle.y - self.height - obstacle.height
                )
                intersect_obstacle_upwards = True

        intersect_downwards = intersect_obstacle_downwards or intersect_agent_downwards
        intersect_upwards = intersect_agent_upwards or intersect_obstacle_upwards

        if intersect_downwards:
            self.vy = 0.0
            self.y = current_max
            self.on_ground = True
        if intersect_upwards:
            self.vy = 0.0
            self.y = current_min

        intersect = intersect_downwards or intersect_upwards

        if not (intersect):
            if new_y <= self.min_y:
                self.vy = 0.0
                self.y = self.min_y
                self.on_ground = True
            elif new_y >= self.max_y:
                self.vy = 0.0
                self.y = self.max_y
            else:
                self.vy = new_vy
                self.y = new_y

        return

    def left_right_update(self, agents, obstacles, direction=0):
        new_vx = min(
            max(
                self.vx + direction * self.force_mag / self.mass * self.tau, self.min_vx
            ),
            self.max_vx,
        )
        new_x = self.x + new_vx * self.tau

        current_max = -1000.0
        current_min = 1000.0
        intersect_agent_leftwards = False
        intersect_agent_rightwards = False
        for agent in agents:
            if not (agent.agent_number == self.agent_number):
                if (
                    (new_x < (agent.x + self.length + agent.length))
                    and (self.x >= agent.x)
                    and (abs(self.y - agent.y) < (self.height + agent.height))
                ):
                    current_max = max(current_max, agent.x + self.length + agent.length)
                    intersect_agent_leftwards = True
                if (
                    (new_x > (agent.x - self.length - agent.length))
                    and (self.x <= agent.x)
                    and (abs(self.y - agent.y) < (self.height + agent.height))
                ):
                    current_min = min(current_min, agent.x - self.length - agent.length)
                    intersect_agent_rightwards = True

        intersect_obstacle_leftwards = False
        intersect_obstacle_rightwards = False
        for obstacle in obstacles:
            if (
                (new_x < (obstacle.x + self.length + obstacle.length))
                and (self.x >= obstacle.x)
                and (abs(self.y - obstacle.y) < (self.height + obstacle.height))
            ):
                current_max = max(
                    current_max, obstacle.x + self.length + obstacle.length
                )
                intersect_obstacle_leftwards = True
            if (
                (new_x > (obstacle.x - self.length - obstacle.length))
                and (self.x <= obstacle.x)
                and (abs(self.y - obstacle.y) < (self.height + obstacle.height))
            ):
                current_min = min(
                    current_min, obstacle.x - self.length - obstacle.length
                )
                intersect_obstacle_rightwards = True

        intersect_leftwards = intersect_obstacle_leftwards or intersect_agent_leftwards
        intersect_rightwards = (
            intersect_obstacle_rightwards or intersect_agent_rightwards
        )

        if intersect_leftwards:
            self.vx = 0.0
            self.x = current_max
        if intersect_rightwards:
            self.vx = 0.0
            self.x = current_min

        intersect = intersect_leftwards or intersect_rightwards

        if not (intersect):
            if new_x < self.min_x:
                self.vx = 0.0
                self.x = self.min_x
            elif new_x > self.max_x:
                self.vx = 0.0
                self.x = self.max_x
            else:
                self.vx = new_vx
                self.x = new_x

        return

    def wait(self, agents, obstacles):
        self.left_right_update(agents, obstacles, direction=0)
        self.up_down_update(agents, obstacles, jump=False)

    def jump(self, agents, obstacles):
        self.left_right_update(agents, obstacles, direction=0)
        self.up_down_update(agents, obstacles, jump=True)

    def move_left(self, agents, obstacles):
        self.left_right_update(agents, obstacles, direction=-1)
        self.up_down_update(agents, obstacles, jump=False)

    def move_right(self, agents, obstacles):
        self.left_right_update(agents, obstacles, direction=1)
        self.up_down_update(agents, obstacles, jump=False)


class Obstacle(object):
    def __init__(self, x, y, length, height):
        self.x = x
        self.y = y
        self.length = length
        self.height = height


class Star(object):
    def __init__(self, x, y, length, height):
        self.x = x
        self.y = y
        self.length = length
        self.height = height
        self.found = False


class CollabEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 1}

    # Actions are: (0:left, 1:wait, 2:right, 3:up) * number of agents

    def __init__(self):
        self.gravity = 9.8
        self.tau = 0.02  # seconds between state updates
        self.number_of_agents = 1
        self.viewer = None

        self.agents = []
        self.obstacles = []

        self.star = None

        self.min_x, self.min_y, self.min_vx, self.min_vy = min_x, min_y, min_vx, min_vy
        self.max_x, self.max_y, self.max_vx, self.max_vy = max_x, max_y, max_vx, max_vy

        one_low = np.array([self.min_x, self.min_y, self.min_vx, self.min_vy])
        one_high = np.array([self.max_x, self.max_y, self.max_vx, self.max_vy])
        low = [one_low for _ in range(self.number_of_agents)]
        high = [one_high for _ in range(self.number_of_agents)]
        self.low = np.concatenate(low)
        self.high = np.concatenate(high)

        self.action_space = spaces.Discrete(4 * self.number_of_agents)
        self.observation_space = spaces.Box(self.low, self.high)

        self._seed()

        # self.steps_beyond_done = None
        # self._reset()

    def update(self):  # updates the state of the environment
        reward = 0.0
        done = False
        for i, agent in enumerate(self.agents):
            self.state[4 * i] = agent.x
            self.state[4 * i + 1] = agent.y
            self.state[4 * i + 2] = agent.vx
            self.state[4 * i + 3] = agent.vy

            target = (agent.min_x + agent.max_x) / 2
            reward += -np.square(agent.x - target) - agent.y / 2  # / 10.0
            # print agent.x, reward

            star = self.star
            if (
                not (star.found)
                and (abs(agent.x - star.x) < (agent.length + star.length))
                and (abs(agent.y - star.y) < (agent.height + star.height))
            ):
                reward += 0.0
                star.found = True
                # done = True

        # reward = reward / self.number_of_agents
        return reward, done

    def others_wait(self, agent_number):
        for agent in self.agents:
            if not (agent.agent_number == agent_number):
                agent.wait(self.agents, self.obstacles)
        return

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        agent_number = int(action / 4)
        to_do = action % 4

        if to_do == 0:
            self.agents[agent_number].move_left(self.agents, self.obstacles)
        if to_do == 1:
            self.agents[agent_number].wait(self.agents, self.obstacles)
        if to_do == 2:
            self.agents[agent_number].move_right(self.agents, self.obstacles)
        if to_do == 3:
            self.agents[agent_number].jump(self.agents, self.obstacles)

        self.others_wait(agent_number)
        reward, done = self.update()

        return np.array(self.state), reward, done, {}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):  # pas pensé pour trop d'agents

        self.state = np.zeros(4 * self.number_of_agents)
        self.agents = []
        self.obstacles = []

        for i in range(self.number_of_agents):
            self.agents.append(
                Agent(
                    x=self.min_x + 0.5 + i,
                    y=0.5,
                    length=0.5,
                    height=0.5,
                    agent_number=i,
                )
            )

        for agent in self.agents:
            i = agent.agent_number
            self.state[4 * i] = agent.x
            self.state[4 * i + 1] = agent.y
            self.state[4 * i + 2] = agent.vx
            self.state[4 * i + 3] = agent.vy

        self.star = Star(3.0, 0.5, 2.0, 0.5)
        self.obstacles = [Obstacle(5.0, 0.1, 0.5, 0.1), Obstacle(2.5, 0.1, 0.5, 0.1)]

        return np.array(self.state)

    def _render(self, mode="human", close=False):

        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.max_x - self.min_x
        world_height = self.max_y - self.min_y
        x_scale = screen_width / world_width
        y_scale = screen_height / world_height
        ground_offset = 0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.agents_render = []
            self.agents_trans = []

            for agent in self.agents:
                i = agent.agent_number
                l, r, t, b = -agent.length, agent.length, agent.height, -agent.height
                l, r, t, b = l * x_scale, r * x_scale, t * y_scale, b * y_scale
                agent_image = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])

                self.agents_trans.append(rendering.Transform())
                agent_image.add_attr(self.agents_trans[-1])
                agent_image.set_color(30 * i, 0.0, 0.0)
                self.viewer.add_geom(agent_image)
                self.agents_render.append(agent_image)

            self.obstacles_render = []
            self.obstacles_trans = []

            for obstacle in self.obstacles:
                l, r, t, b = (
                    -obstacle.length,
                    obstacle.length,
                    obstacle.height,
                    -obstacle.height,
                )
                l, r, t, b = l * x_scale, r * x_scale, t * y_scale, b * y_scale
                obstacle_image = rendering.FilledPolygon(
                    [(l, b), (l, t), (r, t), (r, b)]
                )

                self.obstacles_trans.append(rendering.Transform())
                obstacle_image.add_attr(self.obstacles_trans[-1])
                obstacle_image.set_color(0.8, 0.6, 0.4)

                self.viewer.add_geom(obstacle_image)
                self.obstacles_render.append(obstacle_image)

            # star = self.star
            # l,r,t,b = -star.length, star.length, star.height, -star.height
            # l,r,t,b = l*x_scale, r*x_scale, t*y_scale, b*y_scale
            # star_image = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            # star_image.set_color(255,215,0)
            # star_trans = rendering.Transform()
            # star_image.add_attr(star_trans)
            # self.viewer.add_geom(star_image)
            # x, y = star.x, star.y
            # x = (x - self.min_x)*x_scale
            # y = (y - self.min_y)*y_scale
            # star_trans.set_translation(x, y)

        for i, obstacle in enumerate(self.obstacles):
            x, y = obstacle.x, obstacle.y
            x = (x - self.min_x) * x_scale
            y = (y - self.min_y) * y_scale
            self.obstacles_trans[i].set_translation(x, y)

        for agent in self.agents:
            i = agent.agent_number
            x, y = agent.x, agent.y
            x = (x - self.min_x) * x_scale
            y = (y - self.min_y) * y_scale
            self.agents_trans[i].set_translation(x, y)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
