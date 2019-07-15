from agent import Agent
from controller import Controller
from vision import ScreenshotSource, SeleniumSource, Vision
from game import Game
# from memory import Memory
from prioritized_memory import Memory
import cv2
import numpy as np
from selenium_browser import SeleniumBrowser
# import asyncio
import pykka
import logging
import time

logging.basicConfig(level=logging.DEBUG)

GAME_BOARD_DIMENSION = 64
COLOR_SPACE = 3
GAME_BOARD_X = 35
GAME_BOARD_Y = 15
GAME_BOARD_DEPTH = 4

class AgentActor(pykka.ThreadingActor):
    def __init__(self, config):
        super().__init__()
        self.agent = Agent(
                #state_size=GAME_BOARD_DIMENSION*GAME_BOARD_DIMENSION*COLOR_SPACE,
                state_size=GAME_BOARD_X*GAME_BOARD_Y*GAME_BOARD_DEPTH,
                action_size=560,
                move_size=35,
                memory=Memory(config['memory_size'], epsilon=config['memory_epsilon'], alpha=config['memory_alpha']),
                epsilon=config['epsilon'],
                gamma=config['gamma'],
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'],
                update_target_frequency=config['target_update_frequency'],
                replay_frequency=config['replay_frequency'],
                name=f"dueling1_mse_vsinit_{config['epsilon']}eps_{config['gamma']}gamma_{config['learning_rate']}lr_{config['replay_frequency']}refr_{config['target_update_frequency']}upfr_{config['memory_alpha']}memal_{config['batch_size']}bs_normbinaryrewards_parsedstate_onlycurnext")

    def act(self, state):
        return self.agent.act(state)

    def remember(self, state, action, reward, next_state, done):
        return self.agent.remember(state, action, reward, next_state, done)

    def replay(self, minibatch_size):
        return self.agent.replay(minibatch_size)

    def remember_episode_rewards(self, *rewards):
        return self.agent.remember_episode_rewards(*rewards)

    def after_step(self, step):
        return self.agent.after_step(step)

    def after_episode(self, episode):
        return self.agent.after_episode(episode)

class TrainingSupervisorActor(pykka.ThreadingActor):
    def __init__(self, total_trainers=2):
        super().__init__()
        self.total_trainers = total_trainers

    def on_start(self):
        self.trainers = [TrainerActor.start().proxy() for i in range(self.total_trainers)]

    def train(self, config):
        # print('Got a request to train in supervisor')
        agent_actor = AgentActor.start(config).proxy()
        episodes_per_trainer = config['episodes'] // self.total_trainers
        training_futures = [trainer.train(agent_actor, episodes_per_trainer, config['steps'], config['batch_size'], config['replay_frequency']) for trainer in self.trainers]
        return pykka.get_all(training_futures)

class TrainerActor(pykka.ThreadingActor):

    def on_start(self):
        # print('Got a request to start the trainer')
        self.selenium = SeleniumBrowser(headless=False)
        self.selenium_source = SeleniumSource(self.selenium)
        self.vision = Vision(self.selenium_source, templates_path='templates/')
        self.controller = self.selenium
        self.game = Game(self.vision, self.controller)

        self.selenium.setup()

    def on_stop(self):
        self.selenium.cleanup()

    def train(self, agent, episodes, steps, minibatch_size, replay_frequency=1):
        # print('Got a request to train in trainer')
        episode_rewards = []
        for e in range(episodes):
            state = self.game.get_state()
            total_reward = 0
            steps_taken = 0

            actions_taken = []
            for time_t in range(steps):
                steps_taken += 1
                action = agent.act(state).get()
                actions_taken.append(action)

                reward = self.game.perform_move(action, 400)
                total_reward += reward
                next_state = self.game.get_state()
                done = self.game.is_finished()

                agent.remember(state, action, reward, next_state, done).get()
                agent.after_step(time_t)
                print(f'[AGENT] Episode: {e}/{episodes}, step: {time_t}/{steps}, action: {action}, reward: {reward}/{total_reward}, done: {done}')

                state = next_state

                if done:
                    print(f'[AGENT] episode: {e}/{episodes}, score: {total_reward}')
                    self.game.restart_the_game()
                    break


            agent.after_episode(e)

            episode_rewards.append(total_reward)
            episode_action_variance = np.var(actions_taken)
            agent.remember_episode_rewards(
                total_reward,
                min(episode_rewards),
                sum(episode_rewards)/len(episode_rewards),
                max(episode_rewards),
                episode_action_variance,
                steps_taken).get()


# class TrainingSupervisor:
    # def __init__(self, total_trainers=2):
        # self.total_trainers = total_trainers
        # self.trainers = [self._create_trainer() for i in range(total_trainers)]

    # @asyncio.coroutine
    # async def train(self, configurations):
        # for config in configurations:
            # agent = Agent(
                # state_size=64*64*3,
                # action_size=560,
                # move_size=20,
                # memory=Memory(config['memory_size'], epsilon=config['memory_epsilon'], alpha=config['memory_alpha']),
                # epsilon=config['epsilon'],
                # gamma=config['gamma'],
                # learning_rate=config['learning_rate'],
                # update_target_frequency=config['target_update_frequency'],
                # name=f"doubledqn_pooled_{config['epsilon']}eps_{config['gamma']}gamma_{config['learning_rate']}lr_{config['target_update_frequency']}upfr_{config['memory_alpha']}memal_{config['batch_size']}bs_normrewards_woffsets")
            # episodes_per_trainer = config['episodes'] // self.total_trainers
            # trainings = await asyncio.gather(*[trainer.train(agent, episodes=episodes_per_trainer, steps=config['steps'], minibatch_size=config['batch_size']) for trainer in self.trainers])

    # def _create_trainer(self):
        # return Trainer()

    # @asyncio.coroutine
    # async def close(self):
        # return await asyncio.gather(*[trainer.close() for trainer in self.trainers])

# class Trainer:
    # def __init__(self):
        # self.selenium = SeleniumBrowser(headless=False)
        # self.selenium_source = SeleniumSource(self.selenium)
        # self.vision = Vision(self.selenium_source, templates_path='templates/')
        # self.controller = self.selenium
        # self.game = Game(self.vision, self.controller)

    # @asyncio.coroutine
    # async def train(self, agent, episodes, steps, minibatch_size):
        # self.selenium.setup()

        # episode_rewards = []
        # for e in range(episodes):
            # state = await self.game.get_state()
            # total_reward = 0
            # steps_taken = 0

            # actions_taken = []
            # for time_t in range(steps):
                # steps_taken += 1
                # action = agent.act(state)
                # actions_taken.append(action)

                # reward = await self.game.perform_move(action, 400)
                # total_reward += reward
                # next_state = await self.game.get_state()
                # done = self.game.is_finished()

                # agent.remember(state, action, reward, next_state, done)
                # print(f'[AGENT] Episode: {e}/{episodes}, step: {time_t}/{steps}, action: {action}, reward: {reward}/{total_reward}, done: {done}')

                # state = next_state

                # if done:
                    # print(f'[AGENT] episode: {e}/{episodes}, score: {total_reward}')
                    # await self.game.restart_the_game()
                    # break

            # agent.replay(minibatch_size)
            # episode_rewards.append(total_reward)
            # episode_action_variance = np.var(actions_taken)
            # agent.remember_episode_rewards(
                # min(episode_rewards),
                # sum(episode_rewards)/len(episode_rewards),
                # max(episode_rewards),
                # episode_action_variance,
                # steps_taken)

    # async def close(self):
        # self.selenium.cleanup()

# def train2(agent, episodes, steps, minibatch_size=64):
    # # vision = Vision(ScreenshotSource(), templates_path='templates/')
    # # controller = Controller()

    # selenium = SeleniumBrowser(headless=False)
    # selenium_source = SeleniumSource(selenium)
    # vision = Vision(selenium_source, templates_path='templates/')
    # controller = selenium
    # # selenium.setup()
    # game = Game(vision, controller)

    # episode_rewards = []
    # agent.memory.load_from_file('game_states.pickle')
    # for e in range(episodes):
        # state = game.get_state()
        # total_reward = 0
        # steps_taken = 0

        # actions_taken = []
        # for time_t in range(steps):
            # steps_taken += 1
            # action = agent.act(state)
            # actions_taken.append(action)

            # reward = game.perform_move(action, 400)
            # total_reward += reward
            # next_state = game.get_state()
            # done = game.is_finished()

            # agent.remember(state, action, reward, next_state, done)
            # print(f'[AGENT] Episode: {e}/{episodes}, step: {time_t}/{steps}, action: {action}, reward: {reward}/{total_reward}, done: {done}')

            # state = next_state

            # if done:
                # print(f'[AGENT] episode: {e}/{episodes}, score: {total_reward}')
                # game.restart_the_game()
                # break

        # agent.replay(minibatch_size, e)
        # episode_rewards.append(total_reward)
        # episode_action_variance = np.var(actions_taken)
        # agent.remember_episode_rewards(
            # min(episode_rewards),
            # sum(episode_rewards)/len(episode_rewards),
            # max(episode_rewards),
            # episode_action_variance,
            # steps_taken)
        # agent.memory.persist_to_file('game_states.pickle')

    # selenium.cleanup()

# agent = Agent(state_size=128*128*3, action_size=560, move_size=28, memory=Memory(5000), epsilon=0.9, name='doubledqn_pooled_convo_0.00025lr_smallrewards')
# train2(agent, episodes=400, steps=500)

TOTAL_TRAINERS = 24
configurations = [
        { 'epsilon': 0.99, 'gamma': 0.9, 'learning_rate': 0.00025, 'replay_frequency': 4, 'target_update_frequency': 1000, 'memory_epsilon': 0.01, 'memory_alpha': 0.6, 'memory_size': 20000, 'batch_size': 32, 'episodes': TOTAL_TRAINERS*400, 'steps': 500, },
]

# for config in configurations:
    # agent = Agent(
        # state_size=64*64*3,
        # action_size=560,
        # move_size=20,
        # memory=Memory(config['memory_size'], epsilon=config['memory_epsilon'], alpha=config['memory_alpha']),
        # epsilon=config['epsilon'],
        # gamma=config['gamma'],
        # learning_rate=config['learning_rate'],
        # update_target_frequency=config['target_update_frequency'],
        # name=f"doubledqn_pooledconvo_{config['epsilon']}eps_{config['gamma']}gamma_{config['learning_rate']}lr_{config['target_update_frequency']}upfr_{config['memory_alpha']}memal_{config['batch_size']}bs_normrewards")
    # train2(agent, episodes=config['episodes'], steps=config['steps'], minibatch_size=config['batch_size'])
if __name__ == '__main__':
    # supervisor = TrainingSupervisor(total_trainers=4)
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(supervisor.train(configurations))
    # loop.run_until_complete(supervisor.close())
    # loop.close()
    supervisor = TrainingSupervisorActor.start(total_trainers=TOTAL_TRAINERS).proxy()
    # print('After setup')
    for config in configurations:
        time.sleep(10)
        pykka.get_all([supervisor.train(config)])
    pykka.ActorRegistry.stop_all()
    # print('After train')
