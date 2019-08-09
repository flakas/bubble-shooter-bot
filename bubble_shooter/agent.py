import time
import numpy as np
import random
import tensorflow as tf
import os
from bubble_shooter.modified_tensorboard import ModifiedTensorBoard
from bubble_shooter.coordinate_mapper import CoordinateMapper

GAME_BOARD_DIMENSION = 64
COLOR_SPACE = 3
GAME_BOARD_X = 17
GAME_BOARD_Y = 15
GAME_BOARD_DEPTH = 8

class Agent:
    def __init__(self, state_size, action_size, move_size, memory, epsilon=1.0, gamma=0.9, learning_rate=0.00025, update_target_frequency=10, replay_frequency=4, batch_size=32, preplay_steps=500, name=None):
        self.graph = tf.get_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf.keras.backend.set_session(self.sess)

        self.state_size = state_size
        self.action_size = action_size
        self.move_size = move_size
        self.coordinate_mapper = CoordinateMapper(action_size, move_size)
        self.memory = memory
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99 # 0.995
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.name = name
        self.update_target_frequency = update_target_frequency
        self.replay_frequency = replay_frequency
        self.preplay_steps = preplay_steps
        self.steps = 0
        self.episode = 0

        with self.sess.as_default():
            with self.graph.as_default():
                if name:
                    self.filename = f'models/{name}.h5'

                self.model = self._build_model()

                if self.filename and os.path.exists(self.filename):
                    self.model = self.load_model(self.filename)
                else:
                    pass

                print('Agent model', self.model.summary())
                tf.keras.utils.plot_model(self.model, to_file='model.png', show_shapes=True)

                self.target_model = self._build_model()

                self.model._make_predict_function()
                self.target_model._make_predict_function()

                self.refresh_target_model()

                self.callbacks = self.get_callbacks()

    def load_model(self, path):
        print('Loading model', path)
        with self.sess.as_default():
            with self.graph.as_default():
                self.model.load_weights(path)
                self.model._make_predict_function()
                return self.model

    def save_model(self, path=None):
        if not path:
            assert(self.filename != None)
            path = self.filename
        with self.sess.as_default():
            with self.graph.as_default():
                self.model.save_weights(path)

    def _build_simple_dense_model(self):
        from bubble_shooter.models.simple_dense import SimpleDense
        input_shape = (GAME_BOARD_DIMENSION, GAME_BOARD_DIMENSION, COLOR_SPACE)
        output_shape = self.move_size
        model = SimpleDense(input_shape=input_shape, output_shape=output_shape, self.learning_rate)
        return model.build()

    def _build_convolutional_model(self):
        from bubble_shooter.models.convolutional import Convolutional
        input_shape = (GAME_BOARD_DIMENSION, GAME_BOARD_DIMENSION, COLOR_SPACE)
        output_shape = self.move_size
        model = Convolutional(input_shape=input_shape, output_shape=output_shape, self.learning_rate)
        return model.build()

    def _build_pooled_convolutional_model(self):
        from bubble_shooter.models.pooled_convolutional import PooledConvolutional
        input_shape = (GAME_BOARD_DIMENSION, GAME_BOARD_DIMENSION, COLOR_SPACE)
        output_shape = self.move_size
        model = PooledConvolutional(input_shape=input_shape, output_shape=output_shape, self.learning_rate)
        return model.build()

    def _build_convolutional_model2(self):
        from bubble_shooter.models.deepmind_convolutional import DeepmindConvolutional
        input_shape = (GAME_BOARD_DIMENSION, GAME_BOARD_DIMENSION, COLOR_SPACE)
        output_shape = self.move_size
        model = DeepmindConvolutional(input_shape=input_shape, output_shape=output_shape, self.learning_rate)
        return model.build()

    def _build_dueling_model(self):
        from bubble_shooter.models.dueling import Dueling
        input_shape = (GAME_BOARD_DIMENSION, GAME_BOARD_DIMENSION, COLOR_SPACE)
        output_shape = self.move_size
        model = Dueling(input_shape=input_shape, output_shape=output_shape, self.learning_rate)
        return model.build()

    def _build_dueling_model_ala_inception(self):
        from bubble_shooter.models.dueling_inception import DuelingInception
        input_shape = (GAME_BOARD_DIMENSION, GAME_BOARD_DIMENSION, COLOR_SPACE)
        output_shape = self.move_size
        model = DuelingInception(input_shape=input_shape, output_shape=output_shape, self.learning_rate)
        return model.build()

    def _build_model(self):
        # model = self._build_simple_dense_model()
        # model = self._build_convolutional_model()
        # model = self._build_pooled_convolutional_model()
        # model = self._build_convolutional_model2()
        #model = self._build_dueling_model()
        model = self._build_dueling_model_ala_inception()
        return model

    def refresh_target_model(self):
        with self.sess.as_default():
            with self.graph.as_default():
                self.target_model.set_weights(self.model.get_weights())

    def predict(self, state, target=False):
        tf.keras.backend.set_session(self.sess)
        with self.sess.as_default():
            with self.graph.as_default():
                if target:
                    return self.target_model.predict(state)
                else:
                    return self.model.predict(state)

    def remember(self, state, action, reward, next_state, done):
        index = 0 # because it is generated in memory.sample()
        if done:
            # When the game is done, all matching pieces are set to zero (not matching).
            # Useful only in model-based learning
            next_state = np.zeros(np.shape(state))
        experience = (state, action, reward, next_state, done)
        (states, targets, errors) = self.get_targets([(index, experience)])
        self.memory.add(errors[0], experience)

    def act(self, state):
        if np.random.rand() <= self.epsilon or not self.memory.has_enough_samples(32):
            action = random.randrange(self.move_size)
        else:
            state = np.array(state).reshape((1, GAME_BOARD_Y, GAME_BOARD_X, GAME_BOARD_DEPTH))
            act_values = self.predict(state)
            top_actions = act_values[0].argsort()[-5:][::-1]

            action = top_actions[0]
            top_q_values = list((a, act_values[0][a]) for a in top_actions)
            q_value = np.max(act_values[0])
            print(f'NN moving to {action} (Q: {q_value}) (Top Q: {top_q_values})')

        return self.coordinate_mapper.agent_to_game(action)

    def act_with_stats(self, state):
        """ Chooses the recommended action and returns evaluations for all other actions """

        state = np.array(state).reshape((1, GAME_BOARD_Y, GAME_BOARD_X, GAME_BOARD_DEPTH))
        act_values = self.predict(state)
        top_actions = act_values[0].argsort()[-5:][::-1]

        action = top_actions[0]
        top_q_values = list((a, act_values[0][a]) for a in top_actions)
        q_value = np.max(act_values[0])
        print(f'NN moving to {action} (Q: {q_value}) (Top Q: {top_q_values})')

        evaluations = [(self.coordinate_mapper.agent_to_game(i), q_value) for (i, q_value) in enumerate(act_values[0])]

        return {
            'recommended_action': self.coordinate_mapper.agent_to_game(action),
            'action_evaluations': list(evaluations)
        }

    def replay(self, minibatch):
        (states, targets, errors) = self.get_targets(minibatch, batch_size=len(minibatch))

        for (i, (index, data)) in enumerate(minibatch):
            self.memory.update(index, errors[i])

        tf.keras.backend.set_session(self.sess)
        with self.sess.as_default():
            with self.graph.as_default():
                self.save_model()
                self.model.fit(states, targets, batch_size=len(minibatch), epochs=1, callbacks=self.callbacks, verbose=1)

    def after_step(self, step):
        self.steps += 1
        self.callbacks[1].step = self.steps

        if self.steps % self.update_target_frequency == 0:
            self.refresh_target_model()

        if self.steps % self.replay_frequency == 0 and self.steps >= self.preplay_steps:
            minibatch = self.memory.sample(self.batch_size)
            self.replay(minibatch)

    def after_episode(self, episode):
        self.episode += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_targets(self, batch, batch_size=1):
        no_state = np.zeros(self.state_size)

        starting_states = np.array([observation[1][0] for observation in batch])
        ending_states = np.array([no_state if observation[1][3] is None else observation[1][3] for observation in batch])

        current_predictions = self.predict(starting_states)

        predictions_ending_online = self.predict(ending_states, target=False)
        predictions_ending_target = self.predict(ending_states, target=True)

        targets = []
        states = []
        errors = []

        for (i, (memory_index, (state, action, reward, next_state, done))) in enumerate(batch):
            q_values = current_predictions[i]
            agent_action = self.coordinate_mapper.game_to_agent(action)

            old_action_q_value = q_values[agent_action]
            if done:
                q_values[agent_action] = reward
            else:
                online_target_action = np.argmax(predictions_ending_online[i])
                q_values[agent_action] = reward + self.gamma * predictions_ending_target[i][online_target_action]

            states.append(state)
            targets.append(q_values)
            errors.append(abs(old_action_q_value - q_values[agent_action]))

        states = np.array(states)
        targets = np.array(targets)
        errors = np.array(errors)

        return (states, targets, errors)

    def get_callbacks(self):
        return [
            tf.keras.callbacks.ModelCheckpoint(f'./checkpoints/{self.name}-{int(time.time())}/', save_weights_only=True),
            ModifiedTensorBoard(log_dir='logs_inception/{}-{}'.format(self.name, int(time.time())))
        ]

    def remember_episode_rewards(self, total_reward, min_reward, avg_reward, max_reward, episode_action_variance, steps_played):
        tensorboard = self.callbacks[1]

        tensorboard.update_stats(
            total_reward = total_reward,
            reward_min = min_reward,
            reward_avg = avg_reward,
            reward_max = max_reward,
            epsilon = self.epsilon,
            episode_action_variance = episode_action_variance,
            steps_played = steps_played)
