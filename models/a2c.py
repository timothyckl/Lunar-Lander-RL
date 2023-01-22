import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


class A2C:
    def __init__(self, env, actor_lr, critic_lr, gamma):
        self.env = env
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.action_space = [i for i in range(self.action_size)]
        self.actor, self.critic, self.policy = self.create_ac_network()

    def create_ac_network(self):
        state = Input(shape=(self.state_size,))
        delta = Input(shape=[1])

        dense = Dense(units=1024, activation='relu')(state)
        dense = Dense(units=512, activation='relu')(dense)

        probabilities = Dense(units=self.action_size, activation='softmax')(dense)
        values = Dense(units=1, activation='linear')(dense)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1 - 1e-8)
            log_likelihood = y_true * K.log(out)

            return K.sum(-log_likelihood * delta)
        
        actor = Model(inputs=[state, delta], outputs=[probabilities])
        actor.compile(loss=custom_loss, optimizer=Adam(learning_rate=self.actor_lr))

        critic = Model(inputs=[state], outputs=[values])
        critic.compile(loss='mse', optimizer=Adam(learning_rate=self.critic_lr))

        policy = Model(inputs=[state], outputs=[probabilities])

        return actor, critic, policy
    
    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        probabilities = self.policy.predict_on_batch(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action
    
    def update_network(self, state, action, reward, new_state, done):
        state = np.reshape(state, [1, self.state_size])
        new_state = np.reshape(new_state, [1, self.state_size])

        critic_value = self.critic.predict_on_batch(state)
        new_critic_value = self.critic.predict_on_batch(new_state)

        target = reward + self.gamma * new_critic_value * (1 - int(done))
        delta = target - critic_value

        action_one_hot = np.zeros([1, self.action_size])
        action_one_hot[np.arange(1), action] = 1.0

        self.actor.fit([state, delta], action_one_hot, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)

    def train(self, n_episodes, max_steps):
        for episode in range(n_episodes):
            state = self.env.reset()
            state = state[0]
            episode_rewards = 0
            episode_steps = 0

            for _ in range(max_steps):
                action = self.act(state)
                new_state, reward, done, _, _ = self.env.step(action)
                self.update_network(state, action, reward, new_state, done)
                
                state = new_state
                episode_rewards += reward
                episode_steps += 1

                if done:
                    break

            print(f'Episode: {episode}, Reward: {episode_rewards}, Steps: {episode_steps}')

        self.env.close()