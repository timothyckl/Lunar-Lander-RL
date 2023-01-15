class Fitness:
    def __init__(self, model, env, solution):
        self.model = model
        self.env = env
        self.solution = solution
        self.fitness = 0

    def __call__(self):
        '''
        Compute fitness for a single model.
        '''
        self.model.set_weights(reshape_weights(self.model, solution))
        state_size = self.env.observation_space.shape[0]

        for _ in range(1):
            state = self.env.reset()
            state = state[0].reshape(1, state_size)
            done = False

            while not done:
                action = np.argmax(self.model.predict(state, verbose=0))
                state, reward, done, _, _ = self.env.step(action)
                state = state.reshape(1, state_size)
                self.fitness += reward