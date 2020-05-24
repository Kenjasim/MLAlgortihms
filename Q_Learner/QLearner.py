import State
import StateActionPair
import random
class QLearner():
    """
    Class which impliments the q learing algorithm. The q table is
    initialised and explored depending on the epsilon variable.
    """
    
    def __init__(self, alpha, epsilon, gamma):
        """
        Constructor method for the q learning class. Simply
        initialises variables

        Keyword Arguments
            alpha   - learning rate
            epsilon - exploration rate
            gamma   - discount factor
            action  - The action to be applied
        """
        # Initialise training parameters
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)

        # Initialise previous state, action and reward
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = None

        # Initialse the Q table 
        self.q_table = {}

    def pick_action (self, state, legal):
        """
        Runs through the Q-learning process and returns an
        action for pacman to take. It will either be the best
        action (exploit) or a random action (explore)

        Keyword Arguments
            state   - The gamestate of the pacman game
            legal   - List of legal actions

        Returns
            action  - The action which will be taken
        """

        # Define the current state
        curr_state = State.State(state)

        # intialise the space in q learner
        self.initialise_space(curr_state, legal)

        if (self.previous_state == None):

            # pick a random action
            action = random.choice(legal)

            # collect the reward
            reward = state.getScore()

            # Set the previous state, reward and actions
            self.previous_state = curr_state
            self.previous_action = action
            self.previous_reward = reward

            return action

        else:
            # Define the state-action pair
            sa = StateActionPair.StateActionPair(self.previous_state, self.previous_action)

            # Calculate the score
            score = state.getScore()
            
            # Update the q score
            self.update_q_score(sa, legal, curr_state, score)

            # Return an action
            action = self.epsilon_policy(curr_state, legal)

            # Update the previous state, action and reward
            self.previous_state = curr_state
            self.previous_action = action
            self.previous_score = score

            return action


    def initialise_space(self, state, legal):
        """
        Initialises the state action pairs within the
        Q table to 0

        Keyword Arguments
            state   - The gamestate of the pacman game
            legal   - List of legal actions
        """
        # loop through all the legal actions 
        for a in legal:
            sa = StateActionPair.StateActionPair(state, a)
            # If the state has not been visited then add
            # to q table
            if sa not in self.q_table.keys():
                self.q_table[sa] = 0

    def update_q_score(self, sa, legal, curr_state, score):
        """
        Updates the q score of a state-action pair

        Keyword Arguments
            sa           - The previous state action pair
            legal        - List of legal actions
            curr_state   - The current state of pacman
            score        - The current score of the game
        """
        # initialise  new state action pairs list
        new_sas = []
        # Loop through all new states action pairs
        # for Q score update
        for a in legal:
            # Append the new q score minus the old score 
            # to the list
            new_sa = StateActionPair.StateActionPair(curr_state, a)
            new_sas.append(self.q_table[new_sa] - self.q_table[sa])

        # Calculate the q score update
        q_score = self.q_table[sa] + self.alpha * (score + (self.gamma * max(new_sas)))

        # Update the q_score
        self.q_table[sa] = q_score

    def epsilon_policy(self, state, legal):
        """
        Defines the epsilon policy to choose the actions to take

        Keyword Arguments
            state   - The gamestate of the pacman game
            legal   - List of legal actions
        
        Returns
            action  - The action which should be taken
        """
        if random.uniform(0, 1) < self.epsilon and self.epsilon > 0:
            # If less than epsilon select random action (explore)
            action = random.choice(legal)
        else:
            # Select action with best q score (exploit)
            action = self.argmax(state, legal)
        
        return action

    def argmax(self, state, legal):
        """
        Finds the action with the best q score

        Keyword Arguments
            state   - The gamestate of the pacman game
            legal   - List of legal actions
        
        Returns
            best_action  - The action with the best q score
        """
        best_q_score = -10000000000
        best_action = None

        # Loop through each action
        for a in legal:
            # Loop through each pair
            sa = StateActionPair.StateActionPair(state, a)
            q_score = self.q_table[sa]

            # If the score is better than previous then return action
            if q_score > best_q_score:
                best_q_score = q_score
                best_action = sa.get_action()

        # If no state is found then return random state
        if best_action == None:
            best_action = random.choice(legal)
        
        return best_action

    def end_game(self, state):
        """
        Updates the q score at the end of ths game

        Keyword Arguments
            state   - The gamestate of the pacman game
        """
        # Update the states when a game has finished
        sa = StateActionPair.StateActionPair(self.previous_state, self.previous_action)
        # Assign the endgame score
        self.q_table[sa] = state.getScore()

    def update_parameters(self,alpha, epsilon):
        """
        Updates the parameters of the Q learner

        Keyword Arguments
            alpha   - learning rate
            epsilon - exploration rate
        """
        # change parameters
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)