class StateActionPair():
    """
    Class which represents a state-action pair, will be used
    as the key to the dictionary
    """
    def __init__(self, state, action):
        """
        Constructor method for the state-action pair class

        Keyword Arguments
            state - The current state of the pacman game
            action - The action to be applied
        
        """
        # Store the state and actions here
        self.state = state
        self.action = action

    def get_action(self):
        """
        Getter method for the action variable

        Returns
            action - The action to be applied to the state
        
        """
        return self.action

    def __hash__(self):
        """
        Allows state-action pairs to be keys of dictionaries.

        Returns
            hash - The hash of the tuple of atributes
        """
        return hash((self.state, self.action))

    def __eq__(self, other):
        """
        Allows two state-action pairs to be compared.

        Keyword Argument
            other - The state-action pair to compare 

        Returns
            boolean - Whether the other state-action pair is equal or not
        """
        return (self.state, self.action) == (other.state, other.action)
