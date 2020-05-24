class State():
    """
    Class which represents the current state of Pacman, contains
    position, ghost position and food position
    """
    def __init__(self,state):
        """
        Constructor method for the state class, converts all atributes
        into tuples so they are immutable.

        Keyword Arguments
            state - The current game state of the pacman game
        
        """
        # extract the state information from the current game state
        self.position = tuple(state.getPacmanPosition())
        self.ghost_positions = tuple(state.getGhostPositions())
        self.food_positions = tuple(tuple(x) for x in state.getFood())

    def __hash__(self):
        """
        Allows states to be keys of dictionaries.

        Returns
            hash - The hash of the tuple of atributes
        """
        return hash((self.position, self.ghost_positions, self.food_positions))

    def __eq__(self, other):
        """
        Allows two states to be compared.

        Keyword Argument
            other - The state to compare 

        Returns
            boolean - Whether the other state is equal or not
        """
        if other == None: return False
        return (self.position, self.ghost_positions, self.food_positions) == (other.position, other.ghost_positions, other.food_positions)
