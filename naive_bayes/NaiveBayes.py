import numpy as np

class NaiveBayes():
    """ Implimentation of the naive bayes classifier 
        It takes the data and calculates the priors,
        when it gets a feature vector it calculates 
        the probability of it belonging in a class and
        returns the most likely one. 
    """
    
    def fit(self, data, targets):
        """ Fit the data to a naive bayes classifier

        Keyword Arguments
        data - the arrays which describe the pacman scene
        target - the move associated with each array
        """
        # Find no of samples and features
        data = np.asarray(data)
        targets = np.asarray(targets)
        self.num_samples, self.num_features = data.shape

        # Find the amount of classes
        self.classes = np.unique(targets)
        num_classes = len(self.classes)
 
        # Create an empty array for the priors
        self.priors = np.zeros((num_classes), dtype=np.float64)
        self.features = []


        # Loop through each class and calculate the priors and append all the 
        # classes feature arrays.
        for c in self.classes:
            data_c = data[c == targets]
            self.priors[c] = data_c.shape[0] / float(self.num_samples)
            self.features.append(data_c)
                
    def predict(self, data):
        """ Predict the class of an array of features

        Keyword Arguments
        data - the arrays which describe the pacman scene

        Returns
        target - the class which the features belong to
        """
        # run through all the classes and calculate the conditional probability
        posteriors = []
        
        #Go over each class, calculate the prior, class posterior and posterior.
        for index, _ in enumerate(self.classes):
            prior = self.priors[index]
            probabilities = []
            for idx, val in enumerate(data):
                probabilities.append(self.calculate_probabilities(index, idx, val))
            class_probability = np.sum(np.log(probabilities))
            posteriors.append(np.log(prior) + class_probability)

        #Return the argmax of the posterior
        # print posteriors
        return self.classes[np.argmax(posteriors)]


    def calculate_probabilities(self, target, index, val):
        """ Calculates the probability of each feature value 
            given a class

        Keyword Arguments
        target - the class which is being looped over
        index - the feature given by an index
        val -  the value of the feature

        Returns
        probability - the probability, given a feature, that it 
                      belongs to the class
        """
        #Loop through all the features in that target
        positives = 0
        for feat in self.features[target]:
            # num_same = (feat[index] == val).sum()
            if feat[index] == val:
                positives = positives + 1
        
        numerator = float(positives) + ((len(self.features[target]))*(float(1)/2))
        denom = len(self.features[target]) + len(self.features[target])
        return float(numerator) / denom