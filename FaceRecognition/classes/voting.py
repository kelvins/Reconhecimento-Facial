
class Voting:
    """
    Class the provides voting methods for the ensemble.
    """

    MAJORITY, WEIGHTED = range(2)

    def __init__(self, votingScheme=MAJORITY, weights=[]):
        """
        Define the selected voting scheme (default is majoritary)
        Set the weights (default is an empty list)
        """
        self.votingScheme = votingScheme
        self.weights = weights

    def setWeights(self, weights):
        """
        Set the weights
        """
        self.weights = weights

    def getWeights(self):
        """
        Get the weights
        """
        return self.weights

    def setVotingScheme(self, votingScheme):
        """
        Set the selected voting scheme
        """
        self.votingScheme = votingScheme

    def getVotingScheme(self):
        """
        Get the selected voting scheme
        """
        return self.votingScheme

    def getVotingSchemeName(self):
        """
        Get the name of the selected voting scheme to be used in the report.
        """
        if self.votingScheme == self.MAJORITY:
            return "Majority Voting"
        elif self.votingScheme == self.WEIGHTED:
            return "Weighted Voting"
        return ""

    def vote(self, subjects, weights=[]):
        """
        Call the selected voting scheme
        """
        if self.votingScheme == self.WEIGHTED:
            if not weights:
                weights = self.weights
            return self.weightedVoting(subjects, weights)
        else:
            return self.majorityVoting(subjects)

    def majorityVoting(self, subjects):
        """
        Majority voting.
        Return -1 for empty list
        """

        if len(subjects) == 0:
            return -1

        subjectVoted = []
        numberOfVotes = []

        # Count votes
        for i in range(0, len(subjects)):
            if subjects[i] in subjectVoted:
                index = subjectVoted.index(subjects[i])
                numberOfVotes[index] = numberOfVotes[index] + 1
            else:
                subjectVoted.append(subjects[i])
                numberOfVotes.append(1)

        indexMaxVoted = numberOfVotes.index(max(numberOfVotes))
        return subjectVoted[indexMaxVoted]

    def weightedVoting(self, subjects, weights):
        """
        Weighted voting.
        """

        if len(subjects) == 0 or len(weights) == 0:
            return -1

        if len(subjects) != len(weights):
            return -1

        subjectVoted = []
        valueOfVotes = []

        # Count votes
        for i in range(0, len(subjects)):
            if subjects[i] in subjectVoted:
                index = subjectVoted.index(subjects[i])
                valueOfVotes[index] = valueOfVotes[index] + weights[i]
            else:
                subjectVoted.append(subjects[i])
                valueOfVotes.append(weights[i])

        maxValue = max(valueOfVotes)

        # If we have duplicate values then we have a tie
        if len(valueOfVotes) != len(set(valueOfVotes)):
            mostVoted = []
            # Create a list of the subjects that have more votes
            for i in range(0, len(subjectVoted)):
                if valueOfVotes[i] == maxValue:
                    mostVoted.append(subjectVoted[i])

            tempSubjects = []
            # Create a new list of the subjects
            for i in range(0, len(subjects)):
                if subjects[i] in mostVoted:
                    tempSubjects.append(subjects[i])

            return self.majorityVoting(tempSubjects)
        else:
            indexMaxVoted = valueOfVotes.index(maxValue)
            return subjectVoted[indexMaxVoted]
