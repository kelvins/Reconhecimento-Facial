
class Voting:
    """
    Class the provides voting methods for the committee machine.
    """

    MAJORITY, WEIGHTED = range(2)

    def __init__(self, votingScheme=MAJORITY, weights=[]):
        """
        Define the selected voting scheme (default is majoritary)
        Set the weights (default is an empty list)
        """
        self.votingScheme = votingScheme
        self.weights = weights

    def getVotingSchemeName(self):
        """
        Get the name of the selected voting scheme to be used in the report.
        """
        if self.votingScheme == MAJORITY:
            return "Majority Voting"
        elif self.votingScheme == WEIGHTED:
            return "Weighted Voting"
        return ""

    def vote(self, subjects, weights=[]):
        """
        Call the selected voting scheme
        """
        if self.votingScheme == WEIGHTED:
            if len(weights) is None:
                weights = self.weights
            return weightedVoting(subjects, weights)
        else:
            return majorityVoting(subjects)

    def majorityVoting(self, subjects):
        """
        Majority voting.
        """

        if len(subjects) == 0:
            print "The list is empty."
            return -1

        subjectVoted  = []
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
            print "The list is empty."
            return -1

        if len(subjects) != len(weights):
            print "The two lists must have the same size."
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
