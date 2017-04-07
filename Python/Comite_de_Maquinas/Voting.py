
class Voting:
    """
    Class the provides voting methods for the committee machine.
    """

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

        indexMaxVoted = valueOfVotes.index(max(valueOfVotes))
        return subjectVoted[indexMaxVoted]
