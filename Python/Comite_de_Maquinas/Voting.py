
class Voting:
    """
    Class the provides voting methods for the committee machine.
    """

    def majorityVoting(subjects):
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
            index = subjectVoted.find(subjects[i])
            if index >= 0:
                numberOfVotes[index] = numberOfVotes[index] + 1
            else:
                subjectVoted.append(subjects[i])
                numberOfVotes.append(1)

        indexMaxVoted = numberOfVotes.find(max(numberOfVotes))
        return subjectVoted[indexMaxVoted]

    def weightedVoting(subjects, weights):
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
            index = subjectVoted.find(subjects[i])
            if index >= 0:
                valueOfVotes[index] = valueOfVotes[index] + weights[i]
            else:
                subjectVoted.append(subjects[i])
                valueOfVotes.append(weights[i])

        indexMaxVoted = valueOfVotes.find(max(valueOfVotes))
        return subjectVoted[indexMaxVoted]
