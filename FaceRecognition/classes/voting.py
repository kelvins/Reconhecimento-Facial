
class Voting(object):
    """
    Class the provides voting methods for the ensemble.
    """

    __majority, __weighted = range(2)

    def __init__(self, voting_scheme=__majority, weights=list()):
        """
        Define the selected voting scheme (default is majoritary).
        Set the weights (default is an empty list).
        :param voting_scheme: Define the voting scheme (MAJORITY or WEIGHTED).
        :param weights: When using the WEIGHTED voting scheme, this is used to set the weights.
        """
        self.voting_scheme = voting_scheme
        self.weights = weights

    @property
    def majority(self):
        return self.__majority

    @property
    def weighted(self):
        return self.__weighted

    def get_voting_scheme_name(self):
        """
        Get the name of the selected voting scheme to be used in the report.
        """
        if self.voting_scheme == self.__majority:
            return "Majority Voting"
        elif self.voting_scheme == self.__weighted:
            return "Weighted Voting"

        raise NameError("Invalid voting scheme!")
        return ""

    def vote(self, subjects, weights=list()):
        """
        Call the selected voting scheme.
        :param subjects: The predicted subjects list.
        :param weights: The weights list.
        :return: The subject voted by the voting scheme.
        """
        if self.voting_scheme == self.__weighted:
            if not weights:
                weights = self.weights
            return Voting.weighted_voting(subjects, weights)
        elif self.voting_scheme == self.__majority:
            return Voting.majority_voting(subjects)

        raise NameError("Invalid voting scheme!")
        return -1

    @staticmethod
    def majority_voting(subjects):
        """
        Vote using the majority scheme (it does not use the weights list).
        :param subjects: The predicted subjects list.
        :return: The subject voted by the voting scheme.
        """

        if len(subjects) == 0:
            return -1

        subject_voted = []
        number_of_votes = []

        # Count votes
        for i in range(0, len(subjects)):
            if subjects[i] in subject_voted:
                index = subject_voted.index(subjects[i])
                number_of_votes[index] = number_of_votes[index] + 1
            else:
                subject_voted.append(subjects[i])
                number_of_votes.append(1)

        index_max_voted = number_of_votes.index(max(number_of_votes))
        return subject_voted[index_max_voted]

    @staticmethod
    def weighted_voting(subjects, weights):
        """
        Vote using the weighted scheme.
        :param subjects: The predicted subjects list.
        :param weights: The weights list.
        :return: The subject voted by the voting scheme.
        """

        if len(subjects) == 0 or len(weights) == 0:
            return -1

        if len(subjects) != len(weights):
            return -1

        subject_voted = []
        value_of_votes = []

        # Count votes
        for i in range(0, len(subjects)):
            if subjects[i] in subject_voted:
                index = subject_voted.index(subjects[i])
                value_of_votes[index] = value_of_votes[index] + weights[i]
            else:
                subject_voted.append(subjects[i])
                value_of_votes.append(weights[i])

        max_value = max(value_of_votes)

        # If we have duplicate values then we have a tie
        if len(value_of_votes) != len(set(value_of_votes)):
            most_voted = []
            # Create a list of the subjects that have more votes
            for i in range(0, len(subject_voted)):
                if value_of_votes[i] == max_value:
                    most_voted.append(subject_voted[i])

            temp_subjects = []
            # Create a new list of the subjects
            for i in range(0, len(subjects)):
                if subjects[i] in most_voted:
                    temp_subjects.append(subjects[i])

            return Voting.majority_voting(temp_subjects)
        else:
            index_max_voted = value_of_votes.index(max_value)
            return subject_voted[index_max_voted]
