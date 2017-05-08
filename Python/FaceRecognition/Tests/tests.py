import sys
import unittest

sys.path.append('../Voting')
from Voting import Voting

voting = Voting()

class MajorityVotingTest(unittest.TestCase):
    def test1(self):
        subjects = [1, 5, 2, 1, 1]
        self.assertEqual(voting.majorityVoting(subjects), 1)

    def test2(self):
        subjects = [4, 4, 3, 3, 3]
        self.assertEqual(voting.majorityVoting(subjects), 3)

    def test3(self):
        subjects = [1, 4, 5, 2, 2]
        self.assertEqual(voting.majorityVoting(subjects), 2)

    def test4(self):
        subjects = [5, 1, 5, 3, 3]
        self.assertEqual(voting.majorityVoting(subjects), 5)

    def test5(self):
        subjects = [0]
        self.assertEqual(voting.majorityVoting(subjects), 0)

    def test6(self):
        subjects = []
        self.assertEqual(voting.majorityVoting(subjects), -1)

class WeightedVotingTest(unittest.TestCase):
    def test1(self):
        subjects = [1, 5, 2, 1, 1]
        weights  = [25, 30, 10, 15, 20]
        self.assertEqual(voting.weightedVoting(subjects, weights), 1)

    def test2(self):
        subjects = [4, 4, 3, 3, 3]
        weights  = [25, 30, 10, 15, 20]
        self.assertEqual(voting.weightedVoting(subjects, weights), 4)

    def test3(self):
        subjects = [1, 4, 5, 2, 2]
        weights  = [25, 30, 10, 15, 20]
        self.assertEqual(voting.weightedVoting(subjects, weights), 2)

    def test4(self):
        subjects = [2, 3, 5, 4, 5]
        weights  = [25, 30, 10, 15, 20]
        self.assertEqual(voting.weightedVoting(subjects, weights), 5)

    def test5(self):
        subjects = [1, 2, 1, 3, 3]
        weights  = [25, 30, 10, 15, 20]
        self.assertEqual(voting.weightedVoting(subjects, weights), 1)

    def test6(self):
        subjects = []
        weights  = [25, 30, 10, 15, 20]
        self.assertEqual(voting.weightedVoting(subjects, weights), -1)

    def test7(self):
        subjects = [1, 2, 1, 3, 3]
        weights  = []
        self.assertEqual(voting.weightedVoting(subjects, weights), -1)

    def test8(self):
        subjects = []
        weights  = []
        self.assertEqual(voting.weightedVoting(subjects, weights), -1)

    def test9(self):
        subjects = [1]
        weights  = [1, 2]
        self.assertEqual(voting.weightedVoting(subjects, weights), -1)

    def test10(self):
        subjects = [1, 2]
        weights  = [1]
        self.assertEqual(voting.weightedVoting(subjects, weights), -1)

    def test11(self):
        subjects = [0]
        weights  = [0]
        self.assertEqual(voting.weightedVoting(subjects, weights), 0)

if __name__ == '__main__':
    unittest.main()
