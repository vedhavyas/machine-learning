package knn

import "testing"

func Test_splitClasses(t *testing.T) {
	cases := []struct {
		Spilt float32
		Data  [][]string
	}{
		{
			Spilt: 0.20,
			Data:  [][]string{{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}},
		},

		{
			Spilt: 0.30,
			Data:  [][]string{{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}},
		},

		{
			Spilt: 0.40,
			Data:  [][]string{{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}},
		},

		{
			Spilt: 0.50,
			Data:  [][]string{{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}},
		},

		{
			Spilt: 0.50,
			Data:  [][]string{{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}},
		},

		{
			Spilt: 0.60,
			Data:  [][]string{{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}},
		},

		{
			Spilt: 0.70,
			Data:  [][]string{{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}},
		},

		{
			Spilt: 0.80,
			Data:  [][]string{{"a"}, {"b"}, {"c"}, {"d"}, {"e"}, {"f"}, {"g"}},
		},
	}

	for _, testCase := range cases {
		trainingClasses, testingClasses := splitSet(testCase.Data, testCase.Spilt)
		expectedSplit := int(testCase.Spilt * float32(len(testCase.Data)))

		if len(trainingClasses)+len(testingClasses) != len(testCase.Data) {
			t.Fatalf("Length mismatch - Expected %v, Result %v",
				len(testCase.Data), len(trainingClasses)+len(testingClasses))
		}

		if len(trainingClasses) != expectedSplit {
			t.Fatalf("Training - Expected %v,  Result %v",
				expectedSplit, len(trainingClasses))
		}

		if len(testingClasses) != len(testCase.Data)-expectedSplit {
			t.Fatalf("testing - Expected %v,  Result %v",
				len(testCase.Data)-expectedSplit, len(testingClasses))
		}
	}

}
