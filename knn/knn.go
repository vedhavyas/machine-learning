package knn

import (
	"encoding/csv"
	"math/rand"
	"os"
	"time"
)

type KNNModel struct {
	Split           float32
	originalClasses [][]string
	trainingClasses [][]string
	testingClasses  [][]string
	FileName        string
}

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

func loadData(model *KNNModel) error {
	fh, err := os.Open(model.FileName)
	if err != nil {
		return err
	}

	rows, err := csv.NewReader(fh).ReadAll()
	if err != nil {
		return err
	}

	for _, row := range rows {
		model.originalClasses = append(model.originalClasses, row)
	}

	trainingClasses, testingClasses := splitClasses(model.originalClasses, model.Split)
	model.trainingClasses = trainingClasses
	model.testingClasses = testingClasses
	return nil
}

func splitClasses(originalData [][]string, split float32) ([][]string, [][]string) {
	shuffledData := make([][]string, len(originalData))
	shuffleInstance := rand.Perm(len(originalData))

	for i, v := range shuffleInstance {
		shuffledData[v] = originalData[i]
	}

	trainingIndex := int(split * float32(len(originalData)))

	return shuffledData[:trainingIndex], shuffledData[trainingIndex:]
}
