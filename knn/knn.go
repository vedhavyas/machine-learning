package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"time"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

func loadData(model *KNNModel) error {
	fmt.Printf("Loading data from '%v'...\n", model.FileName)
	fh, err := os.Open(model.FileName)
	if err != nil {
		return err
	}

	rows, err := csv.NewReader(fh).ReadAll()
	if err != nil {
		return err
	}

	for _, row := range rows {
		model.OriginalSet = append(model.OriginalSet, row)
	}

	if model.AttributeNormalisation {
		normaliseData(model)
	}

	trainingClasses, testingClasses := splitSet(model.OriginalSet, model.Split)
	model.TrainingSet = trainingClasses
	model.TestingSet = testingClasses
	fmt.Println("Split successful...")
	fmt.Printf("Tranining Set - %v, Testing Set - %v\n", len(model.TrainingSet), len(model.TestingSet))
	return nil
}

func normaliseData(model *KNNModel) {
	// nothing happening here
}

func splitSet(originalData [][]string, split float32) ([][]string, [][]string) {
	shuffledData := make([][]string, len(originalData))
	shuffleInstance := rand.Perm(len(originalData))

	for i, v := range shuffleInstance {
		shuffledData[v] = originalData[i]
	}

	trainingIndex := int(split * float32(len(originalData)))

	return shuffledData[:trainingIndex], shuffledData[trainingIndex:]
}

func getEuclideanDistance(instance1, instance2 []string, length int) (float64, error) {
	var distance float64
	for i := 0; i < length; i++ {
		x, err := strconv.ParseFloat(instance1[i], 64)
		if err != nil {
			return distance, err
		}
		y, err := strconv.ParseFloat(instance2[i], 64)
		if err != nil {
			return distance, err
		}
		distance += math.Pow(x-y, 2)
	}

	return math.Sqrt(distance), nil
}

func getExpectedClass(model *KNNModel, testInstance []string) (string, error) {
	var neighbours Neighbours

	// calculate distance from all the training instances
	for _, instance := range model.TrainingSet {
		distance, err := getEuclideanDistance(instance, testInstance, model.AttributeIndexRange)
		if err != nil {
			return "", err
		}
		neighbours = append(neighbours,
			Neighbour{
				Distance: distance,
				Class:    instance[model.ClassIndex],
			})
	}

	// sort the distances result in ascending order based on distances
	sort.Sort(neighbours)

	// pick the instances based on K
	selectedInstances := neighbours[:model.K]

	// pick the top rated
	if len(selectedInstances) == 1 {
		return selectedInstances[0].Class, nil
	}

	election := make(map[string]int)
	for _, instance := range selectedInstances {
		election[instance.Class] += 1
	}

	var winningClass string
	var winningCount int
	for k, v := range election {
		// if there is tie we do nothing
		if winningCount >= v {
			continue
		}

		winningCount = v
		winningClass = k
	}

	return winningClass, nil
}

func Benchmark(model *KNNModel) (BenchmarkResponse, error) {

	// load data first
	err := loadData(model)
	if err != nil {
		return BenchmarkResponse{}, err
	}

	var correctPredictions int
	for _, instance := range model.TestingSet {
		class, err := getExpectedClass(model, instance)
		if err != nil {
			return BenchmarkResponse{}, err
		}
		fmt.Printf("Predicted - %v, Actual - %v\n", class, instance[model.ClassIndex])
		if class == instance[model.ClassIndex] {
			correctPredictions += 1
		}
	}

	fmt.Printf("Predicted %v correctly from %v instances\n", correctPredictions, len(model.TestingSet))
	accuracy := float32(correctPredictions) / float32(len(model.TestingSet))

	return BenchmarkResponse{Accuracy: accuracy, K: model.K}, nil
}
