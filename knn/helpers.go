package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
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

	var originalSet [][]string
	for _, row := range rows {
		for i := range row {
			row[i] = strings.TrimSpace(row[i])
		}
		originalSet = append(originalSet, row)
	}

	model.NormaliseData(originalSet)

	trainingClasses, testingClasses := splitSet(model.normaliseSet, model.Split)
	model.trainingSet = trainingClasses
	model.testingSet = testingClasses
	fmt.Println("Split successful...")
	fmt.Printf("Tranining Set - %v, Testing Set - %v\n", len(model.trainingSet), len(model.testingSet))
	return nil
}

func splitSet(originalData [][]float64, split float32) ([][]float64, [][]float64) {
	shuffledData := make([][]float64, len(originalData))
	shuffleInstance := rand.Perm(len(originalData))

	for i, v := range shuffleInstance {
		shuffledData[v] = originalData[i]
	}

	trainingIndex := int(split * float32(len(originalData)))

	return shuffledData[:trainingIndex], shuffledData[trainingIndex:]
}

func getEuclideanDistance(instance1, instance2 []float64, start, end int) float64 {
	var distance float64
	for i := start; i <= end; i++ {
		x := instance1[i]
		y := instance2[i]
		distance += math.Pow(x-y, 2)
	}

	return math.Sqrt(distance)
}

func getExpectedClass(model *KNNModel, testInstance []float64) string {
	var neighbours Neighbours

	// calculate distance from all the training instances
	for _, instance := range model.trainingSet {
		distance := getEuclideanDistance(
			instance, testInstance, model.AttributeIndexStart, model.AttributeIndexEnd)
		neighbour := Neighbour{
			Distance: distance,
			Class:    model.GetClassString(instance[model.ClassIndex]),
		}
		neighbours = append(neighbours, neighbour)
	}

	// sort the distances result in ascending order based on distances
	sort.Sort(neighbours)

	// pick the instances based on K
	selectedInstances := neighbours[:model.K]

	// pick the top rated
	if len(selectedInstances) == 1 {
		return selectedInstances[0].Class
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

	return winningClass
}

func Benchmark(model *KNNModel) (BenchmarkResponse, error) {

	// load data first
	err := loadData(model)
	if err != nil {
		return BenchmarkResponse{}, err
	}

	var correctPredictions int
	for _, instance := range model.testingSet {
		class := getExpectedClass(model, instance)
		actualClass := model.GetClassString(instance[model.ClassIndex])
		fmt.Printf("Predicted - %v, Actual - %v\n", class, actualClass)
		if class == actualClass {
			correctPredictions += 1
		}
	}

	fmt.Printf("Predicted %v correctly from %v instances\n", correctPredictions, len(model.testingSet))
	accuracy := float32(correctPredictions) / float32(len(model.testingSet))

	return BenchmarkResponse{Accuracy: accuracy, K: model.K}, nil
}
