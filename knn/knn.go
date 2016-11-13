package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"time"
)

type KNNModel struct {
	Split       float32
	OriginalSet [][]string
	TrainingSet [][]string
	TestingSet  [][]string
	FileName    string
	K           int
}

type BenchmarkResponse struct {
	K        int
	Accuracy float64
}

func (res BenchmarkResponse) String() string {
	return fmt.Sprintf(
		"K: %v\nAccuracy: %v",
		res.K,
		res.Accuracy,
	)
}

type DistanceResult struct {
	Distance float64
	Class    string
}

type DistanceResults []DistanceResult

func (r DistanceResults) Len() int {
	return len(r)
}

func (r DistanceResults) Less(i, j int) bool {
	return r[i].Distance <= r[j].Distance
}

func (r DistanceResults) Swap(i, j int) {
	r[i], r[j] = r[j], r[i]
}

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
	log.SetFlags(log.Lshortfile | log.Ltime)
}

func loadData(model *KNNModel) error {
	log.Printf("Loading data from '%v'...\n", model.FileName)
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

	trainingClasses, testingClasses := splitSet(model.OriginalSet, model.Split)
	model.TrainingSet = trainingClasses
	model.TestingSet = testingClasses
	log.Println("Split successful...")
	log.Printf("Tranining Set - %v, Testing Set - %v\n", len(model.TrainingSet), len(model.TestingSet))
	return nil
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
	var distanceResults DistanceResults

	// calculate distance from all the training instances
	for _, instance := range model.TrainingSet {
		distance, err := getEuclideanDistance(instance, testInstance, 4)
		if err != nil {
			return "", err
		}
		distanceResults = append(distanceResults,
			DistanceResult{
				Distance: distance,
				Class:    instance[4],
			})
	}

	// sort the distances result in ascending order based on distances
	sort.Sort(distanceResults)

	// pick the instances based on K
	selectedInstances := distanceResults[:model.K]

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

		if class == instance[4] {
			correctPredictions += 1
		}
	}

	accuracy := float64(correctPredictions) / float64(len(model.TestingSet))

	return BenchmarkResponse{Accuracy: accuracy, K: model.K}, nil
}
