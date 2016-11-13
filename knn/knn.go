package knn

import (
	"encoding/csv"
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

type DistanceResult struct {
	Distance float64
	Class    string
}

type DistanceResults []DistanceResult

func (r *DistanceResults) Len() int {
	return len(r)
}

func (r *DistanceResults) Less(i, j int) bool {
	return r[i].Distance <= r[j].Distance
}

func (r *DistanceResults) Swap(i, j int) {
	r[i], r[j] = r[j], r[i]
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
		model.OriginalSet = append(model.OriginalSet, row)
	}

	trainingClasses, testingClasses := splitSet(model.OriginalSet, model.Split)
	model.TrainingSet = trainingClasses
	model.TestingSet = testingClasses
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
	for i := 0; i <= length; i++ {
		x, err := strconv.ParseFloat(instance1[i], 64)
		if err != nil {
			return err
		}
		y, err := strconv.ParseFloat(instance2[i], 64)
		if err != nil {
			return err
		}
		distance += math.Pow(x-y, 2)
	}

	return math.Sqrt(distance)
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

	return winningClass, nil
}

func Benchmark(model **KNNModel) (BenchmarkResponse, error) {

	// load data first
	err := loadData(model)
	if err != nil {
		return BenchmarkResponse{}, err
	}

}
