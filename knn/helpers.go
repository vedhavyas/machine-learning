package knn

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
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

func getEuclideanDistance(instance1, instance2 []float64, classIndex int) float64 {
	var distance float64
	for i := 0; i < len(instance1); i++ {
		if i == classIndex {
			continue
		}
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
		distance := getEuclideanDistance(instance, testInstance, model.ClassIndex)
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

func ExecuteKNN(model *KNNModel) error {

	// load data first
	err := model.PrepareModel()
	if err != nil {
		return err
	}

	trainingClasses, testingClasses := splitSet(model.Data, model.Split)
	model.trainingSet = trainingClasses
	model.testingSet = testingClasses
	fmt.Println("Split successful...")
	fmt.Printf("Tranining Set - %v, Testing Set - %v\n", len(model.trainingSet), len(model.testingSet))

	var correctPredictions int
	for _, instance := range model.testingSet {
		class := getExpectedClass(model, instance)
		actualClass := model.GetClassString(instance[model.ClassIndex])
		if class == actualClass {
			correctPredictions += 1
		}
	}

	fmt.Printf("Predicted %v correctly from %v instances\n", correctPredictions, len(model.testingSet))
	accuracy := float32(correctPredictions) / float32(len(model.testingSet))
	fmt.Printf("Accuracy: %v\n", accuracy*float32(100))

	return nil
}
