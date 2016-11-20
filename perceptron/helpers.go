package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

func getTrainAndTestData(data [][]float64, k int) map[int][][][]float64 {
	indexes := rand.Perm(len(data))
	shuffledData := make([][]float64, len(data))
	for i, k := range indexes {
		shuffledData[i] = data[k]
	}

	distribution := len(data) / k
	folds := make([][][]float64, k)
	var index int
	for i := 0; i < k; i++ {
		if i == k-1 {
			folds[i] = shuffledData[index:]
			continue
		}

		folds[i] = shuffledData[index : index+distribution]
		index += distribution
	}

	splitData := make(map[int][][][]float64)

	for i := 0; i < k; i++ {
		splitData[i] = getSets(folds, k, i)
	}

	return splitData
}

func getSets(folds [][][]float64, k, index int) [][][]float64 {
	testData := append([][]float64(nil), folds[index]...)
	var trainData [][]float64
	for i := 0; i < k; i++ {
		if i == index {
			continue
		}

		trainData = append(trainData, folds[i]...)
	}

	return [][][]float64{
		trainData,
		testData,
	}
}

func predict(data, weights []float64, classIndex int) float64 {
	activation := weights[classIndex]
	for index, attr := range data {
		if index == classIndex {
			continue
		}
		activation += weights[index] * attr
	}

	if activation >= 0.00 {
		return 1.00
	}

	return 0.00
}

func trainWeights(trainData [][]float64, learningRate float64, epoch, classIndex int) []float64 {

	weights := make([]float64, len(trainData[0]))

	for i := 0; i < epoch; i++ {
		for _, row := range trainData {
			result := predict(row, weights, classIndex)

			predictionError := row[classIndex] - result
			weights[classIndex] += learningRate * predictionError

			for column, attr := range row {
				if column == classIndex {
					continue
				}

				weights[column] += learningRate * predictionError * attr
			}

		}
	}

	return weights
}

func executeSet(ID int, wg *sync.WaitGroup, resultCh chan<- float64, trainingSet, testingSet [][]float64, learningRate float64, epoch, classIndex int) {
	fmt.Printf("%v starting with training %v and testing %v\n", ID, len(trainingSet), len(testingSet))
	weights := trainWeights(trainingSet, learningRate, epoch, classIndex)
	predictions := make([]float64, len(testingSet))
	for index, row := range testingSet {
		predictions[index] = predict(row, weights, classIndex)
	}

	var correctPredictions int
	for index, predicted := range predictions {
		expected := testingSet[index][classIndex]
		if expected == predicted {
			correctPredictions += 1
		}
	}
	accuracy := (float64(correctPredictions) / float64(len(testingSet))) * 100
	fmt.Printf("Id %v, predicted %v, total %v, accuracy %v\n", ID, correctPredictions, len(testingSet), accuracy)
	resultCh <- accuracy
	wg.Done()
}

func ExecutePerceptron(model *PerceptronModel) error {
	err := model.loadData()
	if err != nil {
		return err
	}

	splitData := getTrainAndTestData(model.data, model.KFold)
	wg := new(sync.WaitGroup)
	wg.Add(model.KFold)
	resultCh := make(chan float64)
	doneCh := make(chan bool)

	go func(doneCh <-chan bool, resultCh <-chan float64, k int) {
		var totalAccuracy float64
		for {
			select {
			case result := <-resultCh:
				totalAccuracy += result
			case <-doneCh:
				meanAccuracy := totalAccuracy / float64(k)
				fmt.Printf("mean accuracy %v\n", meanAccuracy)
				return
			}
		}
	}(doneCh, resultCh, model.KFold)

	for k, v := range splitData {
		go executeSet(k, wg, resultCh, v[0], v[1], model.LearningRate, model.Epochs, model.ClassIndex)
	}
	wg.Wait()

	doneCh <- true

	return nil
}
