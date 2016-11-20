package main

import (
	"math/rand"
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
