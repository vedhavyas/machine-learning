package main

import (
	"flag"
	"log"
)

func main() {

	folds := flag.Int("folds", 3, "No. of folds for cross validation split")
	fileName := flag.String("data", "data/sonar-all-data.csv", "Data File")
	classIndex := flag.Int("class-index", 60, "Index of the class in a given row")
	learningRate := flag.Float64("l_rate", 0.01, "Learing rate")
	epoch := flag.Int("epochs", 500, "No. of times to train the data")
	flag.Parse()

	perceptronModel := &PerceptronModel{
		FileName:     *fileName,
		KFold:        *folds,
		LearningRate: *learningRate,
		Epochs:       *epoch,
		ClassIndex:   *classIndex,
	}

	err := ExecutePerceptron(perceptronModel)
	if err != nil {
		log.Fatal(err)
	}

}
