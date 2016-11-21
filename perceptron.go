package main

import (
	"flag"
	"log"

	. "github.com/vedhavyas/machine-learning/base"
	. "github.com/vedhavyas/machine-learning/perceptron"
)

func main() {

	folds := flag.Int("folds", 3, "No. of folds for cross validation split")
	fileName := flag.String("data", "data/sonar-all-data.csv", "Data File")
	classIndex := flag.Int("class-index", -1, "Index of the class in a given row")
	learningRate := flag.Float64("l_rate", 0.01, "Learing rate")
	epoch := flag.Int("epochs", 500, "No. of times to train the data")
	normalise := flag.Bool("norm", true, "Normalise the data before training")
	flag.Parse()

	perceptronModel := &PerceptronModel{
		BaseModel: BaseModel{
			FileName:               *fileName,
			ClassIndex:             *classIndex,
			AttributeNormalisation: *normalise,
		},
		KFold:        *folds,
		LearningRate: *learningRate,
		Epochs:       *epoch,
	}

	err := ExecutePerceptron(perceptronModel)
	if err != nil {
		log.Fatal(err)
	}
}
