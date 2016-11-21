package perceptron

import . "github.com/vedhavyas/machine-learning/base"

type PerceptronModel struct {
	BaseModel
	KFold        int
	LearningRate float64
	Epochs       int
}
