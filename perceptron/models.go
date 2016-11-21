package perceptron

import . "github.com/vedhavyas/machine-learning/base"

// PerceptronModel embeds BaseModel and add required attributes for Perceptron Algorithm
type PerceptronModel struct {
	BaseModel
	KFold        int
	LearningRate float64
	Epochs       int
}
