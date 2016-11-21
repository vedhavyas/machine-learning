package main

import (
	"flag"
	"log"
	"os"

	. "github.com/vedhavyas/machine-learning/base"
	. "github.com/vedhavyas/machine-learning/knn"
)

func main() {
	log.SetFlags(log.Lshortfile)

	k := flag.Int("k", 3, "Value K for the Benchmark")
	split := flag.Float64("split", 0.6, "Spilit percentage of data. Should be between 0 - 1 float")
	fileName := flag.String("data", "data/iris.data", "Data File")
	classIndex := flag.Int("class-index", -1, "Index of the class in a given row")
	normalise := flag.Bool("norm", false, "True to force normalise the attributes")
	flag.Parse()

	if *split > 1.0 {
		log.Println("Split cannot be grater than 1.0")
		flag.Usage()
		os.Exit(1)
	}

	model := &KNNModel{
		K:     *k,
		Split: float32(*split),
		BaseModel: BaseModel{
			FileName:               *fileName,
			ClassIndex:             *classIndex,
			AttributeNormalisation: *normalise,
		},
	}

	err := ExecuteKNN(model)
	if err != nil {
		log.Fatal(err)
	}
}
