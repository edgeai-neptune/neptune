package globalmanager

import (
	"encoding/json"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// ContainerPara describes initial values need by creating a pod
type ContainerPara struct {
	volumeMountList []string
	volumeList      []string
	volumeMapName   []string
	env             map[string]string
	frameName       string
	frameVersion    string
	scriptBootFile  string
	nodeName        string
}

// CommonInterface describes the commom interface of CRs
type CommonInterface interface {
	metav1.Object
	schema.ObjectKind
}

// FeatureControllerI defines the interface of an AI Feature controller
type FeatureControllerI interface {
	Start() error
	GetName() string
}

type Model struct {
	Format  string                 `json:"format,omitempty"`
	URL     string                 `json:"url,omitempty"`
	Metrics map[string]interface{} `json:"metrics,omitempty"`
}

// the data of this condition including the input/output to do the next step
type IncrementalCondData struct {
	Input *struct {
		// Only one model cases
		Model  *Model  `json:"model,omitempty"`
		Models []Model `json:"models,omitempty"`

		DataURL   string `json:"dataURL,omitempty"`
		OutputDir string `json:"outputDir,omitempty"`
	} `json:"input,omitempty"`

	Output *struct {
		Model  *Model  `json:"model,omitempty"`
		Models []Model `json:"models,omitempty"`
	} `json:"output,omitempty"`
}

func (m *Model) GetURL() string {
	return m.URL
}

func (cd *IncrementalCondData) joinModelURLs(model *Model, models []Model) []string {
	var modelURLs []string
	if model != nil {
		modelURLs = append(modelURLs, model.GetURL())
	} else {
		for _, m := range models {
			modelURLs = append(modelURLs, m.GetURL())
		}
	}
	return modelURLs
}

func (cd *IncrementalCondData) GetInputModelURLs() []string {
	return cd.joinModelURLs(cd.Input.Model, cd.Input.Models)
}

func (cd *IncrementalCondData) GetOutputModelURLs() []string {
	return cd.joinModelURLs(cd.Output.Model, cd.Output.Models)
}

func (cd *IncrementalCondData) Unmarshal(data []byte) error {
	return json.Unmarshal(data, cd)
}

func (cd IncrementalCondData) Marshal() ([]byte, error) {
	return json.Marshal(cd)
}
