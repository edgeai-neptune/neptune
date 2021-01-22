package manager

import (
	"encoding/json"

	"k8s.io/klog/v2"

	neptunev1 "github.com/edgeai-neptune/neptune/pkg/apis/neptune/v1alpha1"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/db"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/util"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/wsclient"
)

// ModelManager defines model manager
type ModelManager struct {
	Client   *wsclient.Client
	ModelMap map[string]neptunev1.Model
}

const (
	// ModelCacheSize is size of cache
	ModelCacheSize = 100
	// ModelResourceKind is kind of dataset resource
	ModelResourceKind = "model"
)

// NewModelManager creates a model manager
func NewModelManager(client *wsclient.Client) (*ModelManager, error) {
	mm := ModelManager{
		ModelMap: make(map[string]neptunev1.Model),
		Client:   client,
	}

	if err := mm.initModelManager(); err != nil {
		klog.Errorf("init model manager failed, error: %v", err)
		return nil, err
	}

	return &mm, nil
}

// initModelManager inits model manager
func (mm *ModelManager) initModelManager() error {
	if err := mm.Client.Subscribe(ModelResourceKind, mm.handleMessage); err != nil {
		klog.Errorf("register model manager to the client failed, error: %v", err)
		return err
	}
	klog.Infof("init model manager successfully")

	return nil
}

// GetModel gets model
func (mm *ModelManager) GetModel(name string) (neptunev1.Model, bool) {
	model, ok := mm.ModelMap[name]
	return model, ok
}

// addNewModel adds model
func (mm *ModelManager) addNewModel(name string, model neptunev1.Model) {

	mm.ModelMap[name] = model
}

// handleMessage handles the message from GlobalManager
func (mm *ModelManager) handleMessage(message *wsclient.Message) {
	uniqueIdentifier := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)
	switch message.Header.Operation {
	case InsertOperation:
		if err := mm.insertModel(uniqueIdentifier, message.Content); err != nil {
			klog.Errorf("insert %s(name=%s) to db failed, error: %v", message.Header.ResourceKind, uniqueIdentifier, err)
		}

	case DeleteOperation:
		if err := mm.deleteModel(uniqueIdentifier); err != nil {
			klog.Errorf("delete model(name=%s) in db failed, error: %v", uniqueIdentifier, err)
		}
	}
}

// insertModel inserts model config to db
func (mm *ModelManager) insertModel(name string, payload []byte) error {
	model := neptunev1.Model{}

	if err := json.Unmarshal(payload, &model); err != nil {
		return err
	}

	if err := db.SaveResource(name, model.TypeMeta, model.ObjectMeta, model.Spec); err != nil {
		return err
	}

	mm.addNewModel(name, model)

	return nil
}

// deleteModel deletes model in db
func (mm *ModelManager) deleteModel(name string) error {
	if err := db.DeleteResource(name); err != nil {
		return err
	}
	delete(mm.ModelMap, name)

	return nil
}
