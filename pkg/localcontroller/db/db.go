package db

import (
	"encoding/json"
	"os"
	"path/filepath"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	"k8s.io/klog/v2"

	"github.com/edgeai-neptune/neptune/pkg/localcontroller/common/constants"
)

// Resource defines resource (e.g., dataset, model, jointinferenceservice) table
type Resource struct {
	gorm.Model
	Name       string `gorm:"unique"`
	TypeMeta   string
	ObjectMeta string
	Spec       string
}

// SaveResource saves resource info in db
func SaveResource(name string, typeMeta, objectMeta, spec interface{}) error {
	var err error
	dbClient := getClient()

	r := Resource{}

	typeMetaData, _ := json.Marshal(typeMeta)
	objectMetaData, _ := json.Marshal(objectMeta)
	specData, _ := json.Marshal(spec)

	queryResult := dbClient.Where("name = ?", name).First(&r)

	if queryResult.RowsAffected == 0 {
		newR := &Resource{
			Name:       name,
			TypeMeta:   string(typeMetaData),
			ObjectMeta: string(objectMetaData),
			Spec:       string(specData),
		}
		if err = dbClient.Create(newR).Error; err != nil {
			klog.Errorf("failed to save resource(name=%s): %v", name, err)
			return err
		}
		klog.Infof("saved resource(name=%s)", name)
	} else {
		r.TypeMeta = string(typeMetaData)
		r.ObjectMeta = string(objectMetaData)
		r.Spec = string(specData)
		if err := dbClient.Save(&r).Error; err != nil {
			klog.Errorf("failed to update resource(name=%s): %v", name, err)
			return err
		}
		klog.Infof("updated resource(name=%s)", name)
	}

	return nil
}

// DeleteResource deletes resource info in db
func DeleteResource(name string) error {
	var err error
	dbClient := getClient()

	r := Resource{}

	queryResult := dbClient.Where("name = ?", name).First(&r)

	if queryResult.RowsAffected == 0 {
		return nil
	}

	if err = dbClient.Unscoped().Delete(&r).Error; err != nil {
		klog.Errorf("delete resource(name=%s) to db failed, error: %v", name, err)
		return err
	}

	return nil
}

// getClient gets db client
func getClient() *gorm.DB {
	dbURL := constants.DataBaseURL

	if _, err := os.Stat(dbURL); err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(filepath.Dir(dbURL), os.ModePerm); err != nil {
				klog.Errorf("create fold(url=%s) failed, error: %v", filepath.Dir(dbURL), err)
			}
		}
	}

	db, err := gorm.Open(sqlite.Open(dbURL), &gorm.Config{})
	if err != nil {
		klog.Errorf("try to connect the db failed, error: %v", err)
	}

	_ = db.AutoMigrate(&Resource{})

	return db
}
