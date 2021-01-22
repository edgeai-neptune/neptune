package manager

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"k8s.io/klog/v2"

	"github.com/edgeai-neptune/neptune/cmd/neptune-lc/app/options"
	neptunev1 "github.com/edgeai-neptune/neptune/pkg/apis/neptune/v1alpha1"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/db"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/util"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/wsclient"
)

// DatasetManager defines dataset manager
type DatasetManager struct {
	Client            *wsclient.Client
	DatasetMap        map[string]*Dataset
	DataSourcesSignal map[string]bool
	VolumeMountPrefix string
}

// Dataset defines config for dataset
type Dataset struct {
	*neptunev1.Dataset
	DataSource *DataSource `json:"dataSource"`
}

// DatasetSpec defines dataset spec
type DatasetSpec struct {
	Format  string `json:"format"`
	DataURL string `json:"url"`
}

// DataSource defines config for data source
type DataSource struct {
	TrainSamples    []string `json:"trainSamples"`
	ValidSamples    []string `json:"validSamples"`
	NumberOfSamples int      `json:"numberOfSamples"`
}

const (
	// DatasetChannelCacheSize is size of channel cache
	DatasetChannelCacheSize = 100
	// MonitorDataSourceIntervalSeconds is interval time of monitoring data source
	MonitorDataSourceIntervalSeconds = 10
	// DatasetResourceKind is kind of dataset resource
	DatasetResourceKind = "dataset"
)

// NewDatasetManager creates a dataset manager
func NewDatasetManager(client *wsclient.Client, options *options.LocalControllerOptions) (*DatasetManager, error) {
	dm := DatasetManager{
		Client:            client,
		DataSourcesSignal: make(map[string]bool),
		DatasetMap:        make(map[string]*Dataset),
		VolumeMountPrefix: options.VolumeMountPrefix,
	}

	if err := dm.initDatasetManager(); err != nil {
		klog.Errorf("init dataset manager failed, error: %v", err)
		return nil, err
	}

	return &dm, nil
}

// initDatasetManager inits dataset manager
func (dm *DatasetManager) initDatasetManager() error {
	if err := dm.Client.Subscribe(DatasetResourceKind, dm.handleMessage); err != nil {
		klog.Errorf("register dataset manager to the client failed, error: %v", err)
		return err
	}
	klog.Infof("init dataset manager successfully")

	return nil
}

// GetDatasetChannel gets dataset
func (dm *DatasetManager) GetDataset(name string) (*Dataset, bool) {
	d, ok := dm.DatasetMap[name]
	return d, ok
}

// handleMessage handles the message from GlobalManager
func (dm *DatasetManager) handleMessage(message *wsclient.Message) {
	uniqueIdentifier := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)
	switch message.Header.Operation {
	case InsertOperation:
		{
			if err := dm.insertDataset(uniqueIdentifier, message.Content); err != nil {
				klog.Errorf("insert %s(name=%s) to db failed, error: %v", message.Header.ResourceKind, uniqueIdentifier, err)
			}

			if _, ok := dm.DataSourcesSignal[uniqueIdentifier]; !ok {
				dm.DataSourcesSignal[uniqueIdentifier] = true
				go dm.monitorDataSources(message, uniqueIdentifier)
			}
		}
	case DeleteOperation:
		{
			if err := dm.deleteDataset(uniqueIdentifier); err != nil {
				klog.Errorf("delete %s(name=%s) to db failed, error: %v", message.Header.ResourceKind, uniqueIdentifier, err)
			}

			if _, ok := dm.DataSourcesSignal[uniqueIdentifier]; ok {
				dm.DataSourcesSignal[uniqueIdentifier] = false
			}
		}
	}
}

// insertDataset inserts dataset to db
func (dm *DatasetManager) insertDataset(name string, payload []byte) error {
	if _, ok := dm.DatasetMap[name]; !ok {
		dm.DatasetMap[name] = &Dataset{}
	}

	dataset := dm.DatasetMap[name]

	if err := json.Unmarshal(payload, &dataset); err != nil {
		return err
	}

	if err := db.SaveResource(name, dataset.TypeMeta, dataset.ObjectMeta, dataset.Spec); err != nil {
		return err
	}

	return nil
}

// deleteDataset deletes dataset config in db
func (dm *DatasetManager) deleteDataset(name string) error {
	if err := db.DeleteResource(name); err != nil {
		return err
	}

	delete(dm.DatasetMap, name)

	delete(dm.DataSourcesSignal, name)

	return nil
}

// monitorDataSources monitors the data sources
func (dm *DatasetManager) monitorDataSources(message *wsclient.Message, uniqueIdentifier string) {
	for dm.DataSourcesSignal[uniqueIdentifier] {
		time.Sleep(time.Duration(MonitorDataSourceIntervalSeconds) * time.Second)

		ds, ok := dm.DatasetMap[uniqueIdentifier]
		if !ok {
			break
		}

		if ds.Spec.URL == "" {
			klog.Errorf("dataset(name=%s) not found valid data source url.", uniqueIdentifier)
			break
		}

		dataURL := util.AddPrefixPath(dm.VolumeMountPrefix, filepath.Join(ds.Spec.URL))
		dataSource, err := dm.getDataSource(dataURL, ds.Spec.Format)
		if err != nil {
			klog.Errorf("dataset(name=%s) get samples from %s failed", uniqueIdentifier, dataURL)
			continue
		}
		ds.DataSource = dataSource

		klog.Infof("dataset(name=%s) get samples from data source(url=%s) successfully. number of samples: %d",
			uniqueIdentifier, dataURL, dataSource.NumberOfSamples)

		message.Header.Operation = StatusOperation
		if err := dm.Client.WriteMessage(struct {
			NameSpace       string `json:"namespace"`
			Name            string `json:"name"`
			NumberOfSamples int    `json:"numberOfSamples"`
		}{
			message.Header.Namespace,
			message.Header.ResourceName,
			dataSource.NumberOfSamples,
		}, message.Header); err != nil {
			klog.Errorf("dataset(name=%s) publish samples info failed", uniqueIdentifier)
		}
	}
}

// getDataSource gets data source info
func (dm *DatasetManager) getDataSource(dataURL string, format string) (*DataSource, error) {
	switch format {
	case "txt":
		return dm.readByLine(dataURL)
	}
	return nil, fmt.Errorf("not vaild file format")
}

// readByLine reads file by line
func (dm *DatasetManager) readByLine(url string) (*DataSource, error) {
	samples, err := getSamples(url)
	if err != nil {
		klog.Errorf("read file %s failed, error: %v", url, err)
		return nil, err
	}

	numberOfSamples := 0
	numberOfSamples += len(samples)

	dataSource := DataSource{
		TrainSamples:    samples,
		NumberOfSamples: numberOfSamples,
	}

	return &dataSource, nil
}

// getSamples gets samples in a file
func getSamples(url string) ([]string, error) {
	var samples = []string{}
	if !util.IsExists(url) {
		return nil, fmt.Errorf("url(%s) does not exist", url)
	}

	if !util.IsFile(url) {
		return nil, fmt.Errorf("url(%s) is not a file, not vaild", url)
	}

	file, err := os.Open(url)
	if err != nil {
		klog.Errorf("read %s failed, error: %v", url, err)
		return samples, err
	}

	fileScanner := bufio.NewScanner(file)
	for fileScanner.Scan() {
		samples = append(samples, fileScanner.Text())
	}

	if err = file.Close(); err != nil {
		klog.Errorf("close file(url=%s) failed, error: %v", url, err)
		return samples, err
	}

	return samples, nil
}
