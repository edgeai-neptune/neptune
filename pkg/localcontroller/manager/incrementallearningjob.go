package manager

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	"k8s.io/klog/v2"

	"github.com/edgeai-neptune/neptune/cmd/neptune-lc/app/options"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/db"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/trigger"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/util"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/wsclient"
)

// IncrementalLearningJob defines config for incremental-learning-job
type IncrementalLearningJob struct {
	APIVersion string                      `json:"apiVersion"`
	Kind       string                      `json:"kind"`
	MetaData   *MetaData                   `json:"metadata"`
	Spec       *IncrementalLearningJobSpec `json:"spec"`
	JobConfig  *JobConfig                  `json:"jobConfig"`
}

// IncrementalLearningJobSpec defines incremental-learning-job spec
type IncrementalLearningJobSpec struct {
	InitialModel *InitialModel          `json:"initialModel"`
	Dataset      *DatasetConfig         `json:"dataset"`
	TrainSpec    *TrainSpec             `json:"trainSpec"`
	EvalSpec     map[string]interface{} `json:"evalSpec"`
	DeploySpec   *DeploySpec            `json:"deploySpec"`
	OutputDir    string                 `json:"outputDir"`
}

// InitialModel defines initial model config
type InitialModel struct {
	Name string `json:"name"`
}

// TrainSpec defines train spec config
type TrainSpec struct {
	WorkerSpec   map[string]interface{} `json:"workerSpec"`
	TrainTrigger *TrainTrigger          `json:"trigger"`
}

// DeploySpec defines deploy spec config
type DeploySpec struct {
	Model         *DeployModel   `json:"model"`
	DeployTrigger *DeployTrigger `json:"trigger"`
}

// DeployModel defines deploy model config
type DeployModel struct {
	Name string `json:"name"`
}

// TrainTrigger defines config for trigger training
type TrainTrigger struct {
	CheckPeriodSeconds int                    `json:"checkPeriodSeconds"`
	Timer              map[string]interface{} `json:"timer"`
	Condition          map[string]interface{} `json:"condition"`
}

// DeployTrigger defines config for trigger deploy
type DeployTrigger struct {
	Condition map[string]interface{} `json:"condition"`
}

// DeployTriggerCondition defines config for deploy trigger condition
type DeployTriggerCondition struct {
	Operator  string  `json:"operator"`
	Threshold float64 `json:"threshold"`
	Metric    string  `json:"metric"`
}

// DatasetConfig defines config for dataset
type DatasetConfig struct {
	Name      string  `json:"name"`
	TrainProb float64 `json:"trainProb"`
	Format    string  `json:"format"`
}

// JobConfig defines config for incremental-learning-job
type JobConfig struct {
	UniqueIdentifier  string             `json:"uniqueIdentifier"`
	Version           int                `json:"version"`
	Phase             string             `json:"phase"`
	WorkerStatus      string             `json:"workerStatus"`
	TriggerStatus     string             `json:"triggerStatus"`
	TriggerTime       time.Time          `json:"triggerTime"`
	TrainDataURL      string             `json:"trainDataUrl"`
	EvalDataURL       string             `json:"evalDataUrl"`
	OutputDir         string             `json:"outputDir"`
	OutputConfig      *OutputConfig      `json:"outputConfig"`
	DataSamples       *DataSamples       `json:"dataSamples"`
	TrainModelConfig  *TrainModelConfig  `json:"trainModelConfig"`
	DeployModelConfig *DeployModelConfig `json:"deployModelConfig"`
	EvalResult        []*ModelMessage    `json:"evalResult"`
	Lock              sync.Mutex         `json:"lock"`
}

// OutputConfig defines config for job output
type OutputConfig struct {
	SamplesOutput map[string]string `json:"trainData"`
	TrainOutput   string            `json:"trainOutput"`
	EvalOutput    string            `json:"evalOutput"`
}

// DataSamples defines samples information
type DataSamples struct {
	Numbers            int
	TrainSamples       []string
	EvalVersionSamples [][]string
	EvalSamples        []string
}

// TrainModelConfig defines config about training model
type TrainModelConfig struct {
	ModelConfig  *ModelConfig      `json:"modelConfig"`
	TrainedModel map[string]string `json:"trainedModel"`
	OutputURL    string            `json:"outputUrl"`
}

// ModelConfig defines model
type ModelConfig struct {
	Format   string `json:"format"`
	ModelURL string `json:"url"`
}

// DeployModelConfig defines config about deploying model
type DeployModelConfig struct {
	ModelConfig *ModelConfig `json:"modelConfig"`
}

// ModelMessage defines model message from worker
type ModelMessage struct {
	Format  string               `json:"format"`
	URL     string               `json:"url"`
	Metrics map[string][]float64 `json:"metrics"`
}

// TriggeringResultUpstream defines triggering result that will send to GlobalManager
type TriggeringResultUpstream struct {
	Phase  string      `json:"phase"`
	Status string      `json:"status"`
	Input  interface{} `json:"input"`
}

// IncrementalLearningJob defines incremental-learning-job manager
type IncrementalJobManager struct {
	Client               *wsclient.Client
	WorkerMessageChannel chan WorkerMessage
	DatasetManager       *DatasetManager
	ModelManager         *ModelManager
	IncrementalJobMap    map[string]*IncrementalLearningJob
	IncrementalJobSignal map[string]bool
	VolumeMountPrefix    string
}

const (
	// JobIterationIntervalSeconds is interval time of each iteration of job
	JobIterationIntervalSeconds = 10
	// DatasetHandlerIntervalSeconds is interval time of handling dataset
	DatasetHandlerIntervalSeconds = 10
	// ModelHandlerIntervalSeconds is interval time of handling model
	ModelHandlerIntervalSeconds = 10
	// EvalSamplesCapacity is capacity of eval samples
	EvalSamplesCapacity = 5
	//IncrementalLearningJobKind is kind of incremental-learning-job resource
	IncrementalLearningJobKind = "incrementallearningjob"
)

// NewIncrementalJobManager creates a incremental-learning-job manager
func NewIncrementalJobManager(client *wsclient.Client, datasetManager *DatasetManager,
	modelManager *ModelManager, options *options.LocalControllerOptions) *IncrementalJobManager {
	im := IncrementalJobManager{
		Client:               client,
		WorkerMessageChannel: make(chan WorkerMessage, WorkerMessageChannelCacheSize),
		DatasetManager:       datasetManager,
		ModelManager:         modelManager,
		IncrementalJobMap:    make(map[string]*IncrementalLearningJob),
		IncrementalJobSignal: make(map[string]bool),
		VolumeMountPrefix:    options.VolumeMountPrefix,
	}

	return &im
}

// Start starts incremental-learning-job manager
func (im *IncrementalJobManager) Start() error {
	im.IncrementalJobSignal = make(map[string]bool)

	if err := im.Client.Subscribe(IncrementalLearningJobKind, im.handleMessage); err != nil {
		klog.Errorf("register incremental-learning-job manager to the client failed, error: %v", err)
		return err
	}

	go im.monitorWorker()

	klog.Infof("start incremental-learning-job manager successfully")
	return nil
}

// handleMessage handles the message from GlobalManager
func (im *IncrementalJobManager) handleMessage(message *wsclient.Message) {
	uniqueIdentifier := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	switch message.Header.Operation {
	case InsertOperation:
		{
			if err := im.insertJob(uniqueIdentifier, message.Content); err != nil {
				klog.Errorf("insert %s(name=%s) to db failed, error: %v", message.Header.ResourceKind, uniqueIdentifier, err)
			}

			if _, ok := im.IncrementalJobSignal[uniqueIdentifier]; !ok {
				im.IncrementalJobSignal[uniqueIdentifier] = true
				go im.createJob(uniqueIdentifier, message)
			}
		}
	case DeleteOperation:
		{
			if err := im.deleteJob(uniqueIdentifier); err != nil {
				klog.Errorf("delete %s(name=%s) to db failed, error: %v", message.Header.ResourceKind, uniqueIdentifier, err)
			}

			if _, ok := im.IncrementalJobSignal[uniqueIdentifier]; ok {
				im.IncrementalJobSignal[uniqueIdentifier] = false
			}
		}
	}
}

// trainTask starts training task
func (im *IncrementalJobManager) trainTask(incrementalJob *IncrementalLearningJob, message *wsclient.Message) error {
	jobConfig := incrementalJob.JobConfig

	if jobConfig.WorkerStatus == WorkerReadyStatus && jobConfig.TriggerStatus == TriggerReadyStatus {
		payload, ok, err := im.triggerTrainTask(incrementalJob)
		if !ok {
			return nil
		}

		if err != nil {
			klog.Errorf("job(name=%s) complete the %sing phase triggering task failed, error: %v",
				jobConfig.UniqueIdentifier, jobConfig.Phase, err)
			return err
		}

		message.Header.Operation = StatusOperation
		err = im.Client.WriteMessage(payload, message.Header)
		if err != nil {
			return err
		}

		jobConfig.TriggerStatus = TriggerCompletedStatus

		klog.Infof("job(name=%s) complete the %sing phase triggering task successfully",
			jobConfig.UniqueIdentifier, jobConfig.Phase)
	}

	if jobConfig.WorkerStatus == WorkerFailedStatus {
		klog.Warningf("found the %sing phase worker that ran failed, "+
			"back the training phase triggering task", jobConfig.Phase)
		backTask(jobConfig)
	}

	if jobConfig.WorkerStatus == WorkerCompletedStatus {
		klog.Infof("job(name=%s) complete the %s task successfully", jobConfig.UniqueIdentifier, jobConfig.Phase)
		nextTask(jobConfig)
	}

	return nil
}

// evalTask starts eval task
func (im *IncrementalJobManager) evalTask(incrementalJob *IncrementalLearningJob, message *wsclient.Message) error {
	jobConfig := incrementalJob.JobConfig

	if jobConfig.WorkerStatus == WorkerReadyStatus && jobConfig.TriggerStatus == TriggerReadyStatus {
		payload, err := im.triggerEvalTask(incrementalJob)
		if err != nil {
			klog.Errorf("job(name=%s) complete the %sing phase triggering task failed, error: %v",
				jobConfig.UniqueIdentifier, jobConfig.Phase, err)
			return err
		}

		message.Header.Operation = StatusOperation
		err = im.Client.WriteMessage(payload, message.Header)
		if err != nil {
			return err
		}

		jobConfig.TriggerStatus = TriggerCompletedStatus

		klog.Infof("job(name=%s) complete the %sing phase triggering task successfully",
			jobConfig.UniqueIdentifier, jobConfig.Phase)
	}

	if jobConfig.WorkerStatus == WorkerFailedStatus {
		msg := fmt.Sprintf("job(name=%s) found the %sing phase worker that ran failed, "+
			"back the training phase triggering task", jobConfig.UniqueIdentifier, jobConfig.Phase)
		klog.Errorf(msg)
		return fmt.Errorf(msg)
	}

	if jobConfig.WorkerStatus == WorkerCompletedStatus {
		klog.Infof("job(name=%s) complete the %s task successfully", jobConfig.UniqueIdentifier, jobConfig.Phase)
		nextTask(jobConfig)
	}

	return nil
}

// deployTask starts deploy task
func (im *IncrementalJobManager) deployTask(incrementalJob *IncrementalLearningJob, message *wsclient.Message) error {
	jobConfig := incrementalJob.JobConfig

	if jobConfig.WorkerStatus == WorkerReadyStatus && jobConfig.TriggerStatus == TriggerReadyStatus {
		models, payload, err := im.triggerDeployTask(incrementalJob)
		if err != nil {
			klog.Errorf("job(name=%s) complete the %sing phase triggering task failed, error: %v",
				jobConfig.UniqueIdentifier, jobConfig.Phase, err)
			return err
		}

		if err := im.deployModel(incrementalJob, models); err != nil {
			klog.Errorf("job(name=%s) deployed model failed, error: %v", jobConfig.UniqueIdentifier, err)
			return err
		}
		klog.Infof("job(name=%s) deploys deployed model successfully.", jobConfig.UniqueIdentifier)

		message.Header.Operation = StatusOperation
		if err = im.Client.WriteMessage(payload, message.Header); err != nil {
			return err
		}

		jobConfig.TriggerStatus = TriggerCompletedStatus

		klog.Infof("job(name=%s) complete the %sing phase triggering task successfully",
			jobConfig.UniqueIdentifier, jobConfig.Phase)
	}

	nextTask(jobConfig)

	klog.Infof("job(name=%s) complete the %s task successfully", jobConfig.UniqueIdentifier, jobConfig.Phase)

	return nil
}

// createJob creates a job
func (im *IncrementalJobManager) createJob(name string, message *wsclient.Message) {
	var err error
	incrementalJob := im.IncrementalJobMap[name]
	incrementalJob.JobConfig = new(JobConfig)
	jobConfig := incrementalJob.JobConfig
	jobConfig.UniqueIdentifier = name

	err = im.initJob(incrementalJob)
	if err != nil {
		klog.Errorf("init job (name=%s) failed", jobConfig.UniqueIdentifier)
		return
	}

	go im.handleData(incrementalJob)
	go im.handleModel(incrementalJob)

	klog.Infof("creating incremental job(name=%s)", name)

	for im.IncrementalJobSignal[name] {
		time.Sleep(time.Duration(JobIterationIntervalSeconds) * time.Second)

		switch jobConfig.Phase {
		case TrainPhase:
			err = im.trainTask(incrementalJob, message)
		case EvalPhase:
			err = im.evalTask(incrementalJob, message)
		case DeployPhase:
			err = im.deployTask(incrementalJob, message)
		default:
			klog.Errorf("not vaild phase: %s", jobConfig.Phase)
			continue
		}

		if err != nil {
			klog.Errorf("job(name=%s) complete the %s task failed, error: %v",
				jobConfig.UniqueIdentifier, jobConfig.Phase, err)
			continue
		}
	}
}

// insertJob inserts incremental-learning-job config to db
func (im *IncrementalJobManager) insertJob(name string, payload []byte) error {
	if _, ok := im.IncrementalJobMap[name]; !ok {
		im.IncrementalJobMap[name] = &IncrementalLearningJob{}
	}

	incrementalJob := im.IncrementalJobMap[name]

	if err := json.Unmarshal(payload, &incrementalJob); err != nil {
		return err
	}

	metaData, err := json.Marshal(incrementalJob.MetaData)
	if err != nil {
		return err
	}

	spec, err := json.Marshal(incrementalJob.Spec)
	if err != nil {
		return err
	}

	r := db.Resource{
		Name:       name,
		APIVersion: incrementalJob.APIVersion,
		Kind:       incrementalJob.Kind,
		MetaData:   string(metaData),
		Spec:       string(spec),
	}

	if err = db.SaveResource(&r); err != nil {
		return err
	}

	return nil
}

// deleteJob deletes incremental-learning-job config in db
func (im *IncrementalJobManager) deleteJob(name string) error {
	if err := db.DeleteResource(name); err != nil {
		return err
	}

	delete(im.IncrementalJobMap, name)

	delete(im.IncrementalJobSignal, name)

	return nil
}

// initJob inits the job object
func (im *IncrementalJobManager) initJob(incrementalJob *IncrementalLearningJob) error {
	jobConfig := incrementalJob.JobConfig
	jobConfig.OutputDir = util.AddPrefixPath(im.VolumeMountPrefix, incrementalJob.Spec.OutputDir)
	jobConfig.TrainModelConfig = new(TrainModelConfig)
	jobConfig.TrainModelConfig.OutputURL = jobConfig.OutputDir
	jobConfig.DeployModelConfig = new(DeployModelConfig)
	jobConfig.Lock = sync.Mutex{}

	jobConfig.Version = 0
	jobConfig.Phase = TrainPhase
	jobConfig.WorkerStatus = WorkerReadyStatus
	jobConfig.TriggerStatus = TriggerReadyStatus
	jobConfig.TriggerTime = time.Now()

	if err := createOutputDir(jobConfig); err != nil {
		return err
	}

	return nil
}

// triggerTrainTask triggers the train task
func (im *IncrementalJobManager) triggerTrainTask(incrementalJob *IncrementalLearningJob) (interface{}, bool, error) {
	var err error
	jobConfig := incrementalJob.JobConfig
	tt := incrementalJob.Spec.TrainSpec.TrainTrigger

	currentTime := time.Now()
	timeSub := currentTime.Sub(jobConfig.TriggerTime).Seconds()
	if timeSub < float64(tt.CheckPeriodSeconds) {
		klog.Warningf("job(name=%s) is less than %f seconds since the last trigger", jobConfig.UniqueIdentifier, timeSub)
		return nil, false, nil
	}

	triggerMap := map[string]interface{}{
		"timer":     tt.Timer,
		"condition": tt.Condition,
	}
	const numOfSamples = "num_of_samples"
	samples := map[string]interface{}{
		numOfSamples: len(jobConfig.DataSamples.TrainSamples),
	}

	trainTrigger, err := trigger.NewTrigger(triggerMap)
	if err != nil {
		klog.Errorf("train phase: get trigger object failed, error: %v", err)
		return nil, false, err
	}
	isTrigger := trainTrigger.Trigger(samples)

	if !isTrigger {
		klog.Warningf("job(name=%s) train phase: trigger result is false", jobConfig.UniqueIdentifier)
		return nil, false, nil
	}

	jobConfig.Version++

	jobConfig.TrainDataURL, err = im.writeSamples(jobConfig.DataSamples.TrainSamples,
		jobConfig.OutputConfig.SamplesOutput["train"], jobConfig.Version, incrementalJob.Spec.Dataset.Format)
	if err != nil {
		klog.Errorf("train phase: write samples to the file(%s) is failed, error: %v", jobConfig.TrainDataURL, err)
		return nil, false, err
	}

	format := jobConfig.TrainModelConfig.ModelConfig.Format
	m := ModelConfig{
		format,
		jobConfig.TrainModelConfig.TrainedModel[format],
	}
	uts := TriggeringResultUpstream{
		Phase:  TrainPhase,
		Status: WorkerReadyStatus,
		Input: struct {
			Model     ModelConfig `json:"model"`
			DataURL   string      `json:"dataUrl"`
			OutputDir string      `json:"outputDir"`
		}{
			m,
			util.TrimPrefixPath(im.VolumeMountPrefix, jobConfig.TrainDataURL),
			util.TrimPrefixPath(im.VolumeMountPrefix,
				path.Join(jobConfig.OutputConfig.TrainOutput, strconv.Itoa(jobConfig.Version))),
		},
	}
	jobConfig.TriggerTime = time.Now()
	return uts, true, nil
}

// triggerEvalTask triggers the eval task
func (im *IncrementalJobManager) triggerEvalTask(incrementalJOb *IncrementalLearningJob) (interface{}, error) {
	jobConfig := incrementalJOb.JobConfig
	var err error

	(*jobConfig).EvalDataURL, err = im.writeSamples(jobConfig.DataSamples.EvalSamples, jobConfig.OutputConfig.SamplesOutput["eval"],
		jobConfig.Version, incrementalJOb.Spec.Dataset.Format)
	if err != nil {
		klog.Errorf("job(name=%s) eval phase: write samples to the file(%s) is failed, error: %v",
			jobConfig.UniqueIdentifier, jobConfig.EvalDataURL, err)
		return nil, err
	}

	var models = make([]interface{}, 0)
	models = append(models, &ModelConfig{
		Format:   "pb",
		ModelURL: jobConfig.TrainModelConfig.TrainedModel["pb"],
	})

	models = append(models, &ModelConfig{
		Format:   jobConfig.DeployModelConfig.ModelConfig.Format,
		ModelURL: jobConfig.DeployModelConfig.ModelConfig.ModelURL,
	})

	uts := TriggeringResultUpstream{
		Phase:  EvalPhase,
		Status: WorkerReadyStatus,
		Input: struct {
			Models    []interface{} `json:"models"`
			DataURL   string        `json:"dataUrl"`
			OutputDir string        `json:"outputDir"`
		}{
			models,
			util.TrimPrefixPath(im.VolumeMountPrefix, jobConfig.EvalDataURL),
			util.TrimPrefixPath(im.VolumeMountPrefix,
				path.Join(jobConfig.OutputConfig.EvalOutput, strconv.Itoa(jobConfig.Version))),
		},
	}

	return uts, nil
}

// triggerDeployTask triggers the deploy task
func (im *IncrementalJobManager) triggerDeployTask(incrementalJob *IncrementalLearningJob) ([]ModelConfig, interface{}, error) {
	jobConfig := incrementalJob.JobConfig

	if len(jobConfig.EvalResult) != 2 {
		msg := fmt.Sprintf("job(name=%s) deploy phase: get abnormal evaluation result", jobConfig.UniqueIdentifier)
		return nil, nil, fmt.Errorf(msg)
	}

	var models []ModelConfig
	for i := 0; i < len(jobConfig.EvalResult); i++ {
		models = append(models, ModelConfig{
			Format:   jobConfig.EvalResult[i].Format,
			ModelURL: jobConfig.EvalResult[i].URL,
		})
	}

	newMetrics, oldMetrics := jobConfig.EvalResult[0].Metrics, jobConfig.EvalResult[1].Metrics
	metricDelta := make(map[string]interface{})
	tt := incrementalJob.Spec.DeploySpec.DeployTrigger
	cond := DeployTriggerCondition{}
	var err error

	c, err := json.Marshal(tt.Condition)
	if err != nil {
		return nil, nil, err
	}

	err = json.Unmarshal(c, &cond)
	if err != nil {
		return nil, nil, err
	}

	for metric := range newMetrics {
		if strings.HasPrefix(cond.Metric, metric) {
			// keep the full metrics
			metricDelta[metric] = newMetrics[metric]
			var l []float64
			for i := range newMetrics[metric] {
				l = append(l, newMetrics[metric][i]-oldMetrics[metric][i])
			}
			metricDelta[metric+"_delta"] = l
		}
	}

	triggerMap := map[string]interface{}{
		"condition": tt.Condition,
	}
	deployTrigger, err := trigger.NewTrigger(triggerMap)
	if err != nil {
		klog.Errorf("job(name=%s) deploy phase: get trigger object failed, error: %v", jobConfig.UniqueIdentifier, err)
		return nil, nil, err
	}

	isTrigger := deployTrigger.Trigger(metricDelta)

	if !isTrigger {
		msg := fmt.Sprintf("job(name=%s) deploy phase: trigger result is false", jobConfig.UniqueIdentifier)
		return nil, nil, fmt.Errorf(msg)
	}

	uts := TriggeringResultUpstream{
		Phase:  DeployPhase,
		Status: WorkerReadyStatus,
		Input: struct {
			Model ModelConfig `json:"model"`
		}{
			models[0],
		},
	}

	return models, uts, nil
}

// deployModel deploys model
func (im *IncrementalJobManager) deployModel(incrementalJob *IncrementalLearningJob, models []ModelConfig) error {
	jobConfig := incrementalJob.JobConfig
	var err error

	trainedModelFormat := models[0].Format
	deployModelFormat := models[1].Format
	if trainedModelFormat != deployModelFormat {
		msg := fmt.Sprintf("the trained model format(format=%s) is inconsistent with deploy model(format=%s)",
			deployModelFormat, deployModelFormat)
		klog.Errorf(msg)

		return fmt.Errorf(msg)
	}

	trainedModel := util.AddPrefixPath(im.VolumeMountPrefix, models[0].ModelURL)
	deployModel := util.AddPrefixPath(im.VolumeMountPrefix, models[1].ModelURL)
	if _, err = util.CopyFile(trainedModel, deployModel); err != nil {
		klog.Errorf("copy the trained model file(url=%s) to the deployment model file(url=%s) failed",
			trainedModel, deployModel)

		return err
	}

	jobConfig.DeployModelConfig.ModelConfig.Format = models[1].Format
	jobConfig.DeployModelConfig.ModelConfig.ModelURL = models[1].ModelURL

	klog.Infof("job(name=%s) deploys model(url=%s) successfully", jobConfig.UniqueIdentifier, trainedModel)

	return nil
}

// createOutputDir creates the job output dir
func createOutputDir(jobConfig *JobConfig) error {
	if err := util.CreateFolder(jobConfig.OutputDir); err != nil {
		klog.Errorf("job(name=%s) create fold %s failed", jobConfig.UniqueIdentifier, jobConfig.OutputDir)
		return err
	}

	dirNames := []string{"data/train", "data/eval", "train", "eval"}

	for _, v := range dirNames {
		dir := path.Join(jobConfig.OutputDir, v)
		if err := util.CreateFolder(dir); err != nil {
			klog.Errorf("job(name=%s) create fold %s failed", jobConfig.UniqueIdentifier, dir)
			return err
		}
	}

	outputConfig := OutputConfig{
		SamplesOutput: map[string]string{
			"train": path.Join(jobConfig.OutputDir, dirNames[0]),
			"eval":  path.Join(jobConfig.OutputDir, dirNames[1]),
		},
		TrainOutput: path.Join(jobConfig.OutputDir, dirNames[2]),
		EvalOutput:  path.Join(jobConfig.OutputDir, dirNames[3]),
	}
	jobConfig.OutputConfig = &outputConfig

	return nil
}

// handleModel updates model information for training and deploying
func (im *IncrementalJobManager) handleModel(incrementalJob *IncrementalLearningJob) {
	jobConfig := incrementalJob.JobConfig
	jobConfig.TrainModelConfig.ModelConfig = new(ModelConfig)
	jobConfig.TrainModelConfig.TrainedModel = map[string]string{}
	jobConfig.DeployModelConfig.ModelConfig = new(ModelConfig)

	for {
		time.Sleep(time.Duration(ModelHandlerIntervalSeconds) * time.Second)

		var modelName string
		modelName = util.GetUniqueIdentifier(incrementalJob.MetaData.Namespace, incrementalJob.Spec.InitialModel.Name, ModelResourceKind)
		trainModelChannel := im.ModelManager.GetModelChannel(modelName)
		if trainModelChannel == nil {
			klog.Warningf("job(name=%s) gets model(name=%s) failed", jobConfig.UniqueIdentifier, modelName)
			continue
		}

		trainModel, ok := <-trainModelChannel
		if !ok {
			break
		}

		jobConfig.TrainModelConfig.ModelConfig.Format = trainModel.Spec.Format
		jobConfig.TrainModelConfig.ModelConfig.ModelURL = trainModel.Spec.URL
		if _, ok := jobConfig.TrainModelConfig.TrainedModel[trainModel.Spec.Format]; !ok {
			jobConfig.TrainModelConfig.TrainedModel[trainModel.Spec.Format] = trainModel.Spec.URL
		}

		modelName = util.GetUniqueIdentifier(incrementalJob.MetaData.Namespace, incrementalJob.Spec.DeploySpec.Model.Name, ModelResourceKind)
		evalModelChannel := im.ModelManager.GetModelChannel(modelName)
		if evalModelChannel == nil {
			klog.Warningf("job(name=%s) gets model(name=%s) failed", jobConfig.UniqueIdentifier, modelName)
			continue
		}

		evalModel, ok := <-evalModelChannel
		if !ok {
			break
		}
		jobConfig.DeployModelConfig.ModelConfig.Format = evalModel.Spec.Format
		jobConfig.DeployModelConfig.ModelConfig.ModelURL = evalModel.Spec.URL
	}
}

// handleData updates samples information
func (im *IncrementalJobManager) handleData(incrementalJob *IncrementalLearningJob) {
	jobConfig := incrementalJob.JobConfig
	jobConfig.DataSamples = &DataSamples{
		Numbers:            0,
		TrainSamples:       make([]string, 0),
		EvalVersionSamples: make([][]string, 0),
		EvalSamples:        make([]string, 0),
	}

	for {
		time.Sleep(time.Duration(DatasetHandlerIntervalSeconds) * time.Second)

		datasetName := util.GetUniqueIdentifier(incrementalJob.MetaData.Namespace, incrementalJob.Spec.Dataset.Name, DatasetResourceKind)
		datasetChannel := im.DatasetManager.GetDatasetChannel(datasetName)
		if datasetChannel == nil {
			klog.Errorf("job(name=%s) gets dataset (name=%s) failed", jobConfig.UniqueIdentifier, datasetName)
			continue
		}

		dataset, ok := <-datasetChannel
		if !ok {
			break
		}
		samples := dataset.DataSource.TrainSamples

		if len(samples) > jobConfig.DataSamples.Numbers {
			trainNum := int(incrementalJob.Spec.Dataset.TrainProb * float64(len(samples)-jobConfig.DataSamples.Numbers))

			jobConfig.Lock.Lock()
			jobConfig.DataSamples.TrainSamples = append(jobConfig.DataSamples.TrainSamples,
				samples[(jobConfig.DataSamples.Numbers+1):(jobConfig.DataSamples.Numbers+trainNum+1)]...)
			jobConfig.Lock.Unlock()
			klog.Infof("job(name=%s) current train samples nums is %d",
				jobConfig.UniqueIdentifier, len(jobConfig.DataSamples.TrainSamples))

			jobConfig.Lock.Lock()
			jobConfig.DataSamples.EvalVersionSamples = append(jobConfig.DataSamples.EvalVersionSamples,
				samples[(jobConfig.DataSamples.Numbers+trainNum+1):])
			jobConfig.Lock.Unlock()

			for _, v := range jobConfig.DataSamples.EvalVersionSamples {
				jobConfig.DataSamples.EvalSamples = append(jobConfig.DataSamples.EvalSamples, v...)
			}
			klog.Infof("job(name=%s) current eval samples nums is %d",
				jobConfig.UniqueIdentifier, len(jobConfig.DataSamples.EvalSamples))

			jobConfig.DataSamples.Numbers = len(samples)
			incrementalJob.Spec.Dataset.Format = dataset.Spec.Format
		} else {
			klog.Warningf("job(name=%s) didn't get new data from dataset(name=%s)",
				jobConfig.UniqueIdentifier, incrementalJob.Spec.Dataset.Name)
		}
	}
}

// createFile create a file
func createFile(dir string, format string) (string, bool) {
	switch format {
	case "txt":
		return path.Join(dir, "data.txt"), true
	}
	return "", false
}

// writeSamples writes samples information to a file
func (im *IncrementalJobManager) writeSamples(samples []string, dir string, version int, format string) (string, error) {
	subDir := path.Join(dir, strconv.Itoa(version))
	if err := util.CreateFolder(subDir); err != nil {
		return "", err
	}

	fileURL, isFile := createFile(subDir, format)
	if isFile {
		if err := im.writeByLine(samples, fileURL); err != nil {
			return "", err
		}
	} else {
		return "", fmt.Errorf("create a %s format file in %s failed", format, subDir)
	}

	return fileURL, nil
}

// writeByLine writes file by line
func (im *IncrementalJobManager) writeByLine(samples []string, fileURL string) error {
	file, err := os.Create(fileURL)
	if err != nil {
		klog.Errorf("create file(%s) failed", fileURL)
		return err
	}

	w := bufio.NewWriter(file)
	for _, line := range samples {
		_, _ = fmt.Fprintln(w, line)
	}
	if err := w.Flush(); err != nil {
		klog.Errorf("write file(%s) failed", fileURL)
		return err
	}

	if err := file.Close(); err != nil {
		klog.Errorf("close file failed, error: %v", err)
		return err
	}

	return nil
}

// monitorWorker monitors message from worker
func (im *IncrementalJobManager) monitorWorker() {
	for {
		workerMessageChannel := im.WorkerMessageChannel
		workerMessage, ok := <-workerMessageChannel
		if !ok {
			break
		}
		klog.V(4).Infof("handling worker message %+v", workerMessage)

		name := util.GetUniqueIdentifier(workerMessage.Namespace, workerMessage.OwnerName, workerMessage.OwnerKind)
		header := wsclient.MessageHeader{
			Namespace:    workerMessage.Namespace,
			ResourceKind: workerMessage.OwnerKind,
			ResourceName: workerMessage.OwnerName,
			Operation:    StatusOperation,
		}

		if err := im.Client.WriteMessage(workerMessage, header); err != nil {
			klog.Errorf("job(name=%s) uploads worker(name=%s) message failed, error: %v",
				name, workerMessage.Name, err)
		}

		job, ok := im.IncrementalJobMap[name]
		if !ok {
			continue
		}

		im.handleWorkerMessage(job, workerMessage)
	}
}

// handleWorkerMessage handles message from worker
func (im *IncrementalJobManager) handleWorkerMessage(incrementalJob *IncrementalLearningJob, workerMessage WorkerMessage) {
	incrementalJob.JobConfig.TrainModelConfig.TrainedModel = make(map[string]string)

	jobPhase := incrementalJob.JobConfig.Phase
	workerKind := workerMessage.Kind
	if jobPhase != workerKind {
		klog.Warningf("job(name=%s) %s phase get worker(kind=%s)", incrementalJob.JobConfig.UniqueIdentifier,
			jobPhase, workerKind)
		return
	}

	var models []*ModelMessage
	for _, result := range workerMessage.Results {
		metrics := map[string][]float64{}
		if m, ok := result["metrics"]; ok {
			bytes, err := json.Marshal(m)
			if err != nil {
				return
			}

			err = json.Unmarshal(bytes, &metrics)
			if err != nil {
				klog.Warningf("failed to unmarshal the worker(name=%s) metrics %v, err: %v",
					workerMessage.Name,
					m,
					err)
			}
		}

		model := ModelMessage{
			result["format"].(string),
			result["url"].(string),
			metrics}
		models = append(models, &model)
	}

	incrementalJob.JobConfig.WorkerStatus = workerMessage.Status

	if incrementalJob.JobConfig.WorkerStatus == WorkerCompletedStatus {
		switch incrementalJob.JobConfig.Phase {
		case TrainPhase:
			{
				for i := 0; i < len(models); i++ {
					format := models[i].Format
					if format != "" {
						incrementalJob.JobConfig.TrainModelConfig.TrainedModel[format] = models[i].URL
					}
				}
			}

		case EvalPhase:
			incrementalJob.JobConfig.EvalResult = models
		}
	}
}

// forwardSamples deletes the samples information in the memory
func forwardSamples(jobConfig *JobConfig) {
	switch jobConfig.Phase {
	case TrainPhase:
		{
			jobConfig.Lock.Lock()
			jobConfig.DataSamples.TrainSamples = jobConfig.DataSamples.TrainSamples[:0]
			jobConfig.Lock.Unlock()
		}
	case EvalPhase:
		{
			if len(jobConfig.DataSamples.EvalVersionSamples) > EvalSamplesCapacity {
				jobConfig.DataSamples.EvalVersionSamples = jobConfig.DataSamples.EvalVersionSamples[1:]
			}
		}
	}
}

// backTask backs train task status
func backTask(jobConfig *JobConfig) {
	jobConfig.Phase = TrainPhase
	initTaskStatus(jobConfig)
}

// initTaskStatus inits task status
func initTaskStatus(jobConfig *JobConfig) {
	jobConfig.WorkerStatus = WorkerReadyStatus
	jobConfig.TriggerStatus = TriggerReadyStatus
}

// nextTask converts next task status
func nextTask(jobConfig *JobConfig) {
	switch jobConfig.Phase {
	case TrainPhase:
		{
			forwardSamples(jobConfig)
			initTaskStatus(jobConfig)
			jobConfig.Phase = EvalPhase
		}

	case EvalPhase:
		{
			forwardSamples(jobConfig)
			initTaskStatus(jobConfig)
			jobConfig.Phase = DeployPhase
		}
	case DeployPhase:
		{
			backTask(jobConfig)
		}
	}
}

// AddWorkerMessageToChannel adds worker messages to the channel
func (im *IncrementalJobManager) AddWorkerMessageToChannel(message WorkerMessage) {
	im.WorkerMessageChannel <- message
}

// GetKind gets kind of the manager
func (im *IncrementalJobManager) GetKind() string {
	return IncrementalLearningJobKind
}
