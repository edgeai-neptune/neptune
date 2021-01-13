package globalmanager

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	k8scontroller "k8s.io/kubernetes/pkg/controller"

	neptunev1 "github.com/edgeai-neptune/neptune/pkg/apis/neptune/v1alpha1"
	clientset "github.com/edgeai-neptune/neptune/pkg/client/clientset/versioned"
	neptuneclientset "github.com/edgeai-neptune/neptune/pkg/client/clientset/versioned/typed/neptune/v1alpha1"
	informers "github.com/edgeai-neptune/neptune/pkg/client/informers/externalversions"
	neptunev1listers "github.com/edgeai-neptune/neptune/pkg/client/listers/neptune/v1alpha1"
	"github.com/edgeai-neptune/neptune/pkg/globalmanager/config"
	messageContext "github.com/edgeai-neptune/neptune/pkg/globalmanager/messagelayer/ws"
	"github.com/edgeai-neptune/neptune/pkg/globalmanager/utils"
)

type FLJobStage string

const (
	FLJobStageAgg   FLJobStage = "Aggregation"
	FLJobStageTrain FLJobStage = "Training"
)

// controllerKind contains the schema.GroupVersionKind for this controller type.
var controllerKind = neptunev1.SchemeGroupVersion.WithKind("FederatedLearningJob")

var (
	// DefaultFLJobBackOff is the default backoff period, exported for the e2e test
	DefaultFLJobBackOff = 10 * time.Second
	// MaxFLJobBackOff is the max backoff period, exported for the e2e test
	MaxFLJobBackOff = 360 * time.Second
)

// FederatedController ensures that all FLJob objects have corresponding pods to
// run their configured workload.
type FederatedController struct {
	kubeClient kubernetes.Interface
	client     neptuneclientset.NeptuneV1alpha1Interface
	podControl k8scontroller.PodControlInterface

	// To allow injection of updateFLJobStatus for testing.
	updateHandler func(flJob *neptunev1.FederatedLearningJob) error
	syncHandler   func(jobKey string) (bool, error)
	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced cache.InformerSynced
	// jobStoreSynced returns true if the flJob store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	jobStoreSynced cache.InformerSynced

	// A store of jobs
	jobLister neptunev1listers.FederatedLearningJobLister

	// A store of pods, populated by the podController
	podStore corelisters.PodLister

	// FLJobs that need to be updated
	queue workqueue.RateLimitingInterface

	recorder record.EventRecorder

	cfg *config.ControllerConfig
}

// Run the main goroutine responsible for watching and syncing jobs.
func (fc *FederatedController) Start() error {
	workers := 1
	stopCh := messageContext.Done()

	go func() {
		defer utilruntime.HandleCrash()
		defer fc.queue.ShutDown()
		klog.Infof("Starting federatedlearning job controller")
		defer klog.Infof("Shutting down federatedlearning job controller")

		if !cache.WaitForNamedCacheSync("federatedlearning job", stopCh, fc.podStoreSynced, fc.jobStoreSynced) {
			klog.Errorf("failed to wait for caches to sync")

			return
		}

		klog.Infof("Starting federatedlearning job workers")
		for i := 0; i < workers; i++ {
			go wait.Until(fc.worker, time.Second, stopCh)
		}

		<-stopCh
	}()
	return nil
}

func getPodJobs(l neptunev1listers.FederatedLearningJobLister, pod *v1.Pod) (jobs []neptunev1.FederatedLearningJob, err error) {
	if len(pod.Labels) == 0 {
		err = fmt.Errorf("no jobs found for pod %v because it has no labels", pod.Name)
		return
	}

	var list []*neptunev1.FederatedLearningJob
	list, err = l.FederatedLearningJobs(pod.Namespace).List(labels.Everything())
	if err != nil {
		return
	}
	for _, job := range list {
		selector, _ := GenerateSelector(job)
		if !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		jobs = append(jobs, *job)
	}
	if len(jobs) == 0 {
		err = fmt.Errorf("could not find federated jobs for pod %s in namespace %s with labels: %v", pod.Name, pod.Namespace, pod.Labels)
	}
	return
}

// getPodFLJobs returns a list of FLJobs that potentially match a Pod.
func (fc *FederatedController) getPodFLJobs(pod *v1.Pod) []*neptunev1.FederatedLearningJob {

	jobs, err := getPodJobs(fc.jobLister, pod)
	if err != nil {
		return nil
	}
	if len(jobs) > 1 {
		// ControllerRef will ensure we don't do anything crazy, but more than one
		// item in this list nevertheless constitutes user error.
		utilruntime.HandleError(fmt.Errorf("user error! more than one federatedlearning job is selecting pods with labels: %+v", pod.Labels))
	}
	ret := make([]*neptunev1.FederatedLearningJob, 0, len(jobs))
	for i := range jobs {
		ret = append(ret, &jobs[i])
	}
	return ret
}

// resolveControllerRef returns the controller referenced by a ControllerRef,
// or nil if the ControllerRef could not be resolved to a matching controller
// of the correct Kind.
func (fc *FederatedController) resolveControllerRef(namespace string, controllerRef *metav1.OwnerReference) *neptunev1.FederatedLearningJob {
	// We can't look up by UID, so look up by Name and then verify UID.
	// Don't even try to look up by Name if it's the wrong Kind.
	if controllerRef.Kind != controllerKind.Kind {
		return nil
	}
	flJob, err := fc.jobLister.FederatedLearningJobs(namespace).Get(controllerRef.Name)
	if err != nil {
		return nil
	}
	if flJob.UID != controllerRef.UID {
		// The controller we found with this Name is not the same one that the
		// ControllerRef points to.
		return nil
	}
	return flJob
}

// When a pod is created, enqueue the controller that manages it and update it's expectations.
func (fc *FederatedController) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	if pod.DeletionTimestamp != nil {
		// on a restart of the controller, it's possible a new pod shows up in a state that
		// is already pending deletion. Prevent the pod from being a creation observation.
		fc.deletePod(pod)
		return
	}

	// If it has a ControllerRef, that's all that matters.
	if controllerRef := metav1.GetControllerOf(pod); controllerRef != nil {
		flJob := fc.resolveControllerRef(pod.Namespace, controllerRef)
		if flJob == nil {
			return
		}
		_, err := k8scontroller.KeyFunc(flJob)
		if err != nil {
			return
		}
		fc.enqueueController(flJob, true)
		return
	}

	// Otherwise, it's an orphan. Get a list of all matching controllers and sync
	// them to see if anyone wants to adopt it.
	// DO NOT observe creation because no controller should be waiting for an
	// orphan.
	for _, flJob := range fc.getPodFLJobs(pod) {
		fc.enqueueController(flJob, true)
	}
}

// When a pod is updated, figure out what flJob/s manage it and wake them up.
// If the labels of the pod have changed we need to awaken both the old
// and new flJob. old and cur must be *v1.Pod types.
func (fc *FederatedController) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		// Periodic resync will send update events for all known pods.
		// Two different versions of the same pod will always have different RVs.
		return
	}
	if curPod.DeletionTimestamp != nil {
		// when a pod is deleted gracefully it's deletion timestamp is first modified to reflect a grace period,
		// and after such time has passed, the kubelet actually deletes it from the store. We receive an update
		// for modification of the deletion timestamp and expect an flJob to create more pods asap, not wait
		// until the kubelet actually deletes the pod.
		fc.deletePod(curPod)
		return
	}

	// the only time we want the backoff to kick-in, is when the pod failed
	immediate := curPod.Status.Phase != v1.PodFailed

	curControllerRef := metav1.GetControllerOf(curPod)
	oldControllerRef := metav1.GetControllerOf(oldPod)
	controllerRefChanged := !reflect.DeepEqual(curControllerRef, oldControllerRef)
	if controllerRefChanged && oldControllerRef != nil {
		// The ControllerRef was changed. Sync the old controller, if any.
		if flJob := fc.resolveControllerRef(oldPod.Namespace, oldControllerRef); flJob != nil {
			fc.enqueueController(flJob, immediate)
		}
	}

	// If it has a ControllerRef, that's all that matters.
	if curControllerRef != nil {
		flJob := fc.resolveControllerRef(curPod.Namespace, curControllerRef)
		if flJob == nil {
			return
		}
		fc.enqueueController(flJob, immediate)
		return
	}

	// Otherwise, it's an orphan. If anything changed, sync matching controllers
	// to see if anyone wants to adopt it now.
	labelChanged := !reflect.DeepEqual(curPod.Labels, oldPod.Labels)
	if labelChanged || controllerRefChanged {
		for _, flJob := range fc.getPodFLJobs(curPod) {
			fc.enqueueController(flJob, immediate)
		}
	}
}

// When a pod is deleted, enqueue the flJob that manages the pod and update its expectations.
// obj could be an *v1.Pod, or a DeletionFinalStateUnknown marker item.
func (fc *FederatedController) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new flJob will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %+v", obj))
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a pod %+v", obj))
			return
		}
	}

	controllerRef := metav1.GetControllerOf(pod)
	if controllerRef == nil {
		// No controller should care about orphans being deleted.
		return
	}
	flJob := fc.resolveControllerRef(pod.Namespace, controllerRef)
	if flJob == nil {
		return
	}
	fc.enqueueController(flJob, true)
}

func (fc *FederatedController) updateFLJob(old, cur interface{}) {
	curFLJob := cur.(*neptunev1.FederatedLearningJob)

	fc.enqueueController(curFLJob, true)
}

// obj could be an *edgeai.FederatedLearningJob, or a DeletionFinalStateUnknown marker item,
// immediate tells the controller to update the status right away, and should
// happen ONLY when there was a successful pod run.
func (fc *FederatedController) enqueueController(obj interface{}, immediate bool) {
	key, err := k8scontroller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}

	backoff := time.Duration(0)
	if !immediate {
		backoff = getBackoff(fc.queue, key)
	}

	// TODO: Handle overlapping controllers better. Either disallow them at admission time or
	// deterministically avoid syncing controllers that fight over pods. Currently, we only
	// ensure that the same controller is synced for a given pod. When we periodically relist
	// all controllers there will still be some replica instability. One way to handle this is
	// by querying the store for all controllers that this rc overlaps, as well as all
	// controllers that overlap this rc, and sorting them.
	fc.queue.AddAfter(key, backoff)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (fc *FederatedController) worker() {
	for fc.processNextWorkItem() {
	}
}

func (fc *FederatedController) processNextWorkItem() bool {
	key, quit := fc.queue.Get()
	if quit {
		return false
	}
	defer fc.queue.Done(key)

	forget, err := fc.syncHandler(key.(string))
	if err == nil {
		if forget {
			fc.queue.Forget(key)
		}
		return true
	}

	utilruntime.HandleError(fmt.Errorf("Error syncing federatedlearning job: %v", err))
	fc.queue.AddRateLimited(key)

	return true
}

// syncFLJob will sync the flJob with the given key if it has had its expectations fulfilled, meaning
// it did not expect to see any more of its pods created or deleted. This function is not meant to be invoked
// concurrently with the same key.
func (fc *FederatedController) syncFLJob(key string) (bool, error) {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing federatedlearning job %q (%v)", key, time.Since(startTime))
	}()

	ns, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return false, err
	}
	if len(ns) == 0 || len(name) == 0 {
		return false, fmt.Errorf("invalid federatedlearning job key %q: either namespace or name is missing", key)
	}
	sharedFLJob, err := fc.jobLister.FederatedLearningJobs(ns).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			klog.V(4).Infof("FLJob has been deleted: %v", key)
			return true, nil
		}
		return false, err
	}
	flJob := *sharedFLJob
	// set kind for flJob in case that the kind is None
	flJob.SetGroupVersionKind(neptunev1.SchemeGroupVersion.WithKind("FederatedLearningJob"))
	// if flJob was finished previously, we don't want to redo the termination
	if IsFLJobFinished(&flJob) {
		return true, nil
	}
	selector, _ := GenerateSelector(&flJob)
	pods, err := fc.podStore.Pods(flJob.Namespace).List(selector)
	if err != nil {
		return false, err
	}

	activePods := k8scontroller.FilterActivePods(pods)
	active := int32(len(activePods))
	succeeded, failed := getStatus(pods)
	conditions := len(flJob.Status.Conditions)
	// flJob first start
	if flJob.Status.StartTime == nil {
		now := metav1.Now()
		flJob.Status.StartTime = &now
	}

	var manageJobErr error
	jobFailed := false
	var failureReason string
	var failureMessage string
	phase := flJob.Status.Phase

	if failed > 0 {
		jobFailed = true
		failureReason = "workerFailed"
		failureMessage = "the worker of FLJob failed"
	}

	if jobFailed {
		flJob.Status.Conditions = append(flJob.Status.Conditions, NewFLJobCondition(neptunev1.FLJobCondFailed, failureReason, failureMessage))
		flJob.Status.Phase = neptunev1.FLJobFailed
		fc.recorder.Event(&flJob, v1.EventTypeWarning, failureReason, failureMessage)
	} else {
		// in the First time, we create the pods
		if len(pods) == 0 {
			active, manageJobErr = fc.createPod(&flJob)
		}
		complete := false
		if succeeded > 0 && active == 0 {
			complete = true
		}
		if complete {
			flJob.Status.Conditions = append(flJob.Status.Conditions, NewFLJobCondition(neptunev1.FLJobCondComplete, "", ""))
			now := metav1.Now()
			flJob.Status.CompletionTime = &now
			fc.recorder.Event(&flJob, v1.EventTypeNormal, "Completed", "FLJob completed")
			flJob.Status.Phase = neptunev1.FLJobSucceeded
		} else {
			flJob.Status.Phase = neptunev1.FLJobRunning
		}
	}

	forget := false
	// Check if the number of jobs succeeded increased since the last check. If yes "forget" should be true
	// This logic is linked to the issue: https://github.com/kubernetes/kubernetes/issues/56853 that aims to
	// improve the FLJob backoff policy when parallelism > 1 and few FLJobs failed but others succeed.
	// In this case, we should clear the backoff delay.
	if flJob.Status.Succeeded < succeeded {
		forget = true
	}

	// no need to update the flJob if the status hasn't changed since last time
	if flJob.Status.Active != active || flJob.Status.Succeeded != succeeded || flJob.Status.Failed != failed || len(flJob.Status.Conditions) != conditions || flJob.Status.Phase != phase {
		flJob.Status.Active = active
		flJob.Status.Succeeded = succeeded
		flJob.Status.Failed = failed

		if err := fc.updateHandler(&flJob); err != nil {
			return forget, err
		}

		if jobFailed && !IsFLJobFinished(&flJob) {
			// returning an error will re-enqueue FLJob after the backoff period
			return forget, fmt.Errorf("failed pod(s) detected for flJob key %q", key)
		}

		forget = true
	}

	return forget, manageJobErr
}

func NewFLJobCondition(conditionType neptunev1.FLJobConditionType, reason, message string) neptunev1.FLJobCondition {
	return neptunev1.FLJobCondition{
		Type:               conditionType,
		Status:             v1.ConditionTrue,
		LastProbeTime:      metav1.Now(),
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}
}

// getStatus returns no of succeeded and failed pods running a flJob
func getStatus(pods []*v1.Pod) (succeeded, failed int32) {
	succeeded = int32(filterPods(pods, v1.PodSucceeded))
	failed = int32(filterPods(pods, v1.PodFailed))
	return
}

func (fc *FederatedController) updateFLJobStatus(flJob *neptunev1.FederatedLearningJob) error {
	jobClient := fc.client.FederatedLearningJobs(flJob.Namespace)
	var err error
	for i := 0; i <= statusUpdateRetries; i = i + 1 {
		var newFLJob *neptunev1.FederatedLearningJob
		newFLJob, err = jobClient.Get(context.TODO(), flJob.Name, metav1.GetOptions{})
		if err != nil {
			break
		}
		newFLJob.Status = flJob.Status
		if _, err = jobClient.UpdateStatus(context.TODO(), newFLJob, metav1.UpdateOptions{}); err == nil {
			break
		}
	}
	return nil
}

// filterPods returns pods based on their phase.
func filterPods(pods []*v1.Pod, phase v1.PodPhase) int {
	result := 0
	for i := range pods {
		if phase == pods[i].Status.Phase {
			result++
		}
	}
	return result
}

func IsFLJobFinished(j *neptunev1.FederatedLearningJob) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == neptunev1.FLJobCondComplete || c.Type == neptunev1.FLJobCondFailed) && c.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

func (fc *FederatedController) createPod(job *neptunev1.FederatedLearningJob) (active int32, err error) {

	active = 0
	ctx := context.Background()

	modelName := job.Spec.AggregationWorker.Model.Name
	model, err := fc.client.Models(job.Namespace).Get(ctx, modelName, metav1.GetOptions{})

	var modelPath string

	if err != nil {
		modelPath = "/model"
	} else {
		modelPath = model.Spec.ModelURL
	}

	participantsCount := strconv.Itoa(len(job.Spec.TrainingWorkers))

	// convert crd to json, and put them into env of container
	modeljson, _ := json.Marshal(model)
	modelstring := string(modeljson)

	// deliver pod for aggregation worker
	aggWorker := job.Spec.AggregationWorker
	aggCodePath := aggWorker.WorkerSpec.ScriptDir
	parameterJSON, _ := json.Marshal(aggWorker.WorkerSpec.Parameters)
	parameterString := string(parameterJSON)

	// Container VolumeMounts parameters
	aggCodeConPath := codePrefix
	aggModelConPath := dataPrefix + modelPath

	// Env parameters for agg
	aggModelUrl := aggModelConPath

	// Configure container mounting and Env information by initial ContainerPara
	var aggPort int32 = 7363
	var aggContainer *ContainerPara = new(ContainerPara)
	aggContainer.volumeMountList = []string{aggCodeConPath, aggModelConPath}
	aggContainer.volumeList = []string{aggCodePath, modelPath}
	aggContainer.volumeMapName = []string{"code", "model"}
	aggContainer.env = map[string]string{
		"MODEL":              modelstring,
		"WORKER_NAME":        "aggworker-" + utilrand.String(5),
		"JOB_NAME":           job.Name,
		"PARTICIPANTS_COUNT": participantsCount,
		"PARAMETERS":         parameterString,
		"MODEL_URL":          aggModelUrl,
		"NAMESPACE":          job.Namespace,
		"AGG_BIND_PORT":      string(aggPort),
	}
	aggContainer.scriptBootFile = aggWorker.WorkerSpec.ScriptBootFile
	aggContainer.nodeName = aggWorker.NodeName
	aggContainer.frameName = aggWorker.WorkerSpec.FrameworkType
	aggContainer.frameVersion = aggWorker.WorkerSpec.FrameworkVersion

	// create aggpod based on configured parameters
	fc.generatedPod(job, FLJobStageAgg, aggContainer, &active, false)
	var aggIp string
	var aggServicePort int32

	aggIp, err = GetNodeIPByName(fc.kubeClient, job.Spec.AggregationWorker.NodeName)
	aggServicePort, err = CreateKubernetesService(fc.kubeClient, job, aggPort, aggIp)
	if err != nil {
		return active, err
	}
	// deliver pod for training worker
	lcport := 30000
	for _, trainingWorker := range job.Spec.TrainingWorkers {
		// get dataseturl through parsing crd of dataset
		parameterJSON, _ = json.Marshal(trainingWorker.WorkerSpec.Parameters)
		parameterString = string(parameterJSON)
		datasetName := trainingWorker.Dataset.Name
		dataset, err := fc.client.Datasets(job.Namespace).Get(ctx, datasetName, metav1.GetOptions{})
		datasetjson, _ := json.Marshal(dataset)
		datasetstring := string(datasetjson)
		var trainDatasetPath string
		if err != nil {
			trainDatasetPath = "/code/data"
		} else {
			trainDatasetPath = dataset.Spec.URL
		}
		datasetParent := filepath.Dir(trainDatasetPath)
		trainCodePath := trainingWorker.WorkerSpec.ScriptDir

		// Container VolumeMounts parameters
		trainCodeConPath := codePrefix
		trainDataConPath := dataPrefix + datasetParent
		trainModelConPath := dataPrefix + modelPath

		// Env parameters for train
		trainDatasetUrl := dataPrefix + trainDatasetPath
		trainModelUrl := trainModelConPath

		// Configure container mounting and Env information by initial ContainerPara
		var trainContainer *ContainerPara = new(ContainerPara)
		trainContainer.volumeMountList = []string{trainCodeConPath, trainDataConPath, trainModelConPath}
		trainContainer.volumeList = []string{trainCodePath, datasetParent, modelPath}
		trainContainer.volumeMapName = []string{"code", "data", "model"}
		trainContainer.env = map[string]string{
			"DATASET":            datasetstring,
			"AGG_PORT":           strconv.Itoa(int(aggServicePort)),
			"AGG_IP":             aggIp,
			"MODEL_URL":          trainModelUrl,
			"TRAIN_DATASET_URL":  trainDatasetUrl,
			"WORKER_NAME":        "trainworker-" + utilrand.String(5),
			"JOB_NAME":           job.Name,
			"PARAMETERS":         parameterString,
			"PARTICIPANTS_COUNT": participantsCount,
			"NAMESPACE":          job.Namespace,
			"MODEL_NAME":         modelName,
			"DATASET_NAME":       datasetName,
			"LC_PORT":            strconv.Itoa(lcport),
		}
		trainContainer.scriptBootFile = trainingWorker.WorkerSpec.ScriptBootFile
		trainContainer.nodeName = trainingWorker.NodeName
		trainContainer.frameName = trainingWorker.WorkerSpec.FrameworkType
		trainContainer.frameVersion = trainingWorker.WorkerSpec.FrameworkVersion

		// create trainpod based on configured parameters
		err = fc.generatedPod(job, FLJobStageTrain, trainContainer, &active, true)
		if err != nil {
			return active, err
		}
	}
	return

}

func (fc *FederatedController) generatedPod(job *neptunev1.FederatedLearningJob, podtype FLJobStage, containerPara *ContainerPara, active *int32, hostNetwork bool) error {
	var volumeMounts []v1.VolumeMount
	var volumes []v1.Volume
	var envs []v1.EnvVar
	var imagePullPolicy v1.PullPolicy
	ctx := context.Background()
	command := []string{"python"}
	imagePullPolicy = v1.PullAlways
	// get baseImgUrl from imageHub based on user's configuration in job CRD
	baseImgUrl, err := MatchContainerBaseImage(fc.cfg.ImageHub, containerPara.frameName, containerPara.frameVersion)
	// TODO: if matched image is empty, the pod creation process will not proceed, return error directly.
	if err != nil {
		klog.Warningf("federatedlearning job %v/%v %v worker matching container base image occurs error:%v", job.Namespace, job.Name, podtype, err)
		return err
	}
	volumeMounts, volumes = CreateVolumeMap(containerPara)
	envs = CreateEnvVars(containerPara.env)
	podSpec := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:    job.Namespace,
			GenerateName: job.Name + "-" + strings.ToLower(string(podtype)) + "-",
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(job, neptunev1.SchemeGroupVersion.WithKind("FederatedLearningJob")),
			},
			Labels: GenerateLabels(job),
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			NodeName:      containerPara.nodeName,
			Containers: []v1.Container{
				v1.Container{Name: "container-" + job.Name + "-" + strings.ToLower(string(podtype)) + "-" + utilrand.String(5),
					ImagePullPolicy: imagePullPolicy,
					Image:           baseImgUrl,
					Command:         command,
					Args:            []string{containerPara.scriptBootFile},
					Env:             envs,
					VolumeMounts:    volumeMounts,
				}},
			Volumes:     volumes,
			HostNetwork: hostNetwork,
		},
	}
	pod, err := fc.kubeClient.CoreV1().Pods(job.Namespace).Create(ctx, podSpec, metav1.CreateOptions{})
	if err != nil {
		klog.Warningf("failed to create %s pod %s for federatedlearning job %v/%v, err:%s", string(podtype), pod.Name, job.Namespace, job.Name, err)
		return err
	} else {
		klog.V(2).Infof("%s pod %s is created successfully for federatedlearning job %v/%v", string(podtype), pod.Name, job.Namespace, job.Name)
		*active += 1
		return nil
	}
}

func (fc *FederatedController) GetName() string {
	return "FederatedLearningJobController"
}

// NewFederatedController creates a new FederatedLearningJob controller that keeps the relevant pods
// in sync with their corresponding FFederatedLearningJob objects.
func NewFederatedController(cfg *config.ControllerConfig) (FeatureControllerI, error) {
	namespace := cfg.Namespace
	if namespace == "" {
		namespace = metav1.NamespaceAll
	}
	kubeClient, err := utils.KubeClient()
	kubecfg, _ := utils.KubeConfig()
	crdclient, err := clientset.NewForConfig(kubecfg)
	kubeInformerFactory := kubeinformers.NewSharedInformerFactoryWithOptions(kubeClient, time.Second*30, kubeinformers.WithNamespace(namespace))

	podInformer := kubeInformerFactory.Core().V1().Pods()

	jobInformerFactory := informers.NewSharedInformerFactoryWithOptions(crdclient, time.Second*30, informers.WithNamespace(namespace))
	jobInformer := jobInformerFactory.Neptune().V1alpha1().FederatedLearningJobs()

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})

	fc := &FederatedController{
		kubeClient: kubeClient,
		client:     crdclient.NeptuneV1alpha1(),
		podControl: k8scontroller.RealPodControl{
			KubeClient: kubeClient,
			Recorder:   eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "flJob-controller"}),
		},

		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(DefaultFLJobBackOff, MaxFLJobBackOff), "flJob"),
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "flJob-controller"}),
		cfg:      cfg,
	}

	jobInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			fc.enqueueController(obj, true)
		},
		UpdateFunc: fc.updateFLJob,
		DeleteFunc: func(obj interface{}) {
			fc.enqueueController(obj, true)
		},
	})
	fc.jobLister = jobInformer.Lister()
	fc.jobStoreSynced = jobInformer.Informer().HasSynced

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    fc.addPod,
		UpdateFunc: fc.updatePod,
		DeleteFunc: fc.deletePod,
	})
	fc.podStore = podInformer.Lister()
	fc.podStoreSynced = podInformer.Informer().HasSynced

	fc.updateHandler = fc.updateFLJobStatus
	fc.syncHandler = fc.syncFLJob

	stopCh := make(chan struct{})
	kubeInformerFactory.Start(stopCh)
	jobInformerFactory.Start(stopCh)
	return fc, err
}
