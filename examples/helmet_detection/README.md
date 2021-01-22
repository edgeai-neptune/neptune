# Using Incremental Learning Job in Helmet Detection Scenario

This document introduces how to use incremental learning job in helmet detectioni scenario. Using the incremental learning job, our application can automatically retrains, evaluates, and updates models based on the data generated at the edge.

## Helmet Detection Experiment

### Install Neptune

Follow the [Neptune installation document](/docs/setup/install.md) to install Neptune.

### Prepare Data and Model

Download dataset and model to your node:


### Create Incremental Job

Create Dataset

```
kubectl create -f dataset.yaml
```

```yaml
apiVersion: solarcorona.io/v1alpha1
kind: Dataset
metadata:
  name: incremental-dataset
  namespace: neptune-test
spec:
  dataUrl: "/data/train_hard_img_IBT_1002"
  format: "txt"
  nodeName: "edge1"
```

Create Initial Model

```
kubectl create -f initial-model.yaml
```

```yaml
apiVersion: solarcorona.io/v1alpha1
kind: Model
metadata:
  name: initial-model
  namespace: neptune-test
spec:
  modelUrl : "/model/initial"

```

Create Deploy Model

```
kubectl create -f deploy-model.yaml
```

```yaml
apiVersion: solarcorona.io/v1alpha1
kind: Model
metadata:
  name: deploy-model
  namespace: neptune-test
spec:
  modelUrl : "/deploy"

```

Start The Incremental Job

```
kubectl create -f incrementaljob.yaml
```

```yaml
apiVersion: solarcorona.io/v1alpha1
kind: IncrementalJob
metadata:
  name: helmet-detection-demo
  namespace: neptune-test
spec:
  initialModel:
    name: "initial-model"
  dataset:
    name: "incremental-dataset"
    trainProb: 0.8
  trainSpec:
    workerSpec:
      scriptDir: "/code"
      scriptBootFile: "train.py"
      frameworkType: "tensorflow"
      frameworkVersion: "1.15"
      parameters:
        - key: "batch_size"
          value: "32"
        - key: "learning_rate"
          value: "0.001"
        - key: "max_epochs"
          value: "100"

    trigger:
      checkPeriodSeconds: 60
      timer:
        start: 02:00
        end: 04:00
      condition:
        operator: ">"
        threshold: 500
        metric: num_of_samples
  evalSpec:
    workerSpec:
      scriptDir: "/code"
      scriptBootFile: "eval.py"
      frameworkType: "tensorflow"
      frameworkVersion: "1.15"

  deploySpec:
    model:
      name: "deploy-model"
    trigger:
      condition:
        operator: ">"
        threshold: 0.1
        metric: precision_delta

  nodeName: "solar-corona-cloud"
  outputDir: "/output"

```
Ensure that the path of outputDir in the YAML file exists on your edge node. This path will be directly mounted to the container



### Mock Video Stream for Inference in Edge Side

* step1: install the open source video streaming server [EasyDarwin](https://github.com/EasyDarwin/EasyDarwin/tree/dev).
* step2: start EasyDarwin server.
* step3: download [video](https://edgeai-neptune.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection-inference/video.tar.gz).
* step4: push a video stream to the url (e.g., `rtsp://localhost/video`) that the inference service can connect.

```
wget https://github.com/EasyDarwin/EasyDarwin/releases/download/v8.1.0/EasyDarwin-linux-8.1.0-1901141151.tar.gz --no-check-certificate
tar -zxvf EasyDarwin-linux-8.1.0-1901141151.tar.gz
cd EasyDarwin-linux-8.1.0-1901141151
./start.sh

mkdir -p /data/video
cd /data/video
tar -zxvf video.tar.gz
ffmpeg -re -i /data/video/helmet-detection.mp4 -vcodec libx264 -f rtsp rtsp://localhost/video
```


### Check Incremental Job Result

query the service status
```
kubectl get incrementaljob helmet-detection-demo -n neptune-test
```

after the job completed, we can view the updated model in the /output directory in solar-corona-cloud node

