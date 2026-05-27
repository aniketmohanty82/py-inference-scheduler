# verl Integration with py-inference-scheduler

## Compatibility Notice

**This integration is designed specifically for [vERL v0.7.1](https://github.com/volcengine/verl/releases/tag/v0.7.1).** 

It utilizes internal API signatures (such as `load_balancer_handle` and specific `_acquire_server` patterns) that were introduced or modified in this release. It is **not backwards compatible** with earlier versions of vERL and may require updates for future releases.

## Architecture

`verl` manages its own set of Ray Actors for rollouts. By default, it uses a simple LRU cache or least-requests routing. This integration overrides `verl`'s `AgentLoopManager` to delegate routing decisions to the `py-inference-scheduler` engine.

Key components:
- `verl_hook.py`: Contains `InferenceSchedulerServerManager` and `PyInferenceAgentLoopManager` which are injected into the `verl` training loop.
- `InflightStore`: Tracks active requests per worker in real-time to augment slow Prometheus metrics.
- `backends/verl/`: Contains monkey-patches for `vllm` and `sglang` to enable metrics extraction and correct environment propagation.
- `datalayer/metrics/verl/`: Contains backend-specific logic (HTTP scraping) to fetch and parse metrics from the workers.

## Setup Guide

If you have a `verl` instance and want to enable scheduling:

### Requirements

To use this integration, you can use the official pre-built images from `verlai/verl` on Docker Hub. You do **not** need to build custom images!

#### Supported Official Images
- **vLLM Backend**: `verlai/verl:vllm011.latest` 
  - (Digest: `sha256:3ce56ff018516b28ab9c4f4fc09d3aa67589074495ace75e2674b720aa4d0e5d`)
- **SGLang Backend**: `verlai/verl:sgl059.latest` 
  - (Digest: `sha256:7d6502f9a46353792d1c9c855b61c1a9ea29ad74c5cb246e8aa9ac29b30372eb`)

Besides that, the main requirement is having a `runtime-env.yaml` file which has the `working_dir` as the main branch of the scheduler repository (shown below). This will in the future be a package that can be loaded in as a package using `uv` inside `runtime-env`. `runtime-env` is passed in during job submission and has to be in the root of whichever directory you are submitting the job from. 

### Data Preparation

If you want to follow along with the provided examples, you will need to preprocess the **GSM8K** and **MATH** datasets (both are used in the example scripts). You can use `verl`'s data preparation scripts to generate these datasets and store them in your preferred directory.

If you don't have the `verl` repository cloned, you can download the specific scripts directly from GitHub:
- [gsm8k.py](https://github.com/verl-project/verl/blob/v0.7.1/examples/data_preprocess/gsm8k.py)
- [math_dataset.py](https://github.com/verl-project/verl/blob/v0.7.1/examples/data_preprocess/math_dataset.py)

Wherever you choose to store the data, it must be accessible by the cluster pods. We recommend using a **PersistentVolumeClaim (PVC)** or a **Cloud Bucket (e.g., GCS)** for this purpose.

> [!IMPORTANT]
> Once you have stored your data and made it accessible to the pods, you **must** update the data path variables at the top of the example scripts ([run_qwen2_5-32b_math.sh](https://github.com/llm-d-incubation/py-inference-scheduler/blob/main/integration/verl/examples/run_qwen2_5-32b_math.sh) or [run_qwen2_5-32b_math_sglang.sh](https://github.com/llm-d-incubation/py-inference-scheduler/blob/main/integration/verl/examples/run_qwen2_5-32b_math_sglang.sh)) to point to the internal location within the pod where the data is accessible (e.g., `/home/ray/data/...`).

### Configure the Ray Cluster

When configuring your Ray cluster (e.g., via KubeRay), you must ensure that your configuration:
- Uses one of the **supported official images** listed in the [Requirements](#requirements) section.
- Has a **shared volume** (or equivalent access) for the data to be trained on, accessible by all pods. An example of how a GCS bucket can be used to store locally preprocessed data is available in [verl-inference-scheduler.yaml](https://github.com/llm-d-incubation/py-inference-scheduler/blob/main/integration/verl/examples/verl-inference-scheduler.yaml).
- Mounts the ConfigMap for custom scheduler profiles if you choose to use one (see [Customizing Scheduler Configuration](#customizing-scheduler-configuration)). If you want to see an example of how to do these ConfigMap mounts, you can check [verl-inference-scheduler.yaml](https://github.com/llm-d-incubation/py-inference-scheduler/blob/main/integration/verl/examples/verl-inference-scheduler.yaml).

If you would like to see an example of a cluster configuration file, you can check [verl-inference-scheduler.yaml](https://github.com/llm-d-incubation/py-inference-scheduler/blob/main/integration/verl/examples/verl-inference-scheduler.yaml).

To apply your configuration to the Kubernetes cluster:
```bash
kubectl apply -f <your-cluster-config>.yaml
```

### Establish Head Node Connection

Wait for your pods to reach **Running** state. Now you must open a local tunnel to the Ray Head Node dashboard.

- **Establish port forward**:
    ```bash
    kubectl port-forward svc/<ray-head-svc-name> 8265:8265 -n <namespace> &
    ```
- **Export the Ray address**:
    ```bash
    export RAY_ADDRESS="http://127.0.0.1:8265"
    ```



### Runtime Environment (`runtime-env.yaml`)

To run the job with the scheduler, you need a `runtime-env.yaml` file. This file configures the environment for the Ray job.

Example `runtime-env.yaml`:
```yaml
working_dir: "https://github.com/llm-d-incubation/py-inference-scheduler/archive/refs/heads/main.zip"

pip:
  - "verl==0.7.1"

env_vars:
  PYTHONPATH: "."
  PROMETHEUS_MULTIPROC_DIR: "/tmp/metrics"
  ROUTER_CONFIG_PATH: "./integration/verl/examples/scheduler.yaml"
```

*   **`working_dir`**: Set this to point to the root of the `py-inference-scheduler` repository (or a remote zip URL like above). This will be replaced by the package in the pip/uv section later.
*   **`pip`**: Used to install `verl==0.7.1` at runtime.
*   **`PROMETHEUS_MULTIPROC_DIR`**: Must be set to `/tmp/metrics` for multiprocess metrics aggregation.
*   **`ROUTER_CONFIG_PATH`**: Points to the scheduler configuration file.
### Customizing Scheduler Configuration

By default, the system uses the `scheduler.yaml` provided in the repository. If you want to use a custom configuration (e.g., to change scoring plugins or thresholds), you can do so using a Kubernetes ConfigMap:

- **Get the default config**: Download [scheduler.yaml](https://github.com/llm-d-incubation/py-inference-scheduler/blob/main/integration/verl/examples/scheduler.yaml) locally.
- **Edit the config**: Make your desired changes to the file.
- **Apply as ConfigMap**: Upload it to your Kubernetes cluster:
    ```bash
    kubectl create configmap my-custom-scheduler --from-file=scheduler.yaml=scheduler.yaml
    ```
- **Update cluster config**: Ensure your `verl-inference-scheduler.yaml` mounts this ConfigMap (the example file is pre-configured to look for `my-custom-scheduler`).
- **Update `runtime-env.yaml`**: Set the `ROUTER_CONFIG_PATH` to point to the mounted file:
    ```yaml
    env_vars:
      ROUTER_CONFIG_PATH: "/etc/scheduler/scheduler.yaml"
    ```

### Running a Training Job

We provide pre-configured shell scripts for both vLLM and SGLang. You can download them directly from GitHub:
- [run_qwen2_5-32b_math.sh](https://github.com/llm-d-incubation/py-inference-scheduler/blob/main/integration/verl/examples/run_qwen2_5-32b_math.sh) (vLLM)
- [run_qwen2_5-32b_math_sglang.sh](https://github.com/llm-d-incubation/py-inference-scheduler/blob/main/integration/verl/examples/run_qwen2_5-32b_math_sglang.sh) (SGLang)

After downloading and editing the scripts (e.g., to set the correct data paths as mentioned in the [Data Preparation](#data-preparation) section), you can submit the job by pointing to your edited script:

```bash
ray job submit \
    --address http://localhost:8265 \
    --runtime-env ./runtime-env.yaml \
    -- bash path_to_your_scripts/run_qwen2_5-32b_math.sh
```

> [!NOTE]
> Ensure you are in the directory containing your `runtime-env.yaml` file when running this command.

### View the results

This is a very small training loop for testing with 10 steps configured in the `trainer.total_training_steps=10` flag. Viewing the logs on the actual ray submit job where you've ran the training job is the best place for logs in my opinion. This can be found either in the *Overview* or *Jobs* tab of the Ray Dashboard. (localhost:8265). vERL gives us output by step, don't be concerned if you se `ppo` tags on the labels for the logs - vERL uses the same testing infrastructure for its GRPO and PPO runs. Our script is a GRPO trainer.

Logs look as follows (from verl-vllm using gsm8k):

```bash
(TaskRunner pid=15930, ip=10.4.1.33) step:1
 - global_seqlen/min:255486
 - global_seqlen/max:304956
 - global_seqlen/minmax_diff:49470
 - global_seqlen/balanced_min:274237
 - global_seqlen/balanced_max:274268
 - global_seqlen/mean:274260.8125
 - actor/entropy:0.36423397064208984
 - perf/mfu/actor_infer:0
 - actor/pg_loss:0.05410893690350349
 - actor/kl_loss:0.0013340971445359173
 - actor/pg_clipfrac:0.00044460127605816524
 - actor/ppo_kl:2.8409333591383756e-05
 - actor/pg_clipfrac_lower:0.0
 - actor/kl_coef:0.0010000000000000005
 - actor/grad_norm:0.167236328125
 - perf/mfu/actor:0.3707644223640674
 - perf/max_memory_allocated_gb:86.77569150924683
 - perf/max_memory_reserved_gb:94.625
 - perf/cpu_memory_used_gb:144.76549291610718
 - actor/lr:1e-06
 - training/global_step:1
 - training/epoch:0
 - critic/score/mean:0.5760498046875
 - critic/score/max:1.0
 - critic/score/min:0.0
 - critic/rewards/mean:0.5760498046875
 - critic/rewards/max:1.0
 - critic/rewards/min:0.0
 - critic/advantages/mean:-0.0660763531923294
 - critic/advantages/max:2.4748666286468506
 - critic/advantages/min:-2.4748666286468506
 - critic/returns/mean:-0.0660763531923294
 - critic/returns/max:2.4748666286468506
 - critic/returns/min:-2.4748666286468506
 - response_length/mean:424.5494384765625
 - response_length/max:1024.0
 - response_length/min:26.0
 - response_length/clip_ratio:0.07373046875
 - response_length_non_aborted/mean:424.5494384765625
 - response_length_non_aborted/max:1024.0
 - response_length_non_aborted/min:26.0
 - response_length_non_aborted/clip_ratio:0.07373046875
 - response/aborted_ratio:0.0
 - prompt_length/mean:111.1162109375
 - prompt_length/max:994.0
 - prompt_length/min:42.0
 - prompt_length/clip_ratio:0.0
 - num_turns/min:2
 - num_turns/max:2
 - num_turns/mean:2.0
 - timing_s/start_profile:0.00027797603979706764
 - timing_s/agent_loop/num_preempted/min:-1
 - timing_s/agent_loop/num_preempted/max:-1
 - timing_s/agent_loop/num_preempted/mean:-1.0
 - timing_s/agent_loop/generate_sequences/min:1.597659296123311
 - timing_s/agent_loop/generate_sequences/max:20.618110499111935
 - timing_s/agent_loop/generate_sequences/mean:8.733964191658202
 - timing_s/agent_loop/tool_calls/min:0.0
 - timing_s/agent_loop/tool_calls/max:0.0
 - timing_s/agent_loop/tool_calls/mean:0.0
 - timing_s/agent_loop/slowest/generate_sequences:20.618110499111935
 - timing_s/agent_loop/slowest/tool_calls:0.0
 - timing_s/agent_loop/slowest/prompt_length:54
 - timing_s/agent_loop/slowest/response_length:1024
 - timing_s/agent_loop/slowest/num_preempted:-1
 - timing_s/gen:24.533394969068468
 - timing_s/reward:3.863382153213024e-05
 - timing_s/old_log_prob:13.635552056133747
 - timing_s/ref:12.547897297888994
 - timing_s/adv:0.17872913694009185
 - timing_s/update_actor:35.83383133285679
 - timing_s/update_weights:5.458046867046505
 - timing_s/step:93.52434759191237
 - timing_s/stop_profile:5.5649783462285995e-05
 - timing_per_token_ms/gen:0.0070540646604233944
 - timing_per_token_ms/adv:4.0729738080082955e-05
 - timing_per_token_ms/ref:0.0028594809953684584
 - timing_per_token_ms/update_actor:0.008166002418969533
 - perf/total_num_tokens:4388173
 - perf/time_per_step:93.52434759191237
 - perf/throughput:2932.5070910595373
 ```

 Key metrics to look out for are ```perf/throughput``` for sampling throughput and ```timing_s/agent_loop/slowest/generate_sequences``` for your tail latency. Additionally, if you enable preemptions, ```timing_s/agent_loop/slowest/num_preempted``` can be useful too. 