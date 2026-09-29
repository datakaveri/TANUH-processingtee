# TANUH benchmarking: current setup, pending work, and Option C

This document covers four things:

1. **The current setup:** how a benchmarking job runs today, from upload to leaderboard.
2. **Pending work:** what is still needed outside the code for that setup to work end to end.
3. **Option C:** a proposed change where the platform applies the standard model preprocessing, so model owners don't re-export their models.
4. **Option C changes:** the extra work Option C needs.
5. **Instructions for model providers:** what to give owners for each model format, both today and with Option C.

---

## 1. Key terms

| Term | Meaning |
|---|---|
| **Dataset** | Files such as images or DICOM scans, plus `ground_truth.csv`, which holds the correct answer for each file. Uploaded encrypted by a data provider. |
| **Model** | A trained file that takes an image and returns numbers. It comes in one of three formats: ONNX, TorchScript or Hugging Face. |
| **Preprocessing** | Converting a raw image into the exact format the model was trained on: size, colour order, number range. If it's wrong, **nothing fails but the score is lower**. Think of it as sending Fahrenheit to an API trained on Celsius. |
| **Adaptor** (`adaptor.py`) | A small script from the model owner. It turns the model's raw numbers into a standard `predictions.csv`. |
| **Bucket** | A problem type, such as `binary_classification` or `multiclass_classification`. Each bucket has one evaluation script, which we own. |
| **Evaluator** (`evaluate.py`) | Compares `predictions.csv` with `ground_truth.csv` and computes metrics such as accuracy and F2. |
| **Catalogue** | Stores each dataset's settings: its bucket, class names and required metrics. |

---

## 2. Current setup

### 2.1 Components

| Component | What it does |
|---|---|
| **UI** (`ui-dx`) | Data providers onboard datasets. Model owners submit models and see results. |
| **Buffer TEE** (`Buffer_TEE/.../enclave`) | Receives a model submission, stores the uploads, checks their hashes, and sends the job to a Processing TEE. |
| **Processing TEE** (`Processing_TEE`) | Decrypts the dataset, runs the model, adaptor and evaluator, and posts the score to the leaderboard. |
| **Catalogue** | Holds the dataset settings the Processing TEE reads at job time. |
| **Leaderboard** (`tanuh-leaderboards-apis`) | Stores and ranks results. It checks that the metrics match what the catalogue requires. |
| **GCS** | Encrypted datasets are in `gs://file-server-data/<dataset-uuid>/`; evaluators are in `gs://tanuh-evaluators/<bucket>/evaluate.py`. |
| **Cloud KMS** | Holds the key that unlocks each dataset file's encryption key. |

### 2.2 What each party provides

**The data provider**, once, at onboarding:
- **Data files:** one encrypted object per file (`.jpg`/`.jpeg`/`.png` or `.dcm`), each with its own `<name>.manifest.json`.
- **`ground_truth.csv`:** encrypted the same way, with columns `file,label`. `label` is the class number: 0, 1, …
- **Catalogue fields:**

  | Field | Example |
  |---|---|
  | `task_type` | `binary_classification` |
  | `class_names` | `["Non-Suspicious","Suspicious"]`; position = label number |
  | `datasetMetrics`, `primaryMetric` | the required metrics, and the one used to rank |

**The model owner**, per submission:

| Upload slot | ONNX | TorchScript | Hugging Face |
|---|---|---|---|
| `model` | `.onnx` | `.pt` (with `tanuh.json` embedded) | `.zip` of a `save_pretrained()` folder |
| `weights` | `.onnx.data` (optional) | not allowed | not allowed |
| `adaptor` | `adaptor.py` (required) | `adaptor.py` (required) | `adaptor.py` (required) |

For the exact per-format rules, see [model-submission.md](model-submission.md).

### 2.3 How a job runs

1. **Submit (UI → Buffer).**
   - The UI sends `POST /v1/submit` with `{dataset_id, model_format, model_sha256, adaptor_sha256, weights_sha256?}`.
   - The Buffer creates a job in `pending_upload`.
2. **Upload (UI → Buffer).**
   - The UI sends one `PUT /v1/upload/{job_id}/{slot}` per file.
   - The Buffer rejects files it didn't expect, files that are too big, and hashes that don't match what was submitted.
   - When every expected file has arrived, the job becomes `queued`.
3. **Dispatch (Buffer → Processing).**
   - The Buffer re-checks every hash and builds **payload v2**:

     ```json
     {"payload_version": 2, "job_id": "…", "dataset_id": "<uuid>", "model_format": "onnx",
      "keycloak_token": "…", "buffer_job_url": "…",
      "artifacts": {"model": {"sha256": "…", "base64": "…"}, "adaptor": {…}, "weights": {…}}}
     ```

   - It sends the payload to a Processing TEE (`POST /api/load-model`), and the job becomes `dispatched`.
4. **Prepare (Processing, Go).**
   1. **Read the catalogue:** `GET cat/item?id=<uuid>` returns `task_type`, `class_names`, the required metrics and the primary metric.
   2. **Fetch the evaluator:** download the bucket's `evaluate.py` from GCS, recording its hash and version.
   3. **Check and place the model:** the file must really be the declared format. Unsafe Hugging Face zips are rejected.
   4. **Install adaptor libraries:** install any libraries `adaptor.py` imports that the image doesn't have (`dep_scanner.py`).
   5. **Decrypt the dataset:** list the dataset folder, keep only allowed files, and decrypt 16 at a time into a temporary job folder.
   6. **Match the ground truth:** read `ground_truth.csv` and check that every row matches a data file.
5. **Run three stages (Processing, Python).** Each stage is a separate `python3` process with its own timeout.

   | Stage | Command | Output |
   |---|---|---|
   | 1. Inference (our code) | `infer.py --format <fmt> --model-dir … --inputs inputs.txt --output-dir raw/` | `raw/raw_outputs.npz` + `raw/meta.json` |
   | 2. Adaptor (owner's code) | `adaptor.py --raw-dir raw/ --spec dataset_spec.json --output-dir predictions/` | `predictions/predictions.csv` |
   | 3. Evaluator (our code) | `evaluate.py --predictions … --ground-truth … --spec … --results results.json` | `results.json` |

   **Inside inference, for ONNX and TorchScript,** each image goes through these steps:
   1. Decode to RGB. For DICOM: take the middle frame, invert MONOCHROME1, min-max scale to 0–255, and copy grey into 3 channels.
   2. Resize to the model's size, using bilinear or bicubic as the model declares.
   3. Scale to 0–1 (for float inputs).
   4. Arrange the channels and layout the model expects.
   5. Run the model.

   **Anything else the model needs, such as ImageNet normalisation or BGR order, must already be inside the model file.**

   Hugging Face models use their own `preprocessor_config.json` instead.

6. **Report (Processing → Leaderboard → Buffer).**
   1. Rename the evaluator's metric names to the catalogue's keys, for example `f2` → `f2_score`.
   2. Check each value is in range, and send **exactly** the required keys to `POST /submit-solution`.
   3. Tell the Buffer the job finished (`POST /v1/jobs/{job_id}/complete`). The job becomes `complete` or `error`.
   4. Delete the decrypted dataset folder.

### 2.4 Predictions format per bucket

| Bucket | `predictions.csv` columns | Metrics the evaluator computes |
|---|---|---|
| `binary_classification` | `file,label,score` (`score` = probability of class 1) | accuracy, sensitivity, specificity, ppv, npv, fnr, fpr, f1, f2, auc, tp/tn/fp/fn, confusion matrix |
| `multiclass_classification` | `file,label,prob_0,…,prob_{C-1}` (probabilities sum to 1) | accuracy, macro/weighted F1 and F2, sensitivity, specificity, ppv, npv, auc, qwk, confusion matrix, per-class |

### 2.5 Failures and error codes (sent to the leaderboard)

| Code | Meaning | Examples |
|---|---|---|
| 1 | Data or pre/post-processing problem | image can't be decoded; adaptor crashed or wrote bad predictions |
| 2 | Model problem | wrong file format; model fails to load or run |
| 3 | Platform or runtime problem | GPU error, timeout, evaluator error, metric can't be mapped |

### 2.6 What is built and tested

- **Buffer:**
  - upload slots per format and payload v2
  - tests pass
- **Processing:**
  - catalogue reader, evaluator fetch, model checks, parallel decrypt, ground-truth matching, the three stages, metric mapping, error codes
  - tests pass
- **Python:**
  - `infer.py` (all three formats), both evaluators, reference adaptors, and the wrap tools
  - tests pass
- **Results match the old flow on real data:**

  | Case | Result |
  |---|---|
  | OCS ONNX | identical labels and metrics |
  | OCS TorchScript | identical labels; AUC differs because the old code had a rounding flaw |
  | BCD ONNX | identical metrics |

- **Not done here:** building the Docker images (done separately).

---

## 3. Pending work for the current setup to work end to end

### 3.1 Catalogue: add `task_type` and `class_names`

The Processing TEE reads these fields at job time; they are not passed with the job. Without them, every job fails.

- **UI:** add the two fields to the dataset metadata form, and send them as top-level fields in `buildTanuhDatasetItemPayload` (`ui-dx/apps/TANUH/src/app/shared/utils/tanuh-dataset-item-payload.util.ts`). The edit flows (`updateItem`) must keep them.
- **Existing OCS and BCD items:** add the fields with a `PUT cat/item`.

  | Dataset | `task_type` | `class_names` |
  |---|---|---|
  | OCS | `binary_classification` | `["Non-Suspicious","Suspicious"]` |
  | BCD | `multiclass_classification` | `["A","B","C","D"]` |

- **Check with the catalogue owner** that top-level fields are accepted.

### 3.2 Evaluators bucket

```bash
gcloud storage buckets create gs://tanuh-evaluators --project=proj-tanuh-benchmark-ptfm --location=asia-south1 --uniform-bucket-level-access
gcloud storage buckets update gs://tanuh-evaluators --versioning
gcloud storage cp evaluation_LEGACY/binary_classification/evaluate.py     gs://tanuh-evaluators/binary_classification/evaluate.py
gcloud storage cp evaluation_LEGACY/multiclass_classification/evaluate.py gs://tanuh-evaluators/multiclass_classification/evaluate.py
gcloud storage buckets add-iam-policy-binding gs://tanuh-evaluators \
  --member=serviceAccount:<service account that reads gs://file-server-data> --role=roles/storage.objectViewer
```

To find the service account, run `gcloud storage buckets get-iam-policy gs://file-server-data`. It is probably `gpu-cs-sa@p3dx-depa-sandbox.iam.gserviceaccount.com`.

### 3.3 Metric ranges (QWK)

**The rule:** the leaderboard treats a bare number in `datasetMetrics` as "required, 0–1". QWK can be negative, so it needs the string form `"-1-1"`.

**What to change in the UI** (same file as 3.1):
- `buildTanuhDatasetMetricsFromForm` and `buildCustomMetricsMap` should write `"min-max"` strings when a range is given.
- Today the UI puts ranges in `customMetricRanges`, which the leaderboard ignores.

**Existing BCD item:** update it with a `PUT`.

### 3.4 Models: add their preprocessing

The trained models don't include the preprocessing the old evaluators did for them. Uploaded as they are, they score lower with no error: the OCS model drops from accuracy 0.99 to 0.76.

For the first runs, we wrap them before upload, on a normal machine:

| Model | Command |
|---|---|
| `TANUH files/Oral Cancer/mobilevitv2_ocs_trained.onnx` | `wrap_onnx.py --mode ocs` |
| `TANUH files/Breast Cancer/resnet34_density_trained.onnx` | `wrap_onnx.py --mode imagenet` |
| `Data/…/mvit2_fold5_2_latest_traced.pt` | `wrap_torchscript.py --mode ocs` |

**With Option C, the BCD step and every standard model like it goes away** (see section 4).

### 3.5 UI changes

**Dataset onboarding**
1. **Metadata form:** add a "Problem type" dropdown (`task_type`) and a `class_names` list (3.1).
2. **Metric ranges:** as in 3.3.
3. **Ground truth** (`upload.component.ts`):
   - accept only CSV, with columns `file,label` and integer labels
   - check it before encrypting
   - store it as `ground_truth.csv`; `GROUND_TRUTH_STORAGE_NAME` is still `ground_truth.json`
4. **Data files:** accept `.dcm`, `.jpg` and `.png`.
5. **Encryption:** set `encryptedDatasetUpload: true` in `environment.prod.ts` and `environment.stg.ts`. Today it is `false`, so datasets upload unencrypted and the TEE can't read them.

**Model submission** (`reference-datasets.component.ts`, `lib/transfer-onnx.ts`)

6. **Dataset picker:** list datasets from the catalogue and send the dataset **UUID** as `dataset_id`. Today it is a hardcoded `1 | 2`.
7. **Model format:** add a selector (ONNX / TorchScript / Hugging Face), with file inputs per format as in 2.2.
8. **Adaptor:** replace the preprocessing upload with a required `adaptor.py` for every dataset.
9. **Submit body and uploads:**
   - the new submit body from 2.3
   - upload to `/adaptor` instead of `/preprocessing`
   - send `/weights` only for ONNX
10. **Results:** stop polling `GET /v1/results`; nothing ever writes that file. Instead:
    - poll `GET /v1/status/{job_id}` until the job is `complete` or `error`
    - then show the result from the leaderboard's `my-submissions?dataset_id=<uuid>`
    - show the dataset's primary metric, not `accuracy`
11. **Buffer image digest:** update `EXPECTED_IMAGE_DIGEST` in `lib/constants.ts` after the Buffer image is rebuilt.

**Leaderboard pages**

12. **Dataset IDs:** use the catalogue UUID instead of the `oral_cancer` / `breast_cancer` slugs (`hackathon-datasets.data.ts`).
13. **Publishing:** a submission only shows publicly after `PUT /select-submission`, so add a "select" action.

### 3.6 Release order

1. Upload the evaluators and set their access (3.2).
2. Update the catalogue items (3.1, 3.3).
3. Build and deploy the Processing image.
4. Put the new Processing image digest into the Buffer config.
5. Build and deploy the Buffer. Empty the job queue first: old and new payloads are not compatible.
6. Update the UI digest, then deploy the UI.
7. Test jobs:
   - OCS × ONNX
   - OCS × TorchScript
   - BCD × ONNX
   - a Hugging Face test model

### 3.7 Known gaps, not blocking

- **Adaptor has full access:** it is the model owner's code, and it runs with the Processing TEE's own permissions. It can read the decrypted dataset and the ground truth; only the VPC firewall limits its network. The planned sandbox would fix this.
- **Token leak:** `GET /v1/status/{job_id}` on the Buffer returns `keycloak_token`.
- **Data residency:** data stored in asia-south1 is processed on us-central1 VMs.

---

## 4. Option C: the platform applies standard preprocessing

### 4.1 The problem it solves

**Today:** every ONNX and TorchScript model must have its preprocessing built in, so model owners have to edit and re-save their model. Most models only need standard steps (resize, colour order, "ImageNet" normalisation), so this is a lot of work for something routine.

**Option C:** the model owner uploads the model **as it is** and picks their preprocessing from a short list of settings. **Our code** applies them in the Processing TEE.

- **Model owner:** fills in settings; no code, no model editing.
- **Platform:** a fixed, small set of steps. The code doesn't grow with every new model.
- **Unusual preprocessing:** something like OCS's colour balancing still goes inside the model (today's approach), with the settings left at their defaults.

The same pattern is used by TensorFlow Lite (model metadata), Core ML (`ImageType`), OpenVINO (mean/scale flags) and Hugging Face (`preprocessor_config.json`).

### 4.2 The input spec (`input_spec.json`)

A small JSON file the UI builds from the form:

```json
{
  "spec_version": 1,
  "resize":        "bilinear",
  "channel_order": "RGB",
  "normalize":     {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
  "input_shape":   [3, 224, 224],
  "layout":        "NCHW"
}
```

| Field | Values | Default | Notes |
|---|---|---|---|
| `resize` | `bilinear`, `bicubic` | `bilinear` | How the image is resized. |
| `channel_order` | `RGB`, `BGR` | `RGB` | Some models were trained on BGR (the OpenCV default). |
| `normalize` | `null`, or `{mean, std}` with one value per channel | `null` | Applied to 0–1 values. The "ImageNet" preset uses the values shown above. |
| `input_shape` | `[C, H, W]` | read from the model | **TorchScript only:** ONNX declares it and must match if both are given. Replaces `tanuh.json`. |
| `layout` | `NCHW`, `NHWC` | read from the model | **TorchScript only.** |

**Rules:**
- **Validation:** fields not in this table are rejected, and wrong values fail the job with a clear error before the model runs.
- **Optional slot:** without it, the model is treated as having its preprocessing built in, which is the current behaviour.
- **Formats:** ONNX and TorchScript only. Hugging Face already has `preprocessor_config.json`.

**UI presets:**
1. "My model includes its own preprocessing": no spec is sent.
2. "ImageNet standard".
3. "Custom": the owner enters the values.

### 4.3 The flow with Option C

The job flow from section 2.3 changes in only three places, marked **(new)** below.

1. **Submit:** the UI also sends `input_spec_sha256` **(new)**.
2. **Upload:** the UI also uploads `PUT /v1/upload/{job_id}/input_spec` **(new)**. The Buffer checks it like every other file (hash and size), and it goes into payload v2 under `artifacts.input_spec`.
3. **Dispatch:** unchanged.
4. **Prepare (Go):** unchanged, plus **(new)** Go checks `input_spec.json` and writes it next to the model. For TorchScript, `tanuh.json` is no longer required when the spec provides `input_shape`.
5. **Inference:** `infer.py` gets `--input-spec input_spec.json` **(new)**. Each image goes through:

   | Step | Done by | Change |
   |---|---|---|
   | 1. Decode to RGB (DICOM rules unchanged) | platform | same as today |
   | 2. Resize to the model's size, with `resize` | platform | method now from the spec |
   | 3. Reorder colours to `channel_order` | platform | **new** |
   | 4. Scale to 0–1 | platform | same as today |
   | 5. `(x − mean) / std` from `normalize` | platform | **new** |
   | 6. Arrange the `layout`, convert to the model's number type | platform | same as today |
   | 7. Run the model | model | same as today |

6. **Adaptor, evaluator, leaderboard:** unchanged. `results.json` also records the spec and its hash, so every score can be reproduced.

### 4.4 Examples

| Model | With Option C |
|---|---|
| **BCD ResNet34** (ImageNet normalisation, bilinear resize) | Upload the original `resnet34_density_trained.onnx`, with the preset "ImageNet standard". **No wrap needed.** |
| **OCS MobileViTV2** (custom colour balancing + BGR + bicubic) | Colour balancing is not a standard step, so wrap once with `wrap_onnx.py --mode ocs` and choose "My model includes its own preprocessing". Same as today. |
| **A typical new model from a new team** (ImageNet normalisation) | Upload the original file and pick the preset. |
| **Any Hugging Face model** | Nothing changes. |

### 4.5 What does not change

- The datasets, catalogue, buckets, evaluators, adaptors and leaderboard.
- Models that already include their preprocessing keep working without a spec.
- The DICOM and image decoding rules.

### 4.6 Risks

- **A wrong setting still lowers the score silently**, just like a wrongly exported model. To reduce it:
  - **Before upload:** owners run `check_model.py --input-spec …` locally on their own sample images.
  - **After scoring:** the chosen settings are shown with the result.
- **Our code must match common training libraries.** Resize results differ slightly between libraries. We use PIL for bilinear and OpenCV for bicubic, as today, and write this down in the provider docs.

---

## 5. Extra changes for Option C

### 5.1 Processing TEE

| Change | Where |
|---|---|
| Read `input_spec` from the payload and check it strictly (known fields only, valid values, `mean`/`std` length = channels) | `internal/pipeline/pipeline.go` (`materialize`), plus a new `internal/inputspec` package |
| TorchScript: stop requiring `tanuh.json` when the spec has `input_shape` | `internal/modelpkg/modelpkg.go` |
| Add `--input-spec`; apply channel order and normalisation; the spec sets resize, and for TorchScript the shape | `tools/infer/infer.py` (`InputSpec`, `to_tensor`) |
| Record the spec and its hash in `results.json` | `internal/pipeline/pipeline.go` |
| Refuse a spec with Hugging Face (format error) | `internal/pipeline/pipeline.go` |
| Support `--input-spec` in the owner's local check | `tools/check_model.py` |

### 5.2 Buffer TEE

| Change | Where |
|---|---|
| New slot `input_spec`: allowed for ONNX and TorchScript, optional (required if a hash was submitted), max 64 KiB | `internal/jobs/artifacts.go` |
| Add `input_spec_sha256` to the submit request and the job record | `internal/server/submit.go`, `internal/jobs/jobs.go` |
| Include it in payload v2 (automatic once the slot exists) | `internal/scheduler/scheduler.go` |

### 5.3 UI

| Change | Where |
|---|---|
| "Input preprocessing" section on the model submit form: presets + custom fields; shape and layout for TorchScript only; hidden for Hugging Face | `reference-datasets.component.ts` / `.html` |
| Build `input_spec.json`, hash it, send `input_spec_sha256`, upload to `/input_spec` | `lib/transfer-onnx.ts` |
| Show the chosen settings with the result | results view |

### 5.4 Tests

- **Go:**
  - spec validation (good and bad cases)
  - Buffer slot rules (spec with Hugging Face rejected; spec hash without upload keeps the job waiting)
  - TorchScript without `tanuh.json` but with a spec
- **Python:**
  - the preprocessing steps (channel order, normalisation)
  - a spec overriding the model's metadata
  - shape mismatch errors
- **Parity:** run the **original** BCD model with the "ImageNet standard" spec. The metrics must be identical to the old evaluator. OCS stays wrapped and its result must not change.

### 5.5 Docs

- **`model-submission.md`:** add the input spec section. Say clearly which preprocessing the platform does and which must be inside the model.
- **Model owners:** document the decode rules (especially DICOM), so they train and score on the same input.

### 5.6 Order

1. Processing: spec checks + `infer.py` steps + tests + BCD parity.
2. Buffer: new slot + tests.
3. UI: form section + upload.
4. Release with the steps in 3.6. Buffer and Processing must ship together.

---

## 6. Instructions for model providers

This section is the text to give model owners, in the UI or the provider docs. It covers all three formats, both **without Option C** (today) and **with Option C**.

### 6.1 Quick summary

| Format | Without Option C (today) | With Option C |
|---|---|---|
| **ONNX** | Build your preprocessing into the model; set the `tanuh.resize` metadata key. Upload `.onnx` (+ `.onnx.data`) and `adaptor.py`. | Upload your **original** `.onnx` (+ `.onnx.data`) and `adaptor.py`. Pick your preprocessing on the form. |
| **TorchScript** | Build your preprocessing into the model; embed `tanuh.json`. Upload `.pt` and `adaptor.py`. | Upload your **original** `.pt` and `adaptor.py`. Enter the input size and pick your preprocessing on the form. |
| **Hugging Face** | Upload a `.zip` of `save_pretrained()` and `adaptor.py`. Preprocessing comes from `preprocessor_config.json`. | **Same as without Option C.** |

In both cases, preprocessing that isn't a standard step (see 6.3) must be inside the model.

### 6.2 What the platform gives your model (all formats, both cases)

The platform prepares every image the same way before your model sees it:

1. **Decode:**
   - `.jpg`/`.jpeg`/`.png` → RGB.
   - `.dcm` (DICOM) → middle frame, MONOCHROME1 inverted, pixel values min-max scaled to 0–255, grey copied into 3 channels (R = G = B).
2. **Grey models:** if your model has 1 input channel, the RGB image is converted to grey (OpenCV `RGB2GRAY`).
3. **Resize** to your model's height × width: bilinear (Pillow) or bicubic (OpenCV `INTER_CUBIC`).
4. **Number range:** float inputs get values **0–1** (pixel ÷ 255). `uint8` inputs get 0–255.
5. **Layout:** `[N, C, H, W]` (NCHW) or `[N, H, W, C]` (NHWC), as your model declares.
6. **Batching:** images are sent in batches. A dynamic batch size is best; a fixed batch size is padded.

**Train on data prepared the same way,** especially DICOM. For example, if you trained on DICOMs with a different window or scaling, your score will be lower.

### 6.3 Step 1 for every format: write down your preprocessing

Look at the code that prepared images during **training**. Find each step in this table:

| Your training code does | What it is | Without Option C | With Option C |
|---|---|---|---|
| `Resize(...)` (torchvision/PIL), bilinear | resize method | `resize: bilinear` in metadata | form: resize = bilinear |
| `cv2.resize(..., INTER_CUBIC)` | resize method | `resize: bicubic` in metadata | form: resize = bicubic |
| `ToTensor()`, or `/ 255` | scale to 0–1 | nothing: the platform does it | nothing: the platform does it |
| `Normalize(mean, std)` | normalisation | **inside the model** | form: normalisation = ImageNet or custom |
| `cv2.imread(...)` without converting to RGB | BGR colour order | **inside the model** | form: colour order = BGR |
| Anything else: crop after resize, CLAHE, colour balancing, per-image standardisation, windowing | custom | **inside the model** | **inside the model** (leave the form at defaults) |

**Resize, then centre-crop** (e.g. `Resize(256)` + `CenterCrop(224)`) is not a standard step. Either:
- build the crop into the model and declare the larger size (256) as its input, or
- retrain with a direct resize.

### 6.4 ONNX

**Without Option C (today)**

1. Add your preprocessing as the first layers of your model, then export. Your model receives **RGB, 0–1, `[N, 3, H, W]`**. Example for ImageNet normalisation:

   ```python
   import torch, onnx

   class WithPreprocessing(torch.nn.Module):
       def __init__(self, model):
           super().__init__()
           self.model = model
           self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
           self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

       def forward(self, x):                      # x: RGB, 0–1, [N, 3, H, W]
           # x = x.flip(1)                        # add this line if you trained on BGR
           return self.model((x - self.mean) / self.std)

   wrapped = WithPreprocessing(model).eval()
   torch.onnx.export(wrapped, torch.zeros(1, 3, 224, 224), "model.onnx",
                     input_names=["input"], output_names=["logits"],
                     dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
   ```

2. Set the resize method (skip it if bilinear):

   ```python
   m = onnx.load("model.onnx")
   p = m.metadata_props.add(); p.key, p.value = "tanuh.resize", "bicubic"
   onnx.save(m, "model.onnx")
   ```

3. **Only if the model is larger than 2 GB,** save the weights to one external file, and upload it in the weights slot:

   ```python
   onnx.save(m, "model.onnx", save_as_external_data=True,
             all_tensors_to_one_file=True, location="model.onnx.data")
   ```

4. **Already have an ONNX file without its preprocessing, and it only needs ImageNet normalisation?** Use `python3 tools/reference/wrap_onnx.py --mode imagenet --src model.onnx --dst model_tanuh.onnx`. For anything else, rebuild from training code (step 1) or contact the platform team.
5. Upload `model.onnx` (+ `model.onnx.data`) and `adaptor.py`.

**With Option C**

1. Export your model **without** preprocessing. You can upload an existing `.onnx` as it is.

   ```python
   torch.onnx.export(model.eval(), torch.zeros(1, 3, 224, 224), "model.onnx",
                     input_names=["input"], output_names=["logits"],
                     dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
   ```

2. On the submit form, choose:
   - **resize:** bilinear or bicubic
   - **colour order:** RGB or BGR
   - **normalisation:** none, ImageNet, or custom mean/std

   `tanuh.resize` metadata is not needed; the form's value is used.
3. Upload `model.onnx` (+ `model.onnx.data`) and `adaptor.py`.

**ONNX rules (both cases):**
- Height and width must be fixed numbers. A dynamic batch is fine; dynamic H/W is rejected.
- The input type must be float32, float16 or uint8.

### 6.5 TorchScript

**Without Option C (today)**

1. Wrap the model with its preprocessing (same `WithPreprocessing` class as 6.4) and save it with `torch.jit.save`, embedding `tanuh.json`:

   ```python
   import json, torch

   scripted = torch.jit.script(WithPreprocessing(model).eval())
   # or: torch.jit.trace(WithPreprocessing(model).eval(), torch.zeros(1, 3, 224, 224))
   torch.jit.save(scripted, "model.pt", _extra_files={"tanuh.json": json.dumps({
       "input_shape": [3, 224, 224],     # [C, H, W]
       "layout": "NCHW",
       "dtype": "float32",
       "resize": "bilinear"})})          # or "bicubic"
   ```

2. **Already have a TorchScript `.pt`?** Use one of the wrap tools:

   | Your model | Command |
   |---|---|
   | needs only ImageNet normalisation | `python3 tools/reference/wrap_torchscript.py --mode imagenet --src model.pt --dst model_tanuh.pt --input-shape 3 224 224` |
   | already has its preprocessing inside | `wrap_torchscript.py --mode none ... --input-shape 3 224 224 --resize bilinear` (only adds `tanuh.json`) |

3. Upload `model.pt` and `adaptor.py`.

**With Option C**

1. Save the plain model. You can upload an existing TorchScript `.pt` as it is.

   ```python
   torch.jit.save(torch.jit.script(model.eval()), "model.pt")
   ```

2. On the submit form:
   - **Enter:** the input size (`C`, `H`, `W`) and layout (NCHW or NHWC).
   - **Choose:** resize, colour order and normalisation.
3. Upload `model.pt` and `adaptor.py`.

**TorchScript rules (both cases):**
- The file must be saved with `torch.jit.save`. `torch.save` files (state dicts, checkpoints, `.pth`) are rejected, because they need your model's Python code to load.
- The platform uses **PyTorch 2.5.1**. Save with 2.5.1 or older where possible, and confirm with `check_model.py` (6.8).
- The model output can be a tensor, a tuple of tensors, or a dict of tensors.

### 6.6 Hugging Face (same with and without Option C)

1. Save **both** the model and its image processor into one folder, then zip it:

   ```python
   model.save_pretrained("my-model", safe_serialization=True)   # config.json + model.safetensors
   processor.save_pretrained("my-model")                         # preprocessor_config.json
   ```

2. **Check `preprocessor_config.json`** matches your training preprocessing. This file **is** your preprocessing. A typical example:

   ```json
   {"image_processor_type": "ViTImageProcessor",
    "do_resize": true, "size": {"height": 224, "width": 224}, "resample": 2,
    "do_rescale": true, "rescale_factor": 0.00392156862745098,
    "do_normalize": true, "image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225]}
   ```

   - `resample`: `2` = bilinear, `3` = bicubic.
   - The processor receives RGB images, 0–255, straight from decoding (6.2 step 1); the platform does no resizing or scaling for this format.
3. The zip must contain:
   - **Required:** `config.json`, `*.safetensors` and `preprocessor_config.json`.
   - **Not allowed:** `.py`, `.bin`, `.pt`, `.pth`, `.pkl`, `.ckpt` or `.h5` files; `auto_map` in `config.json`. So don't zip a Trainer `checkpoint-XXXX/` folder.
4. The architecture must be built into `transformers` 4.46.3 for image classification, such as ViT, ConvNeXT, ResNet, Swin, EfficientNet or MobileViTV2. Custom model code is not run.
5. **Custom preprocessing** (colour balancing, CLAHE, …) can't be written in `preprocessor_config.json`. Use ONNX or TorchScript instead.
6. Upload `model.zip` and `adaptor.py`. There is no preprocessing form for this format.

Full details: [model-submission.md](model-submission.md).

### 6.7 `adaptor.py` (all formats, both cases)

1. **Input:** `raw/raw_outputs.npz` holds `ids` (file names) and one array per model output. The output names depend on the format:

   | Format | Output names |
   |---|---|
   | ONNX | your ONNX output names, e.g. `logits` |
   | TorchScript | `output` for one tensor, `output_0`, `output_1`, … for a tuple, or the keys of a dict |
   | Hugging Face | `logits` |

2. **Also available:** `dataset_spec.json` gives `task_type`, `class_names` and `num_classes`. The **label number = position in `class_names`**, so make sure your model's output columns follow the same order.
3. **Output:** write `predictions/predictions.csv` with one row per file, every file exactly once:

   | Problem type | Columns |
   |---|---|
   | `binary_classification` | `file,label,score`, where `score` is the probability of class 1 |
   | `multiclass_classification` | `file,label,prob_0,…,prob_{C-1}`, where the probabilities sum to 1 |

4. **Start from a template:**
   - `tools/reference/adaptor_sigmoid_binary.py`: one logit → sigmoid.
   - `tools/reference/adaptor_softmax_multiclass.py`: C logits → softmax.

   Most models can use one of these unchanged.
5. **Environment:** Python 3 with numpy, pandas, scipy, scikit-learn, torch and OpenCV already installed. Other imports are installed automatically from PyPI.
6. **Time limit:** 10 minutes.

### 6.8 Before you upload: check locally

Install the platform's versions (numpy<2, onnxruntime 1.19.2, torch 2.5.1, transformers 4.46.3), then run on one of **your own** sample images:

```bash
python3 tools/check_model.py --format onnx        --model model.onnx [--weights model.onnx.data] --image sample.jpg
python3 tools/check_model.py --format torchscript --model model.pt   --image sample.jpg
python3 tools/check_model.py --format huggingface --model model.zip  --image sample.jpg
# with Option C, add:  --input-spec input_spec.json   (to be added, see 5.1)
```

Compare the printed outputs with what your own training code gives for the same image. **If they differ, your preprocessing is wrong or missing.** This is the only way to catch the silent-score-drop problem before submitting.

### 6.9 Common mistakes

| Mistake | Result | Fix |
|---|---|---|
| Normalisation left out | score silently lower (OCS: 0.99 → 0.76) | put it inside the model, or choose it on the form (Option C) |
| Normalising twice (inside the model **and** on the form) | score silently lower | use one place only |
| Dividing by 255 inside the model | inputs become ~0, score silently lower | don't: the platform already gives 0–1 |
| Model trained on 0–255 floats | score silently lower | multiply by 255 as the model's first step |
| Trained with `cv2.imread` (BGR), declared RGB | score silently lower | flip channels inside the model, or choose BGR (Option C) |
| ONNX with dynamic height/width | job fails (error code 2) | export with a fixed size |
| `torch.save` instead of `torch.jit.save` | job fails (error code 2) | save with `torch.jit.save` |
| Zipped a Hugging Face `checkpoint-XXXX/` folder | job fails (error code 2) | zip the `save_pretrained()` output |
| Adaptor class order differs from `class_names` | score silently lower | reorder columns in the adaptor |
| Adaptor misses or repeats a file | job fails (error code 1) | write every file exactly once |

---

## 7. Open questions

1. **Catalogue:** can the catalogue owner confirm that top-level `task_type` and `class_names` are accepted?
2. **Evaluators bucket:** which service account gets read access? Is the name `tanuh-evaluators` fine?
3. **Option C:** go ahead? If yes, should it be built before the first release, or right after it?
4. **Next datasets:** which datasets come next? A new file type (for example CT, MRI or pathology slides) needs a new decoder. A new problem type (for example segmentation or detection) needs a new bucket: an evaluator, a predictions format and a reference adaptor. Some cases, like CT windowing, also need a dataset-level decode setting in the catalogue.
5. **Sandbox:** when should the deferred sandbox for the adaptor be scheduled?
