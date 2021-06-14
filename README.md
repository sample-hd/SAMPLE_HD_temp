# SAMPLE-HD

This is a repostiory containing a code for *SAMPLE-HD: Simultaneous Action and Motion Planning Learning Environment* submitted to NeurIPS 2021 conference.

Please treat the code as the experimental version.

## Unity Environment
 The environment was prepared in the following setup:
 * Unity 2020.2.4f1
 * High Definition Rendering Pipeline 10.3.1
 * ML Agents 1.8.1
 
The package containing all assets can be found [here](https://imperialcollegelondon.box.com/s/vviei2fxij2rrgkh15fcuf2gewjflbs5).
Various functionalities are split across the scenes (located in Assets):
* scene_generation - use to generate new scenes - see options in *MultiSceneCreator*, run from editor;
* grasp_registration - used to register new grasps - use GameObject -> GraspWizard in the toolbar;
* instr_loader - use to generate ground truth paths - see options in *Action* and *MultiInstructionLoader*, run from editor;
* inference_clean- use for inference - see options in *Exchange*, it is suggested to compile the scene after setting up all options, plese don't use *IL2CPP* as a scripting backend.

## Python code
We have trained our models in 3 steps:
* Instruction to program Seq2seq model
* Mask R-CNN for segmentation
* Attributes extraction

### Scene segmentation
For the scene segmentation we use [Detectron2](https://github.com/facebookresearch/detectron2) implementation of Mask R-CNN. Please follow the installation as suggested in the link. Training is run from `tools` by running `train_mask.py`.

### Seq2seq
 Training is run from `tools` by running `train.py` and passing correct config as `-cfg` argument. A config for seq2seq model is available in `/tools/config/train_seq1.yaml`.
 
###  Attributes extraction
 Training is run from `tools` by running `train.py` and passing correct config as `-cfg` argument. A config for seq2seq model is available in `/tools/config/train_ann1.yaml`. In `tools/inference_annotation.py` a setup for the inference of annotations is shown.

### Executor
In `tools/exec.py` a setup for executor is presented with the result showing correct instructions numbers.

### Inference with Unity
For the inference with Unity, compile the program under your system and provide its path in scripts. We use `tools/inference_batch_nomod_save.py` and `tools/inference_batch_nomod_ann_save.py` for the inference with ground truth and predicted annotations.

## Dataset
Please download dataset from:

Note that the form of dataset will be altered to reduce the number of preprocessing steps for various methods.
