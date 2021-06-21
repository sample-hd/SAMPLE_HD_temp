# SAMPLE-HD

This is a repostiory containing a code for *SAMPLE-HD: Simultaneous Action and Motion Planning Learning Environment* submitted to NeurIPS 2021 conference.

Please treat the code as the experimental version, in case of anything being unclear in use, please raise issue.

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
Please download dataset from the following:
- [Scenes, segmentations, depth maps, instructions](https://imperialcollegelondon.box.com/s/5ia1o1bewwikvvmf6hrlwkrxohlm0cfg)
- Sequences (images + data):
	- [0-749](https://imperialcollegelondon.box.com/s/vete1bwpxkbvtjj1x2hrqbh0u38px4xd)
	- [750-1499](https://imperialcollegelondon.box.com/s/ocf1o1xukyqgqsprssyub5d0zn5vn74v)
	- [1500-2249](https://imperialcollegelondon.box.com/s/fa8pw88npsc135xzvkodt7tk45wfctyu)
	- [2250-2999](https://imperialcollegelondon.box.com/s/ft02zahp08fhax1f95as2bru3lyef2r5)
	- [3000-3749](https://imperialcollegelondon.box.com/s/zygcp2je6n8ztf7ja5qj08totbdf27qr)
	- [3750-4499](https://imperialcollegelondon.box.com/s/ymgze1apq66d1mvtpzym849nxz3hnfiz)
	- [4500-5249](https://imperialcollegelondon.box.com/s/2an51d2krkz0l9myj506zd1cdft0e0ji)
	- [5250-5999](https://imperialcollegelondon.box.com/s/v13rjh1x072fd09wofvmg7a0oc2faegt)
	- [6000-6749](https://imperialcollegelondon.box.com/s/ftd40jvpe93puiibaec8vgp687rwt6vv)
	- [6750-7499](https://imperialcollegelondon.box.com/s/wia667xxs7kyhfsn7zuercxosn481xy8)
	- [7500-8249](https://imperialcollegelondon.box.com/s/8q1xzgdtnhzahoxyp6ui18kxr3gc0gpi)
	- [8250-8359](https://imperialcollegelondon.box.com/s/xcdfc8xhczphy0a9dksmhe50aehkdqbf)
- Sequences (depth):
	- [0-999](https://imperialcollegelondon.box.com/s/qhl77xu1bqfzmj9pll50l6zswgn5p189)
	- [1000-1999](https://imperialcollegelondon.box.com/s/ohjjyzg640gys4ef18atvc1iq5a7g66o)
	- [2000-2999](https://imperialcollegelondon.box.com/s/ge0kxl7hhkjjych77clp562bjq6vmq8a)
	- [3000-3999](https://imperialcollegelondon.box.com/s/ip39bnenm76jfdohivsj6lyvetdnxus3)
	- [4000-4999](https://imperialcollegelondon.box.com/s/petaykvx5joh5olnfpkm45co6ozmhjw5)
	- [5000-5999](https://imperialcollegelondon.box.com/s/q3pk3btjt0lk9d4sb8t6hf2ghcjn9011)
	- [6000-6999](https://imperialcollegelondon.box.com/s/8tzv8ch3k2gjfyd3l41oc5zfz58xn59m)
	- [7000-7999](https://imperialcollegelondon.box.com/s/30mwe1s9l6t9gt9b7wqa4li8e2yk8w5d)
	- [8000-8359](https://imperialcollegelondon.box.com/s/we4sirabo3tta2477y1joofj77wd8jft)
- Sequences (segmentation):
	- [0-2999](https://imperialcollegelondon.box.com/s/fijh15e1fahjp482a5gy9w31408t7sz1)
	- [3000-5999](https://imperialcollegelondon.box.com/s/jp43t5xgukw6frzw2pbsq74b6ddy98zs)
	- [6000-8359](https://imperialcollegelondon.box.com/s/g2vlysvag0zimaok16jncndc4ufjeir6)

Note that the form of dataset will be altered to reduce the number of preprocessing steps for various methods.
