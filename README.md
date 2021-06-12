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
To appear soon.
