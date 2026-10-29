> **Note:** This is the original research plan (v4), converted from `.docx`. It is out of date: the design was
> tightened afterwards. See [Changes relative to research plan v4](../README.md#changes-relative-to-research-plan-v4)
> in the README and FINDINGS.md for the differences.

# Actor-Agnostic Persistent Object Memory for Hidden-Object Localization

# Authors

Haonan Yin

Joris Köster

# Abstract

Vision-language-action systems often rely on current observations or short-term visual history. This limits their ability to operate in changing environments over longer time horizons. A specific problem is that targets may become hidden between episodes. This project will investigate whether an observation-driven memory, which keeps track of objects and the containers they become hidden in, can localize such hidden targets across episodes. In controlled simulated tabletop scenes, we will compare current-only and keyframe baselines with persistent 2D and VGGT-grounded 3D containment memories while the camera viewpoint or the position of a container changes between episodes, using visually distinct or identical containers. All conditions receive the same RGB observations and the same query model, and no simulator information is used as input. The study aims to determine when explicit persistent memory and 3D spatial grounding provide an advantage over simpler keyframe-based and 2D methods. First-choice container accuracy, paired success-rate differences and 95% confidence intervals will be reported. The study evaluates the memory component as a foundation for future robotic retrieval.

# Introduction

## Background and Rationale

Vision-language-action (VLA) policies map visual observations and language instructions to robot actions, promising general-purpose controllers \[1\]. Many of these policies condition on the current observation or a short history; SpatialVLA, for instance, predicts actions from a single current view \[3\], and memory for long-horizon reasoning is identified as an open challenge for VLAs \[1\]. This limits their capability in situations of partial observability. Consider a situation where an apple was hidden under a bowl during one episode and the apple needs to be put on a plate during a following episode. An object which has been visible during one episode may become hidden throughout another episode. We argue that a memory to tackle this should be built from observations rather than privileged access to the robot’s own actions, as such a task may also be performed by a person or another agent.

## Theory and Related Work

Recent work has introduced different methods of providing VLAs with more detailed spatial representations of the environment. VGGT estimates camera parameters, depth maps and 3D point maps from one or more RGB views in a single forward pass, expressed in the coordinate frame of the first camera and up to an unknown scale \[2\]. SpatialVLA injects 3D position encodings derived from estimated depth into a VLA, but conditions on a single current view and reports weaknesses on long-horizon tasks \[3\]. 3D-Mix injects frozen VGGT features into VLA architectures through a gated fusion module and reports improved task success on the SIMPLER and LIBERO benchmarks \[4\]. These methods provide the VLA with a per-frame spatial representation of the environment, but do not by themselves establish how hidden targets can be localized across episodes.

VGGT has also already been used to construct spatial memories for VLA models. SOMA builds a persistent object-centric spatial memory from head-camera scans, lifting detections to 3D boxes with VGGT, and refines it during manipulation \[5\]. It addresses objects outside the camera’s field of view rather than objects hidden inside containers, does not maintain memory across episodes, and names dynamic scenes and container state changes as limitations. FOUND-IT uses VGGT-SLAM 2.0 to build a task-driven hierarchical 3D scene graph from monocular video, maintaining a keyframe-indexed visual memory of a static environment \[6\]. SceneVGGT combines VGGT pose estimates with sensor depth to build temporally coherent semantic maps with persistent object identities, and flags objects as removed when they are expected but not observed \[7\]. These systems show that VGGT can serve as a source of geometric estimates for spatial scene memory, but none of them models containment.

Furthermore, several systems show that persistent object memory can be built without utilizing VGGT. Embodied VideoAgent stores object identities, states, 3D bounding boxes and support and containment relations using egocentric video combined with depth and camera poses, and uses the containment relation to answer which container holds an object \[8\]. It is the closest prior work to ours, but it is evaluated on question answering against video-language baselines rather than in matched comparisons with frame-based inputs, and it does not vary camera or container motion. GESTO, developed partly at KTH, constructs a persistent scene graph and hierarchical memory of observed human–object interactions from RGB-D video streams but evaluates question answering based on the memory rather than robot manipulation \[9\].

Combining these studies, we find that most of the individual components needed to achieve the proposed idea are covered. We will investigate a narrow problem: an observation-driven, action-independent memory built from monocular RGB inputs to keep track of observed containment events, which persists across task episodes. The system should be able to locate objects contained in their visible parent containers. In this project VGGT enables the construction of a 3D scene representation for such a memory, while the research contribution is a controlled study of actor-agnostic, persistent containment memory in which all conditions receive the same observations and differ only in how they represent them. Here, actor-agnostic refers to the absence of actor-specific action logs, rather than demonstrated generalization to arbitrary actors or events.

## Research Problem, Aim, and Research Questions

The research problem is to evaluate a memory system which maintains an actionable belief state about occluded objects within the scene. Without reliable spatial association, a belief state can become insufficient for localization over time if, for example, the container moves. Current observations can update the position of a visible container but do not by themselves identify which specific container holds a hidden target. Even so, the memory cannot reliably detect a manipulation that happens outside the observed area, such as the removal of an apple from a covered container. Such unobserved changes to the contents of a container remain outside the scope of the study, whereas unobserved motion of a container is studied explicitly, assuming that a hidden object moves with its container.

We aim to understand whether a policy-independent, containment-aware memory built from monocular observations can provide a foundation for hidden-object retrieval across multiple episode rollouts. We further want to test whether VGGT-based spatial grounding adds value beyond keyframes and beyond a 2D memory that compensates for viewpoint changes in the image plane when the camera or containers move. The present evaluation will focus on identifying the correct container, rather than physical retrieval or VLA control. To keep the findings transferable to real robots, no condition will receive simulator information or prior knowledge about the scene, such as object sizes, as input.

RQ1: Does an observation-driven containment memory improve the identification of correct containers holding hidden objects across task episodes compared to systems that only utilize current or historical RGB frames?

RQ2: Does VGGT-based geometric grounding improve the identification of correct containers holding hidden objects across task episodes compared to a matched 2D memory with image-plane viewpoint compensation when the camera viewpoint or the position of a container changes between episodes?

We hypothesize that explicit containment memory will improve container identification over frame-based baselines, with the largest gain when the containers are visually identical and the camera or a container has moved between episodes (H1). Furthermore, we expect the benefit of VGGT grounding over the plane-registered 2D memory to increase with the magnitude of the viewpoint change and to be small when the camera is fixed (H2). The project as stated will not claim a new architecture but evaluate the combination of the technologies above in defined cross-episode containment settings.

Our objectives are to establish a reproducible simulation setup, implement the RGB perception pipeline and non-geometric memories, complete the geometry pilot and fix the evaluation protocol, and carry out the matched evaluation and analysis. The results will then be prepared for the final report.

# Research Method

## Research Design and Methodological Rationale

This study is a quantitative evaluation over pairs of consecutive task episodes in a shared tabletop environment. During the first episode, a visible target object will be placed by a scripted external actor or a scripted robot controller and then covered by one of three movable covers, such as inverted bowls or cups, selected at random. Between the episodes, the camera viewpoint or the position of one cover may change while the scene is not observed, and short camera sweeps are recorded before and after this change. Only RGB observations and a task-boundary signal are provided to the memory module. In the second episode, the memory persists and the system is tasked to identify the cover hiding the target.

We are choosing to start in simulation environments as they allow us to control camera motion, occlusion, object identities and external interventions in a repeatable manner while being able to access ground truth for evaluation. Unobserved changes between episodes reflect the realistic cross-episode case, in which a scene is rearranged while the robot is inactive or busy elsewhere, and require the memory to reidentify containers rather than track them continuously. The main experiment will compare four conditions that receive the same RGB recordings, candidate containers and query model, and differ only in how earlier observations are represented:

1\. Current-only: The query model can only access the final RGB observation during container identification. This condition provides a lower bound and checks whether the scenes contain cues that allow guessing above chance.

2\. Keyframe memory: Up to five earlier RGB frames are provided without any explicit object graph. They are selected by the shared perception pipeline at detected placement and covering events and from the sweep after the change, without using simulator information.

3\. 2D containment memory: Persistent object identities and containment relations are maintained in the image plane. After a viewpoint change, earlier container positions are mapped into the new view by a tabletop-plane homography that is estimated automatically from feature matches on the static background.

4\. VGGT containment memory: The same records and update rules are used, but containers are localized in 3D with VGGT. Reconstructions from different sweeps are aligned by a similarity transform estimated from the same static-background matches, so that neither a metric scale nor prior knowledge of object sizes is required.

Simulation-provided geometry and perception may be used as diagnostics to separate reconstruction and perception failures from memory failures. These oracle diagnostics will be reported separately and will neither form an experimental condition nor serve as evidence for the research questions. The memory built will be independent of the downstream VLA. Objects will be stored with persistent identifiers, appearance features, a last supporting observation and an association confidence score. When hidden under a cover, the object should receive a parent relation to that container. When a visible container moves or is reidentified, the memory updates the parent location while preserving the child belief.

Using a controlled simulated setup allows us to directly compare the performance of the proposed memory conditions in a repeatable manner. As an alternative to evaluating system performance in a simulated environment, prerecorded videos of manipulating objects and hiding them in boxes could be used as well, but this would require additional annotation to provide ground-truth data. The same limitation applies to real robot rollouts, which further introduce a risk of slight scene differences between rollouts, creating an extra source of error. As the study is conducted under quite strict time constraints, it may be necessary to reduce the number of different memory systems to compare. In such a case, the keyframe baseline should be compared to the 2D memory system, with the scope revised with the mentor. RQ2 will only be answered if both the 2D and VGGT systems are evaluated.

For the experiment to work, we assume that we can extend an existing simulation environment in a way that allows us to run the experiment reliably. A short pilot will therefore verify that the covers remain physically stable when the target is hidden and a cover is moved, that visually distinct cover variants and textured backgrounds can be added, and which renderer can be used on the available hardware. It is further important that the VGGT scene representations and the static-background registration are reliable enough to support spatial grounding. These assumptions will be checked in pilot experiments before the final protocol is fixed.

## Data, Materials, Participants, or Other Sources

The simulation environment will be ManiSkill3, starting from the shell-game tasks of the MIKASA-Robo memory benchmark, in which three identical inverted mugs cover a small target object on a tabletop. ManiSkill3 allows object poses to be set directly between episodes, provides freely configurable cameras with ground-truth depth and segmentation for scoring, runs headless on GPU clusters and offers an optional ray-traced renderer. We will extend these tasks with cover motion between episodes, visually distinct cover variants, distractor objects and camera sweeps. If these extensions cannot be realized, robosuite with RoboCasa object assets will be used as a fallback. Since registration on the static background requires visual texture, the table and background will be given non-repetitive textures, and the scene will be lit with directional light that casts shadows. Two or three tabletop scenarios will be constructed, each with a target object, three covers and one or two distractor objects. Within each scenario, the covers will either be visually distinct or identical. Neutral covers that do not suggest their content will be used, and the target cover will be assigned at random with balanced assignments. Placement will be observable, and the target will be fully hidden at query time.

The RGB perception pipeline will use pretrained open-vocabulary detection or prompt-based segmentation combined with tracking over time. At every episode boundary, a short camera sweep of about ten frames along an arc around the current camera pose will be recorded with all objects stationary; the same sweep frames are available to all conditions. All conditions choose from the same candidate covers, which are detected in the final image and marked with numeric labels, using the same query model, prompt and selection instructions with deterministic decoding. The current-only baseline will receive the final image; the keyframe baseline will additionally receive up to five event-triggered frames. This provisional limit has not yet been validated. Pilot experiments will check whether these frames adequately capture placement and subsequent motion, and the limit will be adjusted if necessary before the final evaluation. The explicit-memory conditions will receive the final image and their stored object relations as text referring to the numbered candidates. The frame-selection rule and query settings will be fixed before testing.

VGGT will reconstruct the static background and visible containers from each sweep in a separate forward pass. Covers, other movable objects and the robot arm will be masked using the shared detector output, and reconstructions from different sweeps will be aligned by a similarity transform estimated with RANSAC from feature matches on the static background only. The 2D memory uses the same masks and matches to estimate its tabletop-plane homography. To consistently associate the containers, appearance similarity, semantic class and position after registration will be evaluated between updates, with thresholds expressed relative to the observed scene, for example as a fraction of the distance between containers. If too few static matches are found, the affected relations are marked as unknown and logged as registration failures. Containment of a target object will be detected based on its observed placement, followed by its disappearance when a cover is placed over it.

The motion conditions will keep both the camera and the covers fixed, change only the camera viewpoint, or move only one cover between episodes. In the camera condition, the magnitude of the viewpoint change will be drawn at random for each trial within a fixed range, provisionally 10° to 60°. In the cover condition, one cover will be displaced by a random, bounded distance; covers will not swap places or cross paths, so that every trial remains solvable in principle. All magnitudes will be logged. Crossed with the two cover appearances, this gives six cells with 30 matched trials each, distributed across the two or three tabletop scenarios with different placement and motion seeds. This gives 180 test recordings and 720 evaluations across the four conditions. The trial count will be fixed before testing begins. Pilot recordings will be kept separate from the final test set. No human participants or personal data will be used. We expect to use existing KTH computing resources and pretrained models without purchasing additional hardware; relevant software and model licences will be checked before use.

## Work Procedure and Data Collection

As a first step we will create a minimal setup with two successive task episodes. The first will hide the target object; the second will query its parent container. The first episode can therefore be fully scripted, which ensures that query success depends on identifying the correct container in the second episode. In this step, simulator-provided perception will be used for development only. Second, RGB object detection, tracking and relevant frame logging will replace this oracle perception. Third, the 2D graph with plane registration will be implemented and verified. Lastly, VGGT reconstruction and cross-episode association will be implemented using the camera sweeps. Simulator identities, poses, depth and containment labels will only be used for development diagnostics and scoring, never as inputs to any condition during testing.

For each object being observed, the system will store a record containing its ID, class, appearance embedding, parent, relation, location, visibility, timestamp and association confidence score. The location is stored in the image plane in the 2D memory and in 3D in the VGGT memory; all other fields and update rules are shared. Targets inferred to be hidden will be marked as hidden in memory and associated with the container. Confidence is dependent on the expected location and agreement of evidence. The containment relation will be retained while the parent container can be reliably reidentified. If the parent cannot be reidentified or the relation becomes ambiguous, the state of the object will become unknown.

All decisions of the memory system are going to be logged and compared to ground truth. The task boundary will reset the temporary query context while the scene state and memory remain persistent across both episodes. Both the scene and memory will be reset before each new independent trial.

## Data Analysis and Quality Assurance

RQ1 will compare the first-choice container accuracy of the keyframe baseline with each of the two explicit containment memories. These two primary comparisons will be corrected for multiple testing with the Holm method, and results will be reported both pooled and separately for distinct and identical covers. The current-only condition will be compared with the chance level of one in three to check for scene cues. Unknown responses and processing failures will count as unsuccessful trials and will also be reported separately. Paired differences and 95% confidence intervals will be calculated by resampling trial seeds within each scene. If ranked outputs are available, the rank of the correct container will give an indication of query efficiency. As a diagnostic, the container given directly by the memory records will be compared with the choice of the query model to separate memory errors from errors in how the query model uses the memory.

RQ2 will compare the 2D and VGGT-grounded memories in each motion condition. We will measure container reidentification accuracy and first-choice accuracy; 3D container localization error will be reported as a diagnostic for the VGGT memory only. H2 will be examined with a logistic regression of trial success on memory type, the magnitude of the viewpoint change and their interaction, using the trials of the camera condition together with the fixed-camera trials. For the 3D error, a scene-level similarity alignment to simulation coordinates will be used for scoring only, based on static background points rather than separate fits to containers.

The results will be interpreted as evidence about the controlled class of containment problems. We hope to understand in what specific conditions or situations monocular 3D grounding provides more reliable memory than simpler 2D approaches. Claims will be limited to the evaluated scenes and observed placement events. In particular, association and registration thresholds will be fixed in simulation pilots, and their transfer to real cameras remains untested.

All conditions will be evaluated on the same recordings. Randomness during the trials will be controlled by using and documenting seeds for all runs to support reproducibility. All runs will be extensively logged and compared against simulation ground truth. A potential source of error is variability in the outputs of the AI models used; the query model will therefore use deterministic decoding, and repeatability will be tested for each model on a fixed subset of pilot inputs. Model versions, query prompts, generation settings and hardware will be recorded. Pilot tests will check relation persistence, state resets and container association. Pilot and test data will remain separate, and test results will not be used to tune settings. Input files will be checked for ground-truth leakage. Failures will be reviewed jointly and classified as perception, containment inference, registration, container association or geometric reconstruction errors.

## Use of AI Tools

We are planning to use AI tools to assist us in coding tasks for the project as well as providing structure and feedback for our writing. They may further be used to identify and pre-select suitable papers. Scientific claims and bibliographic details will be checked against the original publications. Suggested code and analysis procedures will be reviewed and tested before use. Both authors will remain responsible for the research design, interpretation of results and intellectual content of the submitted work.

# Project Planning and Feasibility

## Work Tasks and Milestones

1.  We plan to carry out the project in sequential milestones for implementation and evaluation. First, the ManiSkill3 shell-game environment will be extended to a minimal setup that hides an object, moves the camera or a cover between two episodes, and records a parent–child relation between the object and its container. This should provide a basic setup to test our memory systems’ capabilities to find the correct container.

2.  As a next step the perception pipeline will be added, which will acquire an RGB stream of the scene, track objects in the scene and log relevant frames.

3.  Based on this system the current-only and keyframe baselines will be implemented. The current-only system is a VLM able to select from all identified containers currently visible. The keyframe memory will instead provide the same VLM with a series of frames depicting the process of objects being hidden and subsequent motion. Query settings will be consistent with Section 2.2.

4.  A persistent 2D memory system with automatic tabletop-plane registration will be built. The system actively aims to keep track of hidden objects and parent relations between objects and containers.

5.  A persistent spatially grounded 3D memory system will be built, with the same aim as the 2D system but potentially higher accuracy. The records, masks and relation-update rules will remain the same.

6.  Once all required conditions are functional, pilot experiments will be used to verify the evaluation pipeline and logging. The final settings, trial count and analysis procedure will then be fixed.

7.  For the final experiments more recordings with random object and target placements as well as varied scene configurations will be created to assess robustness across the tested settings, following the trial allocation in Section 2.2.

8.  After conducting the final experiment all results will be analysed and prepared for the final report.

9.  The final report will be drafted.

10. After revision and feedback we will finalize the report.

## Allocation of Responsibilities

As this project is done collaboratively, both of us will equally contribute to the writing as well as the experiment setups and evaluation. We may split tasks, especially in the developing process between us effectively so some of the milestones may be reached in parallel. A sensible split would be between the simulation environment and the memory system for the beginning. In later stages, once the simulation environment is created, both will contribute to the memory systems. Reporting and documentation should be done equally at all times throughout the project.

## Implementation Risks and Mitigation Strategies

Technical challenges may include insufficient quality of VGGT’s estimates of the 3D environment or of the static-background registration. In such a case, simulation-provided geometry will be used as a separate diagnostic to distinguish failures caused by spatial reconstruction from failures of the memory system. Reconstruction and alignment will first be checked in pilot experiments. If VGGT cannot be evaluated reliably, RQ2 will remain unanswered and the scope will be revised with the mentor; oracle results will not substitute for VGGT evidence.

For running the experiments we expect to have continued access to KTH’s GPU cluster. If we do not have access, we may need to reduce the number of trials during evaluation and cache model outputs where possible. Any reduction will be decided before testing and documented. The strength of our findings will depend on the number of different scenes and trials we are able to evaluate. All findings will be reported according to course requirements, with narrower claims restricted to the actually evaluated settings.

A third risk is that perception or container association errors may prevent the memory systems from retaining correct containment relations. We will start with clearly observable placement events, test the pipeline on pilot recordings, and inspect failures before fixing the protocol. If the RGB pipeline remains unreliable, separate oracle-perception diagnostics will help identify the source of error, but will not replace the RGB evaluation. The final claims will be restricted to the settings and evidence actually evaluated.

# References

\[1\] K. Kawaharazuka, J. Oh, J. Yamada, I. Posner, and Y. Zhu, “Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications,” IEEE Access, 2025, doi: 10.1109/ACCESS.2025.3609980.

\[2\] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “VGGT: Visual Geometry Grounded Transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 5294–5306.

\[3\] D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, Y. Ding, Z. Wang, J. Gu, B. Zhao, D. Wang, and X. Li, “SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model,” in Robotics: Science and Systems, 2025.

\[4\] B. Yu, S. Lian, X. Lin, Z. Shen, Y. Wei, H. Liu, C. Wu, H. Yuan, B. Wang, C. Huang, and K. Chen, “3D-Mix for VLA: A Plug-and-Play Module for Integrating VGGT-based 3D Information into Vision-Language-Action Models,” arXiv:2603.24393, 2026.

\[5\] P. Li, W. Guo, H. Zhang, T. Cai, X. He, Y. Guo, and H. Xiong, “Spatial Memory for Out-of-Vision Manipulation in Vision-Language-Action,” in Proceedings of the International Conference on Machine Learning, 2026, arXiv:2605.22283.

\[6\] D. Maggio, N. Gorlo, K. Hauser, and L. Carlone, “FOUND-IT: Foundation-model-first Task-driven 3D Scene Graphs with Granularity on Demand,” arXiv:2605.25371, 2026.

\[7\] A. Gelencsér-Horváth, G. Dinya, D. B. Erős, P. Halász, I. M. Muqsit, and K. Karacs, “SceneVGGT: VGGT-based Online 3D Semantic SLAM for Indoor Scene Understanding and Navigation,” arXiv:2602.15899, 2026.

\[8\] Y. Fan, X. Ma, R. Su, J. Guo, R. Wu, X. Chen, and Q. Li, “Embodied VideoAgent: Persistent Memory from Egocentric Videos and Embodied Sensors Enables Dynamic Scene Understanding,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 6342–6352.

\[9\] E. Bartoli, B. He, D. Rotondi, S. Koch, F. Tombari, K. O. Arras, P. Jensfelt, Y. Cai, and I. Leite, “GESTO: Human-Centric Spatio-Temporal Memory for Reasoning in Dynamic Scenes,” arXiv:2608.10886, 2026.
