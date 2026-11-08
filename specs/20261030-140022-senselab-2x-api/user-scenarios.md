# senselab 2.x: user scenarios

Status: proposed 2026-11-08, for owner review.

These scenarios describe what a person or an agent does with senselab and what they can observe
afterwards. They hold for every architecture: they name no module, file format, queue or storage
design. A change may implement them any way it likes; it may not break one.

Each scenario links the principles it exercises ([`principles.md`](principles.md),
[`engineering.md`](engineering.md)) and, where relevant, an ADR. The links are for tracing only; the
scenario text is the requirement.

Placeholders: `<recordings>` is a set of input files, `<shared-results>` a location a group can read,
`<site>` an execution setting (laptop, cluster, cloud account). No scenario depends on a real path,
account or dataset.

## How to use this

**Reviewers.**
- For each change, list the areas it touches (planning, execution, caching and reuse, models,
  datasets, environments, privacy, errors) and check it against the scenarios in those groups.
- Ask of each one: after this change, does every guarantee still hold, and can any "never" line now
  happen? Name the scenario in the finding (for example "breaks UX-14, guarantee 2").
- A violation is a blocking finding unless the owner amends the scenario. Amending is an edit here,
  dated, with the owner's words; it is never a silent exception in code.
- A scenario that cannot be checked for a change (no test, no run) is reported as unchecked, not as
  passing.

**Users.**
- The guarantees are what you can rely on in every release of 2.x, in every setting listed.
- If you observe a "never" line happening, it is a defect, whatever the setting.

**Terms.** A *plan* is the inspectable statement of what will be computed, from which inputs, with
which models, and where. A *result* is one output with its record of what produced it. *Reuse*
means a result already computed (by you, a colleague or an earlier run) is returned instead of
being computed again.

---

## A. Personal use on one machine

### UX-1 Transcribe a folder on a laptop without a GPU
- **Actor, setting:** a researcher on a laptop, CPU only, home directory storage.
- **Steps:** installs senselab, points a transcription task at 40 interview recordings, inspects
  the plan, runs it.
- **Guarantees:**
  1. The install does not require a GPU, a GPU driver or a system-wide library beyond what the
     install itself states.
  2. The plan says the work will run on the CPU, which models it will load, and how much will be
     downloaded before anything runs.
  3. Each transcript records the model's exact version, the parameters, the input's content
     identity and the hardware it ran on.
  4. Running the same command again returns the same transcripts from reuse, without loading a
     model.
- **Never:** silently picks a smaller or different model because no GPU is present; requires the
  user to edit the task or pipeline to run on CPU.
- **Relates to:** P-2, P-3, P-5, P-7, P-10, P-13, P-25.

### UX-2 The same work on a workstation with a GPU
- **Actor, setting:** the same researcher, same recordings, now on a workstation with one GPU.
- **Steps:** runs the identical task declaration; does not change any code.
- **Guarantees:**
  1. The plan places the model work on the GPU and states the batch and precision it chose.
  2. The transcripts carry the same identity as the laptop run apart from the recorded hardware,
     and any numeric difference between the two is reported against the measured tolerance for
     that model, not hidden.
  3. With the GPU busy or too small for the model, the plan says so before running and offers the
     CPU placement; it does not fail halfway.
- **Never:** the user selects the device inside the task call to get GPU execution; a GPU result
  and a CPU result are presented as exactly equal when they were not compared.
- **Relates to:** P-2, P-7, P-8, P-10, P-38.

### UX-3 One file at the prompt, a whole study later, same pipeline
- **Actor, setting:** a researcher at an interactive prompt or notebook.
- **Steps:** runs a pipeline (voice activity, transcription, speaker embeddings) on one recording
  to look at the output; a week later runs the same pipeline object over 2,000 recordings.
- **Guarantees:**
  1. The pipeline is written once; scaling up changes only the inputs and the execution setting.
  2. The first recording's results are reused in the large run.
  3. Results are returned as typed objects at the prompt and as the same records in bulk.
- **Never:** a "batch version" of a pipeline has to be written separately from the interactive one.
- **Relates to:** P-1, P-5, P-9, P-11.

### UX-4 A long recording on a small machine
- **Actor, setting:** a laptop with 8 GB of memory; one recording is three hours long.
- **Steps:** runs feature extraction and transcription over a set that includes the long file.
- **Guarantees:**
  1. The plan states the expected peak memory per step, or states that it is unmeasured.
  2. Peak memory does not grow with the length of the recording beyond what the plan stated, or
     the plan refuses the step on this machine before running and says why.
- **Never:** the machine runs out of memory mid-run on a file the plan accepted without warning.
- **Relates to:** P-5, P-7, P-33, P-34.

## B. Shared clusters (HPC)

### UX-5 A study on one cluster node
- **Actor, setting:** a lab member with a batch allocation of one node (CPUs, one or more GPUs) on
  a shared cluster with a shared filesystem and a scheduler.
- **Steps:** plans 10,000 recordings for the cluster setting, submits, logs out, checks back the
  next day.
- **Guarantees:**
  1. The plan for the cluster differs from the laptop plan only in where and how things run, never
     in what is computed.
  2. Progress (done, running, failed, waiting) can be read at any time without attaching to the
     job.
  3. All GPUs allocated are used when the work can use them; idle allocated hardware is visible in
     the progress report.
- **Never:** the user has to write a scheduler script by hand for the plan to run; logging out
  stops the work.
- **Relates to:** P-2, P-5, P-7, P-10, ADR-008.

### UX-6 A study across many nodes
- **Actor, setting:** the same lab member, now with 20 nodes over a day, some CPU-only, some with
  GPUs, under a per-user CPU and GPU cap set by the cluster.
- **Steps:** runs the same plan with a multi-node setting.
- **Guarantees:**
  1. Each recording's result is computed once, however many nodes run.
  2. CPU-only steps run on CPU nodes and GPU steps on GPU nodes, as the plan stated.
  3. The run stays within the site's caps; it never submits more than the user is allowed.
  4. Results are identical to a single-node run of the same plan, apart from recorded hardware.
- **Never:** two nodes compute and write the same result and one silently overwrites the other;
  a slow shared filesystem turns into duplicated work.
- **Relates to:** P-2, P-5, P-10, P-33, ADR-003.

### UX-7 Preemptible or spot allocations
- **Actor, setting:** a user running on a preemptible partition or spot instances that can be
  reclaimed with little notice.
- **Steps:** runs a 5,000-recording plan; a third of the workers are reclaimed during the run;
  new workers start later.
- **Guarantees:**
  1. Work finished before a reclaim is kept; only work in progress at the reclaim is redone.
  2. The new workers pick up remaining work without the user resubmitting anything item by item.
  3. The final report counts reclaimed attempts separately from failures of the data or the model.
- **Never:** a reclaim is reported as a failed recording; a partial result from a reclaimed worker
  is returned as complete.
- **Relates to:** P-5, P-34, ADR-009.

### UX-8 Interrupt and resume
- **Actor, setting:** any setting; the user stops a run (Ctrl-C, job cancel, wall-time limit).
- **Steps:** later, runs the same plan again, possibly on another day or node.
- **Guarantees:**
  1. Only missing results are computed; the report says how many were reused and how many ran.
  2. The resumed run's results are indistinguishable from an uninterrupted run's.
- **Never:** the user must clean up anything left by the interrupted run before resuming; a
  resumed run recomputes finished work because the host or day changed.
- **Relates to:** P-5, P-10, P-19.

## C. Groups and sharing

### UX-9 One person's results serve a colleague
- **Actor, setting:** two members of a lab share a results location (`<shared-results>`) on the
  cluster.
- **Steps:** member A transcribes a corpus; member B later runs a pipeline whose first step is the
  same transcription of the same recordings with the same model and parameters.
- **Guarantees:**
  1. B's run reuses A's transcripts and the plan says so before running ("1,800 of 2,000 reused").
  2. Each reused result still names what produced it; B can see it came from an earlier run.
  3. If B chose a different model revision or parameter, nothing is reused for that step.
- **Never:** B gets A's result for a different model version, parameter or input; B has to copy
  A's files by hand to benefit.
- **Relates to:** P-3, P-4, P-5, P-19, P-38.

### UX-10 Trust and permissions in a shared location
- **Actor, setting:** a group shares results; not everyone has write access, and members may run
  different senselab versions.
- **Steps:** a member with read-only access runs a plan against the shared location; another member
  runs a newer senselab release against the same location.
- **Guarantees:**
  1. A read-only member can reuse shared results and keeps new results in their own location.
  2. A result that fails its integrity check is not reused; it is reported and recomputed.
  3. Users choose how strict reuse is (exact only, or measured-equivalent across machines), and
     the choice is recorded with each result reused.
  4. Mixed releases either share results under the same identity rules or keep them apart; a
     newer release never misreads an older one's results.
- **Never:** a member can make others reuse a result that was not produced by the stated model on
  the stated input; lacking write access makes the run fail rather than fall back to a private
  location.
- **Relates to:** P-3, P-19, P-38, ADR-002, ADR-007.

### UX-11 Measurements shared, decisions kept separate
- **Actor, setting:** a group in which one member computes acoustic measurements for a corpus and
  others apply different thresholds or reviews to them.
- **Steps:** member A publishes measurements; members B and C apply their own thresholds.
- **Guarantees:**
  1. B and C reuse A's measurements; changing a threshold recomputes only what depends on it.
  2. Every threshold applied is visible as configuration with its stated derivation.
- **Never:** a threshold change forces the measurements to be recomputed; a threshold is buried
  where users cannot see or change it.
- **Relates to:** P-12, P-17, P-18.

### UX-12 Personal use stays simple
- **Actor, setting:** a single user who never shares anything.
- **Steps:** installs and runs without configuring any shared location, scheduler or group.
- **Guarantees:** everything in group A works with defaults; sharing features cost nothing when
  unused.
- **Never:** a single user must set up group storage, accounts or a server to run anything.
- **Relates to:** P-1, P-13.

## D. Cloud extensions

### UX-13 Results and inputs in object storage
- **Actor, setting:** a group keeps recordings and shared results in cloud object storage.
- **Steps:** runs a plan whose inputs and results location are object-storage locations.
- **Guarantees:**
  1. Identity, reuse and provenance behave exactly as with a shared filesystem.
  2. A result appears complete or not at all; a reader never sees half of one.
  3. Credentials are taken from the user's environment and never written into results, plans,
     logs or provenance.
- **Never:** moving storage from a filesystem to object storage changes any result's identity or
  any task or pipeline definition.
- **Relates to:** P-2, P-3, P-11, ADR-007.

### UX-14 Ephemeral cloud nodes
- **Actor, setting:** workers start on fresh cloud machines that have nothing installed and are
  discarded afterwards.
- **Steps:** runs a plan across 50 such machines.
- **Guarantees:**
  1. Each machine builds or fetches exactly the locked environments and model versions the plan
     names; none resolves a newer one on its own.
  2. A machine discarded mid-task is handled as in UX-7.
  3. The plan states what each new machine will download before it starts work.
- **Never:** a fresh machine produces a result with a different dependency or model version than
  the plan stated.
- **Relates to:** P-4, P-6, P-16, P-25.

### UX-15 Bursting from a cluster to the cloud
- **Actor, setting:** a lab's cluster queue is full before a deadline; the lab adds cloud machines
  to the same run.
- **Steps:** keeps the cluster workers running and adds cloud workers to the same plan.
- **Guarantees:**
  1. Work is not duplicated between the two; each result is computed once.
  2. Results from both sites carry their recorded hardware and are equal or reported against the
     model's measured tolerance.
  3. If cloud workers cannot reach the shared results, the plan or the first check says so before
     they take work.
- **Never:** adding a site requires a new plan identity or changes what is computed.
- **Relates to:** P-2, P-10, P-38.

## E. Moving a plan between machines

### UX-16 Plan on a laptop, run on the cluster
- **Actor, setting:** a researcher drafts and inspects a plan on a laptop, then runs it on a
  cluster.
- **Steps:** produces a dry-run plan locally, reviews it, carries it to the cluster, runs it there.
- **Guarantees:**
  1. The plan keeps one identity across hosts; only the execution part is re-adapted to the
     cluster.
  2. What is computed (tasks, models, parameters, inputs) is exactly what was reviewed.
  3. If the cluster cannot honour part of the plan (a model not permitted there, an environment
     that cannot build), it refuses before running and names the part.
- **Never:** the cluster quietly substitutes a model, a parameter or a dependency version.
- **Relates to:** P-2, P-10, P-3.

### UX-17 Reproducing a colleague's result elsewhere
- **Actor, setting:** a collaborator at another institution receives a result record.
- **Steps:** reads the record, re-runs the same computation on their own hardware.
- **Guarantees:**
  1. The record alone states the exact model content, parameters, input identity, software and
     environment needed to reproduce it.
  2. The re-run's output is compared with the original under a stated tolerance, and the report
     says whether they agree.
- **Never:** the record carries the original user's paths, account names or hostnames.
- **Relates to:** P-3, P-4, P-28, P-38, ADR-010.

## F. Models and versions

### UX-18 Reproducing last year's result
- **Actor, setting:** a researcher must reproduce a figure from a run a year ago; the model's
  public repository has since been updated.
- **Steps:** re-runs from last year's plan or result record.
- **Guarantees:**
  1. The same model content is loaded as last year, not the repository's current version.
  2. If that content is no longer obtainable, the run stops and says which model version is
     missing and where it was expected; it does not fall back.
- **Never:** a moving name ("latest", a branch) is used to load a model; a result is recorded with
  one model version while computed with another.
- **Relates to:** P-3, P-4, P-16.

### UX-19 Adopting a new model version deliberately
- **Actor, setting:** a new revision of a model the lab uses is released.
- **Steps:** the user changes the model revision in configuration and plans the corpus again.
- **Guarantees:**
  1. Nothing changes until the user asks for the new version.
  2. The plan shows which steps will be recomputed because of the change and which results remain
     reusable.
  3. Old and new results coexist, each naming its model version, so they can be compared.
- **Never:** a model update is picked up silently by an install, an upgrade or a fresh machine; new
  results overwrite old ones.
- **Relates to:** P-3, P-4, P-16, P-18, P-19.

### UX-20 Changing the inference engine, not the model
- **Actor, setting:** a user switches the engine serving a model (for speed) without changing the
  model.
- **Steps:** changes only the execution setting.
- **Guarantees:**
  1. The task call and the result shape are unchanged.
  2. The engine is recorded, and results are treated as distinct where the engine can change the
     output.
- **Never:** an engine change requires editing the pipeline.
- **Relates to:** P-8, P-2.

### UX-21 Licence visibility
- **Actor, setting:** a user preparing a commercial or clinical deployment.
- **Steps:** asks what licences a plan involves before running it, under a policy that allows only
  permissive licences.
- **Guarantees:**
  1. The answer covers every model, package and dataset the plan uses, without running anything.
  2. A plan that violates the policy, or uses something with an unknown licence, is refused before
     running unless the user relaxed the policy, and the relaxation is recorded.
- **Never:** a licence is discovered only after results exist.
- **Relates to:** P-23.

## G. Dataset changes

### UX-22 Adding recordings to a finished study
- **Actor, setting:** 200 new recordings are added to a 2,000-recording study.
- **Steps:** re-plans and runs the same pipeline over the grown dataset.
- **Guarantees:** only the 200 new recordings and anything that depends on the whole set are
  computed; the plan says so beforehand.
- **Never:** adding inputs invalidates results for unchanged inputs.
- **Relates to:** P-5, P-11, P-19.

### UX-23 Duplicates and renamed files
- **Actor, setting:** the same recording appears twice under different names, or a folder is
  renamed.
- **Steps:** runs the pipeline.
- **Guarantees:** identical content is computed once and each name gets its result; a rename does
  not cause recomputation.
- **Never:** two different recordings sharing a name are treated as one.
- **Relates to:** P-3, P-19, ADR-002.

### UX-24 A bad input among many
- **Actor, setting:** one of 5,000 recordings is truncated, another is silent, another is not
  audio.
- **Steps:** runs the pipeline over all of them.
- **Guarantees:**
  1. The other recordings complete.
  2. Each bad input is listed with its cause in one line and what to do about it.
  3. Re-running after fixing or removing a bad input computes only that input.
- **Never:** one bad file stops the run; a bad file yields an empty result that looks valid.
- **Relates to:** P-34, P-31, ADR-009.

### UX-25 A file changes after it was processed
- **Actor, setting:** a recording is edited (trimmed, re-exported) after results exist for it.
- **Steps:** re-runs.
- **Guarantees:** the changed recording is recomputed; its old results are not presented as
  belonging to the new content.
- **Never:** a result is matched to an input by name or date alone.
- **Relates to:** P-3, P-19.

### UX-26 Another dataset, same pipeline
- **Actor, setting:** a lab applies its pipeline to a second dataset with different file naming
  and metadata fields.
- **Steps:** writes a dataset configuration (naming, metadata mapping, fitted thresholds) and runs
  the unchanged pipeline.
- **Guarantees:** the pipeline code is not edited; all dataset knowledge is in that configuration.
- **Never:** a pipeline or task only works for the dataset it was first written for.
- **Relates to:** P-11, P-12, P-17.

## H. Offline and air-gapped use

### UX-27 Prepare online, run offline
- **Actor, setting:** a secure enclave with no internet access.
- **Steps:** on a connected machine, asks senselab to fetch everything a plan needs; moves it into
  the enclave; runs the plan with networking declared off.
- **Guarantees:**
  1. The fetch step lists exactly what the plan needs (models, environments, reference data).
  2. In the enclave, the run makes no network attempt; anything missing is reported by name before
     work starts.
  3. Results are identical in identity to an online run of the same plan.
- **Never:** a run waits on network timeouts in an offline setting; a missing model is replaced by
  another one that happens to be present.
- **Relates to:** P-4, P-6, P-25, ADR-007.

## I. Sensitive data

### UX-28 Identifiable participants
- **Actor, setting:** a clinical study with identifiable voice recordings and participant metadata.
- **Steps:** runs pipelines, shares results with a collaborator, and reports a bug upstream with a
  log.
- **Guarantees:**
  1. Logs, progress reports, error messages and provenance identify inputs by content identity or
     a user-chosen label, not by participant names, paths or metadata values, unless the user
     opts in.
  2. Sharing results shares only what the user selects; exporting provenance applies the
     configured redaction.
  3. Nothing about the data leaves the machine or site (no telemetry, no remote service) unless the
     user configured that service.
- **Never:** a transcript, participant identifier or input path appears in a log, an error, a
  shared reuse record or an exported provenance file the user did not choose to include.
- **Relates to:** P-3, P-28, ADR-007, ADR-010.

### UX-29 Sensitive results in a shared location
- **Actor, setting:** a group in which some studies are restricted to named members.
- **Steps:** a member runs a restricted study with a shared results location configured.
- **Guarantees:** restricted results are written only where the user's configuration allows; reuse
  never makes one study's restricted content readable by someone without access to that study.
- **Never:** reuse turns a permission boundary into a leak (for example, by revealing whether a
  given recording was processed).
- **Relates to:** P-38, ADR-007.

### UX-30 Remote model code
- **Actor, setting:** a user loads a model whose repository ships executable code.
- **Steps:** plans a task with that model.
- **Guarantees:** the plan says the model runs its own code, and it runs only if the user's
  allowlist permits that source.
- **Never:** downloaded model code executes without an explicit permission.
- **Relates to:** P-6, ADR-007.

## J. Seeing cost and reuse before running

### UX-31 An estimate before committing hardware
- **Actor, setting:** a user deciding whether to run 20,000 recordings on a laptop or request
  cluster time.
- **Steps:** asks for the plan for each setting, without running.
- **Guarantees:**
  1. Each plan states expected wall time, GPU hours, peak memory, disk and download size, each
     either as a figure from measurement or explicitly as unmeasured.
  2. The plan states how much will be reused and how much computed.
  3. After the run, the report shows actual figures next to the estimate.
- **Never:** a figure is shown without saying whether it was measured; producing the plan starts
  computation or downloads models.
- **Relates to:** P-5, P-7, P-10, P-33.

### UX-32 Explaining a plan
- **Actor, setting:** a reviewer or supervisor reading someone else's plan.
- **Steps:** renders the plan's steps and their connections without running.
- **Guarantees:** every step, model, parameter, environment and placement is readable, nested
  pipelines one level at a time.
- **Never:** what a plan will do can only be learned by running it.
- **Relates to:** P-9, P-10, P-31.

## K. AI agents

### UX-33 An agent plans from the public description alone
- **Actor, setting:** an AI agent with no access to the source code or repository, given a
  research goal ("measure speech rate for each recording in `<recordings>`").
- **Steps:** lists capabilities, reads their machine-readable descriptions, builds a plan, inspects
  its cost and licences, then runs it.
- **Guarantees:**
  1. Every task, model, backend and pipeline it can call is listed with typed parameters,
     defaults, outputs, supported hardware and licence.
  2. A plan built from the descriptions alone is valid; an invalid one is refused with one-line
     errors naming the cause and the fix.
  3. The agent sees the same results and records a human would.
- **Never:** an agent must read source, documentation pages or examples to call something
  correctly; a description differs from what the code accepts.
- **Relates to:** P-1, P-10, P-23, P-31.

### UX-34 An agent cannot exceed what the user allowed
- **Actor, setting:** an agent acting for a user under the user's site, licence and privacy
  settings.
- **Steps:** plans and runs work.
- **Guarantees:** the agent's plans are bound by the same policies (licences, allowed model
  sources, network, resource caps) as the user's own.
- **Never:** a plan submitted by an agent bypasses a policy that would refuse the same plan from
  the user.
- **Relates to:** P-1, P-23, ADR-007.

## L. Failure modes

Every failure below is reported with its kind (data, model, environment, infrastructure, user
error), the affected items and the next action, in one line per cause.

### UX-35 Out of memory
- **Setting:** any; a step exceeds host or GPU memory on some inputs.
- **Guarantees:** the affected items are retried in a smaller configuration where one exists, or
  reported as out of memory with the size that failed; other items continue.
- **Never:** an out-of-memory crash loses results already finished; it is reported as a data error.
- **Relates to:** P-7, P-34, ADR-009.

### UX-36 Node or worker loss
- **Setting:** cluster or cloud; a node dies, or a worker process is killed.
- **Guarantees:** its in-progress items are taken over after a bounded, stated delay; finished
  results are kept.
- **Never:** a dead worker blocks others indefinitely; an item retries forever without being
  reported.
- **Relates to:** P-5, P-34, ADR-003.

### UX-37 Network or storage outage
- **Setting:** the shared filesystem, object storage or model host is unreachable for a time.
- **Guarantees:** the run pauses or fails with an infrastructure error naming the unreachable
  location; transient outages are retried within stated limits; when service returns, resuming
  (UX-8) completes the work.
- **Never:** an outage is recorded as a failed input; a half-written result is later reused.
- **Relates to:** P-34, ADR-003, ADR-009.

### UX-38 A corrupted or tampered cache
- **Setting:** a stored result is truncated, altered or replaced.
- **Guarantees:** it is detected when read, not reused, reported, and recomputed.
- **Never:** a corrupted result is returned or used downstream.
- **Relates to:** P-3, P-19, P-34, ADR-007.

### UX-39 A crash mid-run
- **Setting:** the process running the plan crashes or the machine reboots.
- **Guarantees:** running the plan again resumes (UX-8); results written before the crash are
  either complete and reusable or absent.
- **Never:** the crash leaves state that blocks the next run until removed by hand.
- **Relates to:** P-5, P-34.

### UX-40 A backend that cannot install
- **Setting:** an isolated environment for one backend cannot be built (no compatible wheel for
  this platform or GPU, a quarantined package, no network).
- **Guarantees:**
  1. This is reported at planning or environment preparation, before work starts where possible,
     naming the backend, the platform and the action (another backend, another setting, an
     override).
  2. Steps that do not need the backend still run.
- **Never:** senselab falls back to unlocked or system packages, or to another backend, without
  the user choosing it.
- **Relates to:** P-6, P-25, P-34.

### UX-41 A gated or missing model
- **Setting:** a model needs an access token the user lacks, or its repository was removed.
- **Guarantees:** the plan or preparation step names the model and what is needed (accept the
  terms, provide a token, supply the weights) before work starts.
- **Never:** the failure appears hours into a run, or as a generic download error.
- **Relates to:** P-4, P-23, P-31.

### UX-42 Two people run the same plan at once
- **Setting:** two lab members start overlapping plans on the same shared results location.
- **Guarantees:** each result is computed once; both runs see it; both finish.
- **Never:** they corrupt each other's results or block each other permanently.
- **Relates to:** P-5, ADR-003.

## M. Provenance

### UX-43 What produced this result
- **Actor, setting:** a user, reviewer or auditor holding any one result, in any setting.
- **Steps:** asks what produced it.
- **Guarantees:** the answer states, from the record alone:
  1. the inputs and their content identity;
  2. the task, its version and the effective parameters, including defaults the user did not set;
  3. the model and its exact revision or content hash, and the inference engine;
  4. the software and environment that ran it;
  5. where (hardware, site kind) and when it ran;
  6. whether it was computed or reused, and if reused, from which earlier run or group location;
  7. which items of the same run failed or were skipped, and why.
- **Never:** the record names a model revision, parameter or environment other than the one that
  ran; a field is filled with a guess instead of being marked unknown.
- **Relates to:** P-3, P-4, P-6, P-8, P-16, ADR-002, ADR-010.

### UX-44 From a figure back to its records
- **Actor, setting:** a researcher preparing a paper; a figure summarises 2,000 results.
- **Steps:** asks which results, inputs, models and parameters the figure was built from.
- **Guarantees:** every number in the figure traces to the results it came from, and each of those
  to its record (UX-43); results excluded from the figure are listed with the reason.
- **Never:** a figure's inputs can be reconstructed only from memory or notes.
- **Relates to:** P-3, P-9, ADR-010.

### UX-45 Export provenance to cite or share
- **Actor, setting:** a user depositing results with a paper or sending them to a collaborator.
- **Steps:** exports the provenance of a selected set of results.
- **Guarantees:**
  1. The export is in a standard, documented provenance form that tools unaware of senselab can
     read.
  2. It contains what is needed to cite and reproduce the results (UX-17), and the configured
     redaction is applied.
- **Never:** the export carries local paths, account names, hostnames or participant identifiers
  the user did not choose to include.
- **Relates to:** P-3, P-28, ADR-007, ADR-010.

### UX-46 Provenance survives moves, reuse and resume
- **Actor, setting:** results copied to another machine, served to a colleague by reuse, or
  completed across an interrupted and resumed run.
- **Steps:** asks for the provenance of those results afterwards.
- **Guarantees:**
  1. The record is complete and the same after a copy to another machine or storage.
  2. A reused result keeps the record of the run that computed it and adds that it was reused, by
     which run.
  3. A resumed run's record shows both attempts; nothing finished before the interruption loses
     its record.
- **Never:** provenance is silently dropped, truncated or replaced on reuse, resume or move.
- **Relates to:** P-3, P-5, P-10, P-19, ADR-010.

## N. Reports

### UX-47 A readable report of a run
- **Actor, setting:** a user after a run, or partway through one.
- **Steps:** asks for the run's report.
- **Guarantees:** the report states what ran (plan, models, parameters, setting); counts of done,
  reused, failed, skipped and pending items; each failed item's cause; timing and cost against the
  estimate (UX-31); and quality summaries the pipeline declares. Partway through, it says it is
  partial and what remains.
- **Never:** a report claims success while items failed, were skipped or are pending; counts in
  the report disagree with the stored results.
- **Relates to:** P-5, P-10, P-31, ADR-008.

### UX-48 An interactive report
- **Actor, setting:** a user reviewing a corpus run.
- **Steps:** opens the interactive version of the report.
- **Guarantees:** the user can browse records, filter to failures or to any declared measure,
  listen to or view an item's input next to its outputs, and open that item's provenance (UX-43).
- **Never:** the interactive view shows data that the static report and stored results do not
  support; it requires a server or account the user did not set up.
- **Relates to:** P-1, P-3, P-24, ADR-008.

### UX-49 Reports regenerate without rerunning
- **Actor, setting:** a user, or an AI agent, a month after a run, on any machine with the
  results.
- **Steps:** asks for the report again, or in another form.
- **Guarantees:** the report is rebuilt from stored results and records alone, without computing
  or downloading anything; an agent gets the same report a human does.
- **Never:** regenerating a report reruns tasks or loads models; an agent's report and a human's
  report of the same run differ in content.
- **Relates to:** P-1, P-3, P-5.

### UX-50 Reports for group and sensitive runs
- **Actor, setting:** a group run on restricted, identifiable data.
- **Steps:** a member generates and shares a report.
- **Guarantees:**
  1. A reader sees only the items and inputs they have access to.
  2. Audio, video, transcripts and participant metadata are embedded only if the user asks and is
     allowed; otherwise items appear by content identity or a user-chosen label.
- **Never:** a report leaks identifiable participant data, input paths or another member's
  restricted results.
- **Relates to:** P-28, P-38, ADR-007, ADR-010.

---

## Change log

- 2026-11-08: proposed, 50 scenarios in groups A–N (M provenance and N reports added at the owner's
  request: "add in provenance related information and report (potentially interactive) generation
  for users").
