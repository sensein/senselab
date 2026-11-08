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
- List the areas a change touches and check it against every scenario in those groups:

  | Area | Groups |
  |---|---|
  | running on one machine, devices, placement | A, O |
  | clusters, preemptible resources, resume | B |
  | groups, sharing, reuse across people | C |
  | cloud storage, cloud machines, hosted models, spend | D |
  | plans moved between hosts | E |
  | models, versions, engines, licences, senselab upgrades, extensions | F |
  | datasets and input changes | G |
  | offline use, downloads, environments | H, F, L |
  | sensitive data and identifier visibility (and Terms) | I, M, N |
  | cost, plan inspection | J |
  | agents | K |
  | failures, errors, retries | L |
  | provenance | M |
  | reports | N |
  | video, text, multimodal timelines, languages, long signals | O |
  | caching and reuse | B, C, G, M |

- Ask of each scenario: after this change, does every guarantee still hold, and can any "never"
  line now happen?
- A blocking finding quotes the scenario id and guarantee (for example "UX-14, guarantee 2") and
  the `file:line`, test or run that breaks it. A finding without a quoted location is not blocking
  (as `engineering.md` requires).
- A scenario that a change touches but that was not checked (no test, no run) is recorded in the
  review as unchecked. It is never assumed to pass.
- Where a scenario and an ADR disagree, the scenario governs and the ADR gets a dated amendment
  (ADR-000) before the next change in that area merges.
- Only the owner amends or adds a scenario, as a dated edit here with the owner's words. An agent
  or reviewer may propose an amendment, never apply one, and code never carries a silent exception.

**Users.**
- The guarantees are what you can rely on in every release of 2.x, in every setting listed.
- If you observe a "never" line happening, it is a defect, whatever the setting.

## Terms

- **Plan.** The inspectable statement of what will be computed, from which inputs, with which
  models, and where.
- **Result.** One output with its record of what produced it.
- **Identity.** What makes two results the same computation: input content, task and its version,
  model content, effective parameters, and the inference engine where it can change outputs. Hardware, host and environment are recorded in provenance,
  not part of identity.
- **Reuse.** A result already computed (by you, a colleague or an earlier run) is returned instead
  of being computed again. Across contexts (another host, hardware or environment), the user chooses
  the reuse level: exact, measured-equivalent, or any context (the result is labelled as such in its
  record, with no comparison made). Unmeasured cases behave as exact.
- **Stated tolerance.** A configured threshold per output, with its derivation. Where none is
  declared for an output, two results are reported as uncompared, never as agreeing.
- **Silent fallback.** Not allowed anywhere. Anything that changes what is computed, which model or
  configuration runs, or where results go is the user's choice, visible in the plan or the report.
- **Identifiers.** Participant and record identifiers (names, study ids, metadata values, input
  paths) and the identifiers of people and places that ran the work (account names, hostnames,
  site names).
- **Data owner.** The person or group that holds the data and sets its visibility policy. When
  several hold the same corpus, each sets the policy for their own copy and for what they share.
  With nothing configured, the data owner is the user who runs, for their own runs.
- **Audience.** Whoever sees an output: the data owner, an approved team, a collaborator, a group,
  other users of a cluster who can read a location, a public export. An AI agent is an audience like
  any other: it sees what its principal (the user it acts for) is allowed to see.
- **Visibility policy.** Set by the data owner per project or dataset, it decides per audience
  whether identifiers are shown as real values, as pseudonymous ids, or not at all. Plans, logs,
  progress views, errors, provenance, reports and exports of the same run render according to who views or receives
  them.
  - **Default.** With no policy set, the data owner sees real identifiers and every other audience
    sees pseudonymous ids.
  - **Pseudonyms** are stable per dataset: the same identifier maps to the same pseudonym across
    runs, outputs and every audience allowed pseudonyms, so those allowed can link results.
  - This is a runtime rule. P-28 is a separate repository rule (no identifiers in tracked files or
    commits).

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
- **Actor, setting:** the same researcher, same recordings, now on a workstation with one GPU
  (a discrete GPU or Apple silicon).
- **Steps:** runs the identical task declaration; does not change any code.
- **Guarantees:**
  1. The plan states the device and numeric precision it will use, and any setting that can change
     outputs.
  2. The transcripts have the same identity as the laptop run; the hardware is recorded in
     provenance. Any numeric difference between the two is reported against the stated tolerance for that output (none declared: uncompared).
  3. With the GPU busy or too small for the model, the plan says so before running and offers the
     CPU placement as the user's choice; it does not fail halfway.
- **Never:** the user selects the device inside the task call to get GPU execution; a GPU result
  and a CPU result are said to agree when they were not compared.
- **Relates to:** P-2, P-7, P-8, P-10, P-38.

### UX-3 One file at the prompt, a whole study later, same pipeline
- **Actor, setting:** a researcher at an interactive prompt or notebook.
- **Steps:** runs a pipeline (voice activity, transcription, speaker embeddings) on one recording
  to look at the output; a week later runs the same pipeline declaration over 2,000 recordings.
- **Guarantees:**
  1. The pipeline is written once; scaling up changes only the inputs and the execution setting.
  2. The first recording's results are reused in the large run, under a reuse level that admits
     its setting.
  3. An item's result has the same fields and values whether obtained interactively or from the
     bulk run.
- **Never:** a "batch version" of a pipeline has to be written separately from the interactive one.
- **Relates to:** P-1, P-5, P-9, P-11.

### UX-4 A long recording on a small machine
- **Actor, setting:** a laptop with 8 GB of memory; one recording is three hours long.
- **Steps:** runs feature extraction and transcription over a set that includes the long file.
- **Guarantees:**
  1. The plan states the expected peak memory per step, or states that it is unmeasured; an
     unmeasured statement is the warning, and out-of-memory then follows UX-35.
  2. Peak memory per step is bounded independently of recording length. The plan refuses a step
     only when its bounded requirement exceeds the machine, and names the requirement.
- **Never:** the machine runs out of memory mid-run on a file the plan accepted without warning; a
  step's memory grows with the length of the recording.
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
     job, rendered under the reader's audience rule (Terms).
  3. The plan states how many allocated GPUs it will use and why any stay idle; progress shows
     per-device use.
- **Never:** the user has to write a scheduler script by hand for the plan to run; logging out
  stops the work.
- **Relates to:** P-2, P-5, P-7, P-10, ADR-008.

### UX-6 A study across many nodes
- **Actor, setting:** the same lab member, now with 20 nodes over a day, some CPU-only, some with
  GPUs of different types, under a per-user CPU and GPU cap set by the cluster.
- **Steps:** runs the same plan with a multi-node setting.
- **Guarantees:**
  1. Each recording's result is computed once, however many nodes run.
  2. CPU-only steps run on CPU nodes and GPU steps on GPU nodes, as the plan stated.
  3. The run stays within the site's caps; it never submits more than the user is allowed.
  4. Results equal a single-node run of the same plan, or are reported against the stated tolerance for that output (none declared: uncompared)
     where hardware differs.
- **Never:** two nodes compute and write the same result and one silently overwrites the other;
  a slow shared filesystem turns into duplicated work.
- **Relates to:** P-2, P-5, P-10, P-33, P-38, ADR-003.

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
  2. The results, their values and identity, equal those of an uninterrupted run, or are reported
     against the stated tolerance for that output (none declared: uncompared) where the hardware differs (UX-6).
  3. The run's provenance shows the interruption and both attempts; each result's record is the
     record of the attempt that produced it (UX-46).
- **Never:** the user must clean up anything left by the interrupted run before resuming; a
  resumed run recomputes finished work because the day changed, or because the host changed within
  a reuse level that admits the new host.
- **Relates to:** P-5, P-10, P-19.

### UX-56 Disk quota or scratch purge
- **Actor, setting:** a cluster user whose scratch space has a quota and a purge policy.
- **Steps:** a run hits the quota midway; weeks later the purge deletes some stored results.
- **Guarantees:**
  1. Hitting the quota stops new writes with an infrastructure error naming the location and the
     space needed; finished results are kept, and resuming after space is freed completes the run.
  2. Purged results are detected as missing and recomputed when next needed; the plan says how
     many will be.
- **Never:** a result partly written when the quota hit, or partly purged, is read as complete.
- **Relates to:** P-5, P-19, P-34, ADR-009.

### UX-57 Logs other cluster users can read
- **Actor, setting:** a shared cluster where scheduler output files or scratch locations are
  readable by other users.
- **Steps:** runs a plan on identifiable data whose logs land in such a location.
- **Guarantees:** anything written where other users can read it follows the visibility rule of
  the widest audience that can read that location; the log states the rule it was rendered under,
  and the data owner's own rendering stays available to the data owner.
- **Never:** identifiers appear in a world-readable log under a rule meant only for the data owner.
- **Relates to:** P-3, ADR-007, ADR-008.

## C. Groups and sharing

### UX-9 One person's results serve a colleague
- **Actor, setting:** two members of a lab share a results location (`<shared-results>`) on the
  cluster.
- **Steps:** member A transcribes a corpus; member B later runs a pipeline whose first step is the
  same transcription of the same recordings with the same model and parameters.
- **Guarantees:**
  1. B's run reuses A's transcripts and the plan says so before running ("1,800 of 2,000 reused").
  2. Each reused result still names what produced it; B can see it came from an earlier run,
     rendered under B's audience rule (Terms).
  3. If B chose a different model revision or parameter, nothing is reused for that step.
- **Never:** B gets A's result for a different model version, parameter or input; B has to copy
  A's files by hand to benefit.
- **Relates to:** P-3, P-4, P-5, P-19, P-38.

### UX-10 Trust and permissions in a shared location
- **Actor, setting:** a group shares results; not everyone has write access, and members may run
  different senselab releases.
- **Steps:** a member with read-only access runs a plan against the shared location; another member
  runs a newer senselab release against the same location.
- **Guarantees:**
  1. A read-only member can reuse shared results; where new results will go is stated in the plan
     before running.
  2. A result that fails its integrity check is not reused; it is reported and recomputed.
  3. Users choose the reuse level (Terms), and the level is recorded with each result reused.
  4. A release that cannot interpret a stored result refuses it with a typed error and does not
     reuse it.
- **Never:** a result whose stored content does not match its recorded identity is reused; a member
  without write access can make others reuse anything; the run discovers missing write access only
  after computing; a value is returned that the release cannot interpret.
- **Relates to:** P-3, P-19, P-38, ADR-002, ADR-007.

### UX-11 Shared measurements, different thresholds
- **Actor, setting:** a group in which one member computes acoustic measurements for a corpus and
  others apply different thresholds to them.
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
- **Guarantees:** with no sharing configured, the install adds no group, server or account
  dependency, the plan names no shared location, and run time does not depend on sharing being
  available. The default visibility policy (Terms) applies.
- **Never:** a single user must set up group storage, accounts or a server to run anything.
- **Relates to:** P-1, P-13.

### UX-58 Models and environments shared by a group
- **Actor, setting:** ten lab members on one cluster use the same models and backends.
- **Steps:** each runs plans that need the same model versions and environments, with a group
  cache configured.
- **Guarantees:**
  1. Each model version and environment is downloaded or built once for the group, not per user;
     the plan says what is already present.
  2. A group cache whose permissions are wrong is reported as one permissions error naming the
     location, before work starts.
- **Never:** wrong permissions appear as scattered, unrelated failures; one member's partial
  download is used by another.
- **Relates to:** P-4, P-5, P-6, P-25, ADR-003.

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
  1. Each machine installs exactly the dependency versions and model versions the plan names; none
     resolves a newer one on its own.
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
  2. Results from both sites carry their recorded hardware and are equal or reported against
     the stated tolerance for that output (none declared: uncompared).
  3. If cloud workers cannot reach the shared results, the plan or the first check says so before
     they take work.
- **Never:** adding a site requires a new plan identity or changes what is computed.
- **Relates to:** P-2, P-10, P-38.

### UX-59 A hosted model behind an API
- **Actor, setting:** a user calls a model served by an external service, whose weights cannot be
  inspected or hashed.
- **Steps:** plans a task with that model over 500 recordings.
- **Guarantees:**
  1. The plan names the external service and states that input data will leave the site; data is
     sent only if the user has permitted that service.
  2. Each result records the service and the model version the service reports, and is marked as
     not exactly reproducible.
  3. Visibility policy still applies to anything senselab writes about the call.
- **Never:** a hosted result is presented as identified by model content; data is sent to a service
  the user did not permit.
- **Relates to:** P-3, P-4, P-23, ADR-007.

### UX-60 Cloud spend before and during a run
- **Actor, setting:** a user paying for cloud machines, storage, transfer or hosted model calls.
- **Steps:** asks for the plan, sets a spend cap, runs.
- **Guarantees:**
  1. The plan estimates money and data transfer out of the cloud, marked as measured or not.
  2. New work stops before the cap would be exceeded; the report says what remains and what it
     would cost.
- **Never:** a run exceeds a user-set cap; spend is visible only on the provider's bill.
- **Relates to:** P-5, P-10, P-33.

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
- **Relates to:** P-2, P-3, P-10.

### UX-17 Reproducing a colleague's result elsewhere
- **Actor, setting:** a collaborator at another institution receives a result record.
- **Steps:** reads the record, re-runs the same computation on their own hardware.
- **Guarantees:**
  1. The record alone states the exact model content, parameters, input identity, software and
     environment needed to reproduce it.
  2. The re-run's output is compared with the original under the stated tolerance for that output (none declared: uncompared), and the report
     says whether they agree.
- **Never:** the record shows the collaborator identifiers beyond what the visibility policy allows
  the collaborator; fields the policy allows that audience, including those needed to reproduce, are
  stripped.
- **Relates to:** P-3, P-4, P-38, ADR-007, ADR-010.

## F. Models, versions and extensions

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
  3. Old and new results coexist, each naming its model version, so they can be compared (UX-66).
- **Never:** a model update is picked up silently by an install, an upgrade or a fresh machine; new
  results overwrite old ones.
- **Relates to:** P-3, P-4, P-16, P-18, P-19.

### UX-20 Changing the inference engine, not the model
- **Actor, setting:** a user switches the engine serving a model (for speed) without changing the
  model.
- **Steps:** changes only the execution setting.
- **Guarantees:**
  1. The task call and the result shape are unchanged.
  2. The engine is recorded. Results from two engines are reused for each other only under a
     measured equivalence for that model and engine pair; otherwise they are distinct.
- **Never:** an engine change requires editing the pipeline.
- **Relates to:** P-2, P-8, P-38.

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

### UX-61 Upgrading senselab itself
- **Actor, setting:** a lab upgrades senselab to a new release with a corpus already processed.
- **Steps:** plans the same pipeline under the new release.
- **Guarantees:**
  1. The plan shows which steps recompute because a task's version changed and which results stay
     reusable.
  2. Results from the old release remain readable and name the release that produced them, or are
     refused with a typed error (UX-10).
- **Never:** an upgrade changes a task's output without that task's version changing; an upgrade
  recomputes, beyond what the user's reuse level requires, tasks whose version did not change, or
  does so without the plan saying how many.
- **Relates to:** P-3, P-19, P-38.

### UX-62 Bringing your own model or task
- **Actor, setting:** a lab with fine-tuned weights on local disk and a new analysis of its own.
- **Steps:** registers the local model and adds the new task or backend, then plans with them.
- **Guarantees:**
  1. Local weights are identified by their content: every file in the model's directory counts
     except version-control metadata; files outside it do not.
  2. The new task is added without forking senselab, and its machine-readable description (UX-33)
     is generated from it.
- **Never:** a local model is identified by its folder name or path; an added task needs a separately
  maintained description.
- **Relates to:** P-1, P-3, P-4, P-26.

### UX-63 Stochastic models
- **Actor, setting:** a user runs a sampling decoder, text-to-speech or voice cloning.
- **Steps:** runs it twice with the same seed, then once with another seed.
- **Guarantees:** the seed is part of the effective parameters and the identity; the same seed
  reproduces the output, or the difference is reported against the stated tolerance for that output (none declared: uncompared); a different
  seed is a different result.
- **Never:** a stochastic result is reused for a different seed; an unseeded run is presented as
  reproducible.
- **Relates to:** P-3, P-38.

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
- **Actor, setting:** one of 5,000 recordings is truncated, another is not audio, another is
  silent.
- **Steps:** runs the pipeline over all of them.
- **Guarantees:**
  1. The other recordings complete.
  2. Each bad input (the truncated file, the non-audio file) is listed with its cause in one line
     and what to do about it, rendered under the reader's audience rule (Terms).
  3. The silent recording is valid data: it yields a result stating that no speech or activity was
     detected, distinguishable from a failure and not listed as a bad input.
  4. Re-running after fixing or removing a bad input computes only that input.
- **Never:** one bad file stops the run; a bad file yields an empty result that looks valid.
- **Relates to:** P-31, P-34, ADR-009.

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
- **Guarantees:** running on the second dataset needs only a dataset configuration.
- **Never:** a task or pipeline branches on a dataset, a path or a field name.
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

### UX-28 Identifiable participants, seen by different audiences
- **Actor, setting:** a clinical study with identifiable voice recordings and participant metadata;
  the data owner has set a visibility policy (Terms).
- **Steps:** the study team runs pipelines and reads logs and reports; a collaborator receives
  results; a team member files a bug upstream with a log.
- **Guarantees:**
  1. Logs, progress reports, error messages and provenance show identifiers exactly as the policy
     allows the person reading them: for example real ids to the study team, pseudonymous ids to
     the collaborator, none in the bug report's log.
  2. Pseudonymous ids are stable per dataset (Terms), so an allowed audience can link results
     back to participants.
  3. The user can see which policy and which audience rule applied to any log, report or export.
  4. Nothing about the data leaves the machine or site (no telemetry, no remote service) unless the
     user configured that service.
- **Never:** identifiers are shown to an audience the policy does not allow; identifiers are
  silently dropped or altered for an audience that is allowed them; anything is less protected
  than the policy states.
- **Relates to:** P-3, ADR-007, ADR-010.

### UX-29 Sensitive results in a shared location
- **Actor, setting:** a group in which some studies are restricted to named members.
- **Steps:** a member runs a restricted study with a shared results location configured.
- **Guarantees:**
  1. Restricted results are written only where the user's configuration allows.
  2. Members with access see the study's identifiers as the policy allows them; members outside
     the study see its records only as the policy allows their audience.
  3. Reuse never makes one study's restricted content readable by someone without access to it.
- **Never:** reuse turns a permission boundary into a leak (for example, by revealing to an
  audience without access whether a given recording was processed); reuse across the boundary
  hides identifiers from members who are allowed them.
- **Relates to:** P-38, ADR-007.

### UX-30 Remote model code
- **Actor, setting:** a user loads a model whose repository ships executable code.
- **Steps:** plans a task with that model.
- **Guarantees:** the plan says the model runs its own code, and it runs only if the user has
  permitted that source.
- **Never:** downloaded model code executes without an explicit permission.
- **Relates to:** P-6, ADR-007.

### UX-64 A participant withdraws
- **Actor, setting:** a participant withdraws consent; the data owner must remove everything
  derived from their recordings.
- **Steps:** asks for all results derived from that participant's inputs, including in shared
  locations, and deletes them.
- **Guarantees:**
  1. Every derived result (direct outputs and anything computed from them) is found and listed
     before deletion.
  2. After deletion none is reused or reported again; dependent results are recomputed without
     them if needed.
  3. The deletion is recorded without re-identifying the participant to any audience the policy does
     not allow.
- **Never:** a derived result of a withdrawn participant survives in a shared location or a
  report; the deletion record names the participant to an audience the policy does not allow.
- **Relates to:** P-3, P-19, ADR-007, ADR-010.

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
- **Guarantees:** the rendering, under the reader's audience rule (Terms), lists every step with
  model identity, effective parameters, environment and placement; a nested pipeline is shown as one node, expandable one level at a
  time.
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
  3. Under the same audience rule, the agent sees the same results and records a human would.
- **Never:** an agent must read source, documentation pages or examples to call something
  correctly; a description differs from what the code accepts.
- **Relates to:** P-1, P-10, P-23, P-31.

### UX-34 An agent cannot exceed what the user allowed
- **Actor, setting:** an agent acting for a user under the user's site, licence and privacy
  settings.
- **Steps:** plans and runs work.
- **Guarantees:** the agent's plans are bound by the same policies (licences, permitted model
  sources, network, resource caps, spend caps, visibility) as the user's own.
- **Never:** a plan submitted by an agent bypasses a policy that would refuse the same plan from
  the user; an agent sees identifiers its principal is not allowed to see.
- **Relates to:** P-1, P-23, ADR-007.

### UX-65 An agent operates a long run
- **Actor, setting:** an agent submits a cluster run for its user and looks after it over days.
- **Steps:** submits, checks progress, sees failures, resumes after an interruption, regenerates
  the report.
- **Guarantees:** progress, failure causes and the resume action are available to the agent in
  machine-readable form, rendered under its principal's audience rule; resuming behaves as UX-8.
- **Never:** operating a run requires the agent to parse human-oriented text output or read
  internal files; the agent's actions are absent from the run's provenance.
- **Relates to:** P-1, P-5, P-31, ADR-008.

## L. Failure modes

Every failure below is reported with its kind (data, model, environment, infrastructure, user
error), the affected items and the next action, in one line per cause. Affected items are named
under the reader's audience rule (Terms, UX-28).

### UX-35 Out of memory
- **Setting:** any; a step exceeds host or GPU memory on some inputs.
- **Guarantees:**
  1. The affected items are reported with the size that failed; other items continue.
  2. Any automatic retry changes only settings that do not change the result, within the same
     model, and the plan says whether such retries are enabled.
- **Never:** an out-of-memory crash loses results already finished; it is reported as a data error;
  a retry switches model or changes the result.
- **Relates to:** P-7, P-34, ADR-009.

### UX-36 Node or worker loss
- **Setting:** cluster or cloud; a node dies, or a worker process is killed.
- **Guarantees:** the takeover delay is shown in the plan or execution settings; in-progress items
  are taken over within it; finished results are kept.
- **Never:** a dead worker blocks others beyond that delay; an item retries forever without being
  reported.
- **Relates to:** P-5, P-34, ADR-003.

### UX-37 Network or storage outage
- **Setting:** the shared filesystem, object storage or model host is unreachable for a time.
- **Guarantees:** retry count and wait are shown in the plan; past them the run reports an
  infrastructure error naming the unreachable location; when service returns, resuming (UX-8)
  completes the work.
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
     naming the backend, the platform and the options (another backend, another setting, an
     override), for the user to choose.
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
- **Never:** they corrupt each other's results; either waits longer than the stated takeover
  delay (UX-36) on the other.
- **Relates to:** P-5, ADR-003.

## M. Provenance

Every provenance view below is rendered under the reader's audience rule (Terms).

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
  7. the run it belongs to, whose report (UX-47) lists the failed and skipped items.
- **Never:** the record names a model revision, parameter or environment other than the one that
  ran; a field is filled with a guess instead of being marked unknown.
- **Relates to:** P-3, P-4, P-6, P-8, P-16, ADR-002, ADR-010.

### UX-44 From a figure back to its records
- **Actor, setting:** a researcher preparing a paper; a figure made by senselab's reporting
  summarises 2,000 results.
- **Steps:** asks which results, inputs, models and parameters the figure was built from.
- **Guarantees:** every number in the figure traces to the results it came from, and each of those
  to its record (UX-43); results excluded from the figure are listed with the reason.
- **Never:** a senselab figure's inputs can be reconstructed only from memory or notes.
- **Relates to:** P-3, P-9, ADR-010.

### UX-45 Export provenance to cite or share
- **Actor, setting:** a user depositing results with a paper or sending them to a collaborator.
- **Steps:** exports the provenance of a selected set of results.
- **Guarantees:**
  1. The export is in a standard, documented provenance form that tools unaware of senselab can
     read.
  2. It contains what is needed to cite and reproduce the results (UX-17).
  3. Exporting applies the visibility rule of the export's audience (for example none for a public
     deposit, pseudonymous ids for a collaborator, real ids for the study team), and the export
     states which rule applied.
- **Never:** the export shows identifiers to an audience the policy does not allow; it drops or
  alters identifiers for an audience that is allowed them.
- **Relates to:** P-3, ADR-007, ADR-010.

### UX-46 Provenance survives moves, reuse and resume
- **Actor, setting:** results copied to another machine, served to a colleague by reuse, or
  completed across an interrupted and resumed run.
- **Steps:** asks for the provenance of those results afterwards.
- **Guarantees:**
  1. The record is complete and the same after a copy to another machine or storage.
  2. A reused result keeps the record of the run that computed it, and the reusing run records
     that it reused it.
  3. After a resume, the run's provenance shows the interruption and both attempts; each result's
     record is that of the attempt that produced it (UX-8).
- **Never:** provenance is silently dropped, truncated or replaced on reuse, resume or move.
- **Relates to:** P-3, P-5, P-10, P-19, ADR-010.

## N. Reports

Every report below is rendered under its reader's audience rule (Terms); a human and an agent under
the same rule get the same content.

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
- **Guarantees:** the user can browse records, filter to failures or to any declared measure, open
  an item's provenance (UX-43), and listen to or view an item's input next to its outputs where the
  visibility policy allows that audience the raw audio or video.
- **Never:** the interactive view shows data that the static report and stored results do not
  support; it plays or shows identifying media to an audience the policy does not allow; it
  requires a server or account the user did not set up.
- **Relates to:** P-1, P-3, P-24, ADR-007, ADR-008.

### UX-49 Reports regenerate without rerunning
- **Actor, setting:** a user, or an AI agent, a month after a run, on any machine with the
  results.
- **Steps:** asks for the report again, or in another form.
- **Guarantees:** the report is rebuilt from stored results and records alone, without computing
  or downloading anything.
- **Never:** regenerating a report reruns tasks or loads models; an agent and a human under the same
  audience rule get different content.
- **Relates to:** P-1, P-3, P-5.

### UX-50 Reports for group and sensitive runs
- **Actor, setting:** a group run on restricted, identifiable data under a visibility policy.
- **Steps:** a member generates a report, reads it, then shares it with a collaborator and later
  publishes a summary.
- **Guarantees:**
  1. A reader sees only the items and inputs they have access to.
  2. Identifiers, audio, video, transcripts and participant metadata appear as the policy allows
     the report's audience; sharing or publishing re-renders the report under the new audience's
     rule rather than passing on the owner's view.
  3. The report states which policy and audience rule it was rendered under.
- **Never:** a report shows identifiers or another member's restricted results to an audience the
  policy does not allow; the owner's own report hides identifiers the owner is allowed to see.
- **Relates to:** P-38, ADR-007, ADR-010.

### UX-66 Comparing two versions or two sites
- **Actor, setting:** a user with results from an old and a new model version (UX-19), or from two
  sites (UX-15).
- **Steps:** asks for a comparison report.
- **Guarantees:** the report shows per-item differences and their distribution, each against the stated tolerance for that output (none declared: uncompared),
  and lists items present in only one set.
- **Never:** two result sets are reported as agreeing when they were not compared item by item.
- **Relates to:** P-3, P-38.

## O. Other modalities, languages and long signals

### UX-51 Pose estimation on a long video
- **Actor, setting:** a researcher with a one-hour session video on a workstation with a GPU.
- **Steps:** runs pose estimation over the whole video.
- **Guarantees:** the same placement, memory, reuse and provenance guarantees as audio (UX-2,
  UX-4, UX-43); each pose carries its frame time on the session timeline.
- **Never:** video work needs a separate pipeline style or bypasses the plan.
- **Relates to:** P-2, P-24, P-26, P-33.

### UX-52 Text embeddings over transcripts
- **Actor, setting:** a user who has transcribed a corpus.
- **Steps:** computes text embeddings over the transcripts as the next step of the same pipeline.
- **Guarantees:** the transcripts feed the embedding step without conversion code; a changed
  transcript recomputes only its own embeddings; each embedding traces to its transcript and the
  recording behind it.
- **Never:** text results lose their link to the input they came from.
- **Relates to:** P-9, P-24, ADR-010.

### UX-53 Audio, video and text on one timeline
- **Actor, setting:** a session recorded with separate audio and video, with a transcript.
- **Steps:** runs a pipeline combining speaker diarization, face tracking and transcript
  alignment.
- **Guarantees:** results from all three stay aligned on one session timeline; any offset or clock
  difference between the streams is recorded and applied, not assumed zero.
- **Never:** a cross-modal result silently mixes times from different clocks.
- **Relates to:** P-24, P-26, ADR-005.

### UX-54 A multilingual corpus
- **Actor, setting:** a corpus mixing several languages, some declared per recording and some not.
- **Steps:** plans transcription over it.
- **Guarantees:** the plan states each model's language support; a recording in an unsupported or
  undeclared language is reported before running, with the options for the user to choose; the
  language used is recorded with each result.
- **Never:** a recording is transcribed under a wrong language assumption without that being
  stated.
- **Relates to:** P-3, P-12, P-31.

### UX-55 Chunk boundaries in long recordings
- **Actor, setting:** a three-hour recording processed in pieces, in any setting.
- **Steps:** transcribes and diarizes it.
- **Guarantees:** outputs have continuous timestamps over the whole recording; nothing is lost or
  duplicated at piece boundaries; where an unpieced run can be run, the result equals it or is reported against
  the stated tolerance for that output (none declared: uncompared).
- **Never:** a word, segment or speaker turn is dropped or doubled at a boundary; timestamps restart
  per piece.
- **Relates to:** P-5, P-33, ADR-005.

---

## Change log

- 2026-11-08: proposed, 42 scenarios in groups A–L.
- 2026-11-08: groups M (provenance) and N (reports) added at the owner's request: "add in provenance
  related information and report (potentially interactive) generation for users".
- 2026-11-08: identifier visibility is per audience under a data owner's policy, not a blanket rule
  (owner: "identifier specification may depend on use-case, that shouldn't be a blanket rule but
  dependent on who sees the identifiers").
- 2026-11-08: revised after review. Terms gains identity, reuse levels, no silent fallback, the
  identifier list, audience (agents included), the default policy and pseudonym scope;
  contradictions on resume, fallbacks, reuse and hardware in identity resolved; scenarios made
  decidable; UX-51..UX-66 added (66 scenarios, groups A–O); how-to-use gains the area map, the
  quoted-location rule, unchecked scenarios and owner-only amendment.
- 2026-11-08: final review. Reuse level three is "any context, labelled"; cross-context reuse in
  UX-3, UX-8 and UX-61 is qualified by the reuse level; identity includes the engine where it can
  change outputs; one tolerance phrase; data owner defined; plans and exports render per audience;
  scenarios govern over ADRs (ADR-007 amended the same day).
