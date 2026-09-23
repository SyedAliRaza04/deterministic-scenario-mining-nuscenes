# References

Every external work this project stands on, annotated with what it is used for. This is the
working bibliography; the formatted reference list of the dissertation is generated from it.

**Rules for this file.**
- One entry per work, with **what it is used for** and **where in the repository it appears**.
- Nothing goes in unless it has been checked against arXiv, a DOI or the publisher. A
  fabricated or half-remembered citation is worse than no citation.
- If a design decision (D-nnn, logged in `PROCESS.md`) was influenced by a paper, the paper is
  named in that decision.
- A warning marker flags a work that **changes a claim made elsewhere in the project**. Those
  are listed first, in section 0.

Section 11 states what remains defensible as a contribution after everything below, and
section 12bis records the adversarial novelty audit that tested those claims against the
literature rather than after them.

---

## 0. The three that change what we can claim

These were found on 2026-09-01. Each one weakens or reframes a novelty claim in PROCESS.md §2.

**⚠️ [R-01] de Gelder, E., Manders, J., Grappiolo, C., Paardekooper, J.-P., Op den Camp, O.,
De Schutter, B. (2020). *Real-World Scenario Mining for the Assessment of Automated Vehicles.*
IEEE ITSC 2020. arXiv:2006.00483.**

The direct methodological ancestor of Stage C. Their pipeline is *exactly* our two-step shape:
**automatically label sensor data with tags, then mine scenarios as combinations of tags.**

*Consequence for RQ1.* RQ1 currently reads "Can scenario ground truth be derived
automatically, without human annotation?" — the field answered that yes, six years ago.
RQ1 must be **restated** so the contribution is the part that is actually ours: using
deterministic scenario mining as *auditable reference labels to benchmark a VLM
auto-labeller*, and quantifying the error of the annotation-presence baseline. See §11.

**⚠️ [R-02] Karnchanachari, N., Geromichalos, D., Tan, K. S., Li, N., Eriksen, C., Yaghoubi, S.,
Mehdipour, N., Bernasconi, G., Fong, W. K., Guo, Y., Caesar, H. (2024). *Towards learning-based
planning: The nuPlan benchmark for real-world autonomous driving.* IEEE ICRA 2024.
arXiv:2403.04133.**

nuPlan ships **1,282 h of driving automatically tagged into 73 scenario types** by SQL queries
over atomic primitives computed from the logs, with 100 examples per type manually QA'd.

*Corrected 2026-09-20 (F-081), and this correction has teeth.* The count is **73**, stated
twice in the paper, not "~75". Only **14 types are named and parameterised in the paper**
(Table XII, the planning-challenge set); Fig. 11 is a distribution plot with no name list, so
the other 59 names exist only in the devkit. **The §12 to-do "write the taxonomy alignment
table against nuPlan's ~75 types" is therefore not executable from this paper** — an alignment
table presented as sourced from the ICRA paper would fabricate its provenance. Worse for the
QA claim: the paper publishes **no error rate whatsoever** from its manual QA (no per-type
precision, no aggregate false-positive rate, no discard count), declares recall out of scope,
and states its acceptance gate as *"if the false positive rate is below 90%, we either refine
the query ... or discard the scenario"*, whose printed direction is backwards and cannot be
recovered from the text. This is the strongest available evidence for the reference-quality
argument: at production scale, by the nuScenes authors themselves, the quality of the mined
reference is **asserted rather than measured**.
This is a production-scale instance of what C1–C7 build, from the same authors as nuScenes.

*Two consequences.* (a) Our scenario taxonomy should be **compared against nuPlan's** rather
than invented in isolation — an alignment table is cheap and makes the 37 tags defensible.
(b) nuPlan is the concrete answer to **L-001**: it explicitly contains merges and lane changes,
which nuScenes does not (F-042). Take this to Yoann alongside `c7_coverage.png`.

**⚠️ [R-03] Elrofai, H., Worm, D., Op den Camp, O. (2016). *Scenario Identification for
Validation of Automated Driving Functions.* In: Advanced Microsystems for Automotive
Applications 2016, Springer, pp. 153–163.**

Earlier still: automatic extraction and classification of scenarios from real traffic data.
Establishes that "derive scenario labels deterministically from logs" is a decade-old idea in
the AV-homologation community, not a novelty. Cite in the related-work section so the examiner
does not raise it first.

---

## 1. Datasets we use — attribution is obligatory, not optional

**[R-04] Caesar, H., Bankiti, V., Lang, A. H., Vora, S., Liong, V. E., Xu, Q., Krishnan, A.,
Pan, Y., Baldan, G., Beijbom, O. (2020). *nuScenes: A multimodal dataset for autonomous
driving.* CVPR 2020. arXiv:1903.11027.**
The primary dataset. 1,000 scenes × 20 s, 6 cameras + 5 radar + 1 lidar, 360° coverage,
3D boxes for 23 classes with 8 attributes. Used everywhere in Stage C.
Cite in PROCESS.md §4.1 and beside every number derived from trainval.
*Licence:* CC BY-NC-SA 4.0 — non-commercial. Must be stated in the report.

**[R-05] nuScenes map expansion v1.3** (part of the nuScenes release, R-04).
HD semantic maps for 4 locations. The `road_segment`, `lane`, `ped_crossing`, `stop_line`,
`carpark_area`, `walkway`, `drivable_area` layers plus `connectivity` and `arcline_path_3`
are the entire basis of C3 (D-017 to D-021) and of the roundabout detector (F-026 to F-028).
The working rule to check whether the dataset already answers a question, before inventing
a proxy for it, exists because we invented one for something this shipped.

**[R-06] Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V., Darrell, T.
(2020). *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning.*
CVPR 2020 (oral). arXiv:1805.04687.**
Stage H / RQ4. The only source of traffic-light **state** available to us, via the
`trafficLightColor` attribute. Named in the SystemX slides, so sanctioned by the brief (D-007).
*Not yet downloaded* — Stage H1.

**[R-07] nuScenes-lidarseg** (Caesar et al., part of R-04). Acquired, currently unused.
Cite only if it ends up enriching ground truth.

---

## 2. The model and the runtime

**[R-08] Qwen Team, Alibaba Group (2025). *Qwen2.5-VL Technical Report.* arXiv:2502.13923.**
The VLM under test in Stage E. Cite for: the 3B/7B/72B family, native dynamic-resolution ViT,
window attention, and the **multi-image and video input** capability that Steps E7 and E9
depend on — E9's whole premise is that the old code claimed five frames while sending one.
*Licence:* Apache 2.0 for 3B/7B. State it.

**[R-34] Qwen Team, Alibaba Group (2025). *Qwen3-VL.* Model release, October 2025;
technical report arXiv:2511.21631. Sizes include 2B / 4B / 8B / 32B.**
The candidate for the E10 model-sensitivity slice (D-048). Reported to beat previous-
generation 7B models at 4B. **Not adopted as the primary**: the T4 is Turing (compute 7.5,
no bfloat16, no Flash Attention 2) and Turing support in Qwen3-VL backends is an open
issue, so it is unverified on the hardware this thesis actually has, while Qwen2.5-VL-7B
is verified (F-059). Cite when reporting E10, and state plainly if the slice is not run.

**[R-09] Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. (2023). *QLoRA: Efficient
Finetuning of Quantized LLMs.* NeurIPS 2023. arXiv:2305.14314.**
Cite for the 4-bit NF4 quantisation used by `bitsandbytes` on the Colab T4 (Step E4).
Needed because L-005 claims conclusions are about *accessible* models — the quantisation is
part of what "accessible" means, and 4-bit is not free of accuracy cost.

**[R-10] nuscenes-devkit** (Motional, Apache 2.0). `BoxVisibility`, `box_velocity`,
`get_sample_data`, `NuScenesMap`, `create_splits_scenes`. D-002 and D-025 rest on its frustum
filter; F-033 is about correcting its **global-frame** `box_velocity` output.

**[R-11] Shapely / GEOS** (BSD-3). The STRtree index behind D-018's 3,785× speedup.
F-019 (predicate direction) and R15 are about its API semantics — worth a footnote, since the
bug was in *our* reading of the library, not the library.

---

## 3. Scenario-based validation — the framing the brief actually lives in

The SystemX brief is a homologation-adjacent document (EU AI Act, Catena-X, MOSAR). These
give the vocabulary that makes the thesis legible to that audience.

**[R-12] Menzel, T., Bagschik, G., Maurer, M. (2018). *Scenarios for Development, Test and
Validation of Automated Vehicles.* IEEE IV 2018, pp. 1821–1827. DOI 10.1109/IVS.2018.8500406.**
The **functional / logical / concrete** abstraction levels. Our `label_schema.json` sits at the
functional–logical boundary; the Stage G scenario description is a logical scenario. Saying so
in these terms is a one-paragraph upgrade to the report's credibility.

**[R-13] Scholtes, M., Westhofen, L., Turner, L. R., Lotto, K., Schuldes, M., Weber, H.,
Wagener, N., Neurohr, C., Bollmann, M. H., Körtke, F., Hiller, J., Hoss, M., Bock, J.,
Eckstein, L. (2021). *6-Layer Model for a Structured Description and Categorization of Urban
Traffic and Environment.* IEEE Access 9, pp. 59131–59147. DOI 10.1109/ACCESS.2021.3072739.**
Layers: road network and traffic guidance objects / roadside structures / temporary
modifications / moving objects / environment / digital information.

*Corrected 2026-09-20 (F-081).* This entry previously read "PEGASUS six-layer model (German
PEGASUS project; see R-12 and the survey R-14 for the canonical statement)" — no author, no
year, no venue, and **both cross-references were false**: R-12 is about abstraction levels and
does not state the layer model, and R-14 is a standard, not a survey, and does not state it
either. The model's lineage is Schuldt (2013, 2017, four layers) → Bagschik, Menzel, Maurer
(2018, five layers) → Bock et al. (2018, adds digital information) → Scholtes et al. (2021,
the consolidation the field cites). Scholtes et al. exist *because* entities had migrated
between layers across that history until the model was ambiguous.
Our four label families map onto layers 1–2 (map context), 4 (objects, interactions) and the
ego-motion channel. Use it to argue **coverage**, and to name honestly what we do *not* cover:
layer 5 (weather) and layer 6 (V2X) are absent from our schema entirely.

**[R-14] ISO 34502:2022 — *Road vehicles — Test scenarios for automated driving systems —
Scenario based safety evaluation framework.*** Published 2022-11-02.
Cite once, in the introduction, to place the work in the standards landscape. We do not
implement it; claiming otherwise would be overreach.

**[R-15] ASAM OpenSCENARIO (XML 1.x / DSL 2.x) and ASAM OpenDRIVE.**
`https://www.asam.net/standards/detail/openscenario-dsl/`
The target format for the optional Step G5 export. OpenDRIVE = static road, OpenSCENARIO =
dynamic content. This is the "European homologation workflows speak OpenX" claim in PLAN.md —
it needs this citation or it is an assertion.

---

## 4. VLMs for driving — RQ2 context

**[R-16] Zhou, X., et al. (2023/2024). *Vision Language Models in Autonomous Driving: A Survey
and Outlook.* arXiv:2310.14414.**
The one survey to cite for "where VLMs sit in AD". Covers perception, planning, control,
end-to-end and **data generation** — the last is our category. Use for the related-work
skeleton rather than reading 40 papers.

**[R-17] Sima, C., Renz, K., Chitta, K., Chen, L., Zhang, H., Xie, C., Beißwenger, J., Luo, P.,
Geiger, A., Li, H. (2024). *DriveLM: Driving with Graph Visual Question Answering.*
ECCV 2024 (oral). arXiv:2312.14150.**
nuScenes-based, graph-structured perception→prediction→planning QA. The external benchmark
named in PLAN F5. Cite as (a) the closest published VLM-on-nuScenes evaluation and (b) evidence
that **multi-step decomposition beats single-round VQA** — which is the hypothesis behind our
agentic yes/no prompt in E8.

**[R-18] Qian, T., Chen, J., Zhuo, L., Jiao, Y., Jiang, Y.-G. (2024). *NuScenes-QA: A
Multi-modal Visual Question Answering Benchmark for Autonomous Driving Scenario.* AAAI 2024,
38(5), pp. 4542–4550. arXiv:2305.14836.**
34K scenes / 460K QA pairs, **generated programmatically from the 3D annotations via scene
graphs and question templates**. Methodologically our nearest neighbour: derive reference
answers from annotations rather than from humans. Cite in RQ1's related work, and note the
difference — they template questions, we derive *scenario-level* labels including ego motion,
which their scene graphs do not contain.

**[R-19] Vision-language models for auto-labelling — the 2025 wave.** Representative:
*Structured Labeling Enables Faster Vision-Language Models for End-to-End Autonomous Driving*
(arXiv:2506.05442); *AutoVDC: Automated Vision Data Cleaning Using Vision-Language Models*
(arXiv:2507.12414); *Can Vision-Language Models Replace Human Annotators?* (arXiv:2410.09416).
Cite collectively for the motivating claim in the brief — that VLM auto-labelling is being
taken seriously as a replacement for human annotation — and for the standard caveat that most
production pipelines still keep a human verification step.
*Status: 2410.09416 (`lu2024`) and 2507.12414 (`vasa2025`) VERIFIED 2026-09-22 against arXiv and
cited in the literature chapter, with Qi et al. CVPR 2021 (`qi2021`, Crossref DOI) for the term.
2506.05442 read and not cited: it is a structured benchmark (NuScenes-S), not auto-labelling.*

---

## 5. BEV + language — RQ3a, and a warning for Stage D

**[R-20] Choudhary, T., Dewangan, V., Chandhok, S., Priyadarshan, S., Jain, A., Singh, A. K.,
Srivastava, S., Jatavallabhula, K. M., Krishna, K. M. (2024). *Talk2BEV: Language-enhanced
Bird's-eye View Maps for Autonomous Driving.* IEEE ICRA 2024. arXiv:2310.02251.**
The reference point for RQ3a. Note carefully: Talk2BEV builds a **language-enhanced** BEV — BEV
objects are converted into text/structured descriptions consumed by an LLM — rather than
handing a raster BEV picture to a VLM. Our Stage D plan does the latter. That is a real
difference and it must be argued, not glossed.

**[R-21] Ding, X., Han, J., Xu, H., Liang, X., Zhang, W., Li, X. (2024). *Holistic Autonomous
Driving Understanding by Bird's-Eye-View Injected Multi-Modal Large Models* (NuInstruct /
BEV-InMLLM). arXiv:2401.00988.**
91K multi-view video-QA pairs over 17 subtasks on nuScenes; BEV features injected into an MLLM
give **~9% improvement**. Evidence that BEV information helps — but via *learned feature
injection*, again not via a rendered picture. Cite as support for the RQ3a hypothesis and as
the honest statement of what has actually been shown.

**⚠️ [R-22] The abstract-image weakness. Representative:** *Multimodal Self-Instruct: Synthetic
Abstract Image and Visual Reasoning Instruction Using Language Model* (arXiv:2407.07053);
*MATHGLANCE: Multimodal Large Language Models Do Not Know Where to Look in Mathematical
Diagrams* (arXiv:2503.20745).
The finding that matters for Stage D: current multimodal LLMs perform **sub-optimally on
abstract, non-photographic images** — charts, diagrams, maps — failing at tasks as simple as
route-planning on a map, because the training distribution is natural photographs.

*Consequence.* A hand-rolled 512×512 BEV raster is exactly such an abstract image. **The
plausible Stage D outcome is that BEV underperforms the front camera**, and if we do not say so
in advance it will read as a failed experiment rather than a predicted one. It also argues for
the hybrid arm in the Stage D plan (rendered BEV *plus* a text legend / symbolic description),
which is the design R-20 and R-21 both converged on.
*Status: found by abstract. Read both before citing.*

---

## 6. Temporal understanding — RQ3b and RQ5

**⚠️ [R-23] Cannons, K., et al. (2025/2026). *From Segments to Scenes: Temporal Understanding
for Agentic Autonomous Driving via Vision-Language Models* (TAD benchmark). arXiv:2512.05277.**
~6,000 QA pairs over 7 temporal tasks, built on **nuScenes videos with human-annotated ego and
non-ego action labels**, evaluating 9 generalist and AD-specialist models. Current SoTA sits
substantially below human accuracy; their training-free Scene-CoT and TCogMap add up to 17.72%.

*Two consequences.* (a) It is the closest published work to RQ3b and Stage G — read it properly
before writing either. (b) If their **human ego-action annotations are released**, they are an
independent human reference for our C2 manoeuvre labels — which would upgrade R9's
cross-validation from "two derivations of ours agree" to "ours agrees with humans". Worth an
hour to check.

**[R-24] ScVLM: Enhancing Vision-Language Model for Safety-Critical Event Understanding**
(arXiv:2410.00982). Temporal localisation of event start/end from driving video — the task
shape of Stage G1/G2. *Status: abstract only.*

---

## 7. Structured output — Step E2

**[R-25] Geng, S., et al. (2025). *JSONSchemaBench: A Rigorous Benchmark of Structured Outputs
for Language Models.* arXiv:2501.10868.** (Also as *Generating Structured Outputs from Language
Models: Benchmark and Studies*.)
10K real-world schemas; compares Guidance, Outlines, XGrammar, llama.cpp grammars on efficiency,
coverage and output quality.

*Consequence for E2.* PLAN E2 currently fixes a **regex** parser. Constrained decoding makes the
parse-failure rate structurally zero instead of merely measured. But RQ2b asks *how often the
model fails to emit parseable output* — a real finding. So: **report the free-generation parse
failure rate (RQ2b), then run the scored conditions under constrained decoding**, and cite this
work for why both numbers are needed.

**⚠️ [R-26] Tam, Z. R., et al. (2024). *Let Me Speak Freely? A Study on the Impact of Format
Restrictions on Performance of Large Language Models.* arXiv:2408.02442.**
Format restriction can *degrade* reasoning quality. So constrained decoding is not a free win,
and if we use it we owe the report a sentence acknowledging the trade-off. Directly relevant to
the E8 structured-JSON vs. chain-of-thought ablation.
*Status: abstract only.*

---

## 8. Regulatory framing

**[R-27] Regulation (EU) 2024/1689 (the AI Act), Article 10 — *Data and Data Governance*.**
`https://artificialintelligenceact.eu/article/10/`
The specific hook: data governance practices must cover **annotation, labelling, cleaning,
updating, enrichment and aggregation**, and datasets must be documented as to sources,
collection and preprocessing.

*This is the article D-001 is actually about* — and it should be cited by number, not gestured
at as "the EU AI Act". `label_schema.json` (37 tags, each with derivation, parameters, decision
ID, prevalence and scope) plus the provenance columns on every table are a concrete Article 10
artefact. Currently PROCESS.md asserts this framing eight times with zero citations.
*Caveat to keep us honest:* Article 10 binds high-risk **system** providers. We are producing a
labelling method, not placing a high-risk system on the market. Frame as "the traceability
Article 10 would require", not "compliant with Article 10".

---

## 9. Datasets we do *not* use — but which L-001 obliges us to name

C7 established that **highway merge and parallel parking do not occur in nuScenes** (F-042).
A reader will immediately ask "so which dataset has them?". Answering with three named
alternatives converts a limitation into a scoping decision.

**[R-28] nuPlan** — see **R-02**. Merges and lane changes present, same devkit lineage,
1,282 h. **The default answer.** Free, non-commercial licence, and the metadata-only trick that
made Stage C possible on a laptop should transfer.

**[R-29] Krajewski, R., Bock, J., Kloeker, L., Eckstein, L. (2018). *The highD Dataset: A Drone
Dataset of Naturalistic Vehicle Trajectories on German Highways...* IEEE ITSC 2018.**
**110,000 vehicles, 45,000 km, 16.5 h of recording** from six locations. Drone/BEV by
construction — no ego camera, so unusable for a *front-camera* VLM arm, but a natural fit for a
BEV-only arm.

*Corrected 2026-09-20 (F-081).* This entry previously read "110,500 trajectories, 44,500 km,
**147 driving-hours**", which are the **vendor website's** figures presented as if they came
from the ITSC paper. The paper says 110,000 vehicles and 45,000 km. "147 driving-hours" is not
a recording duration at all: recording is 16.5 h, and the cumulative travel time summed over
all tracked vehicles is given in the paper's own Table I as 447 h. Do not use the 147 figure.

**[R-30] Moers, T., Vater, L., Krajewski, R., Bock, J., Zlocki, A., Eckstein, L. (2022). *The
exiD Dataset: A Real-World Trajectory Dataset of Highly Interactive Highway Scenarios in
Germany.* IEEE IV 2022.**
69,172 road users, 16 h, recorded specifically **at motorway entries and exits** — i.e. the
merge scenario the brief names and nuScenes lacks. German data, which also serves the brief's
European data-sovereignty framing better than Boston and Singapore do.
*Status: bibliographic details from a secondary survey. Verify against the publisher before
citing.*

---

## 10. Methods and algorithms

**[R-31] Savitzky, A., Golay, M. J. E. (1964). *Smoothing and Differentiation of Data by
Simplified Least Squares Procedures.* Analytical Chemistry 36(8), 1627–1639.**
The C1 smoother (`SMOOTH_S = 0.20`). Cite it, and cite it *next to D-011* — the reason we
resample to a uniform 50 Hz grid first is that Savitzky-Golay assumes uniform spacing. That is
a methodological point, not a detail.

**[R-32] Guttman, A. (1984). *R-Trees: A Dynamic Index Structure for Spatial Searching.*
ACM SIGMOD.** The index behind D-018. Optional; cite only if the report explains the speedup.

**[R-33] Li, Z., et al. (2022). *BEVFormer: Learning Bird's-Eye-View Representation from
Multi-Camera Images via Spatiotemporal Transformers.* ECCV 2022. arXiv:2203.17270.**
Only needed if the report discusses *learned* BEV. Our Stage D BEV is **rendered from ground
truth**, not perceived — a distinction worth one explicit sentence, because a reader from the
AD community will assume BEVFormer-style perception by default.
*Status: verify before citing — included from background knowledge, not from this search pass.*

---

## 11. What is actually novel, after all of the above

Written down now so the contribution claim survives contact with a supervisor.

**Not novel** — and must be presented as replication-with-attribution:
- deterministic derivation of scenario tags from driving logs (R-01, R-02, R-03);
- programmatic generation of reference labels from nuScenes annotations (R-18);
- the claim that BEV information helps a language model reason about driving (R-20, R-21).

**Defensibly ours:**

> ⚠️ **Items 1 and 2 below are SUPERSEDED as of 2026-09-19 (F-081).** They are kept, struck,
> because the audit trail is the point. The novelty audit found the claims pre-empted; the
> corrected wording follows them. Do not quote the struck text to anyone.

1. ~~**The auto-labeller is the system under test, and the scenario mining is its instrument.**
   R-01/R-02 mine scenarios to *assemble test suites*. We mine them to *score a VLM*. Nobody in
   the cited set uses deterministic scenario mining as the reference against which a VLM
   auto-labeller is measured per tag family.~~
   **Superseded:** R-37 (EgoDyn-Bench, ECCV 2026) does exactly this for ego motion, with a
   deterministic kinematic oracle, 20+ models and a camera-versus-trajectory-text comparison.
   R-38 (RefAV) scores VLMs on scenario mining over Argoverse 2. **Corrected claim:** the
   instrument is applied to a *four-family scenario vocabulary* (ego manoeuvre, map context,
   visible objects, interactions) scored **per family**, where the published instances cover
   ego motion alone or natural-language queries alone. Cite R-37 and R-38 in the sentence that
   states it, and verify STSBench (§13) before the claim is made at all.
2. ~~**Quantified cost of a bad reference.** F-029 (3.51× object over-count), F-033
   (`following_vehicle` at **12.2%** precision, 56.4% of hits pointing backwards) and Step F3's
   old-GT/new-GT rescoring turn "your ground truth was wrong" into a measured number. This is
   the finding a methods reviewer will care about most, and we have not seen it published.~~
   **Superseded:** the reference-swap-at-fixed-outputs experiment is a five-year-old programme
   with a canonical paper (R-35, NeurIPS 2021) and a VLM instance on binary presence probes
   (R-36, 2025). **Corrected claim, three parts, all narrower and all still true:**
   (a) the axis is **definitional, not error-corrective** — three defensible derivations of the
   same tag differing in *scope*, with no human relabelling anywhere in the loop, where every
   published instance corrects *errors* against a human gold set;
   (b) the reference is **machine-derived and auditable**, so each defect is attributable to a
   named predicate and a decision ID rather than to annotator sloppiness;
   (c) the **sign differs per tag** on identical outputs: `following_vehicle` flatters by 0.020
   while its siblings penalise.
   Frame as an instantiation plus one new measured property, positioned downstream of R-35 and
   R-36, and use the diagnostic literature's own term, **imperfect reference standard**.
3. **A negative dataset-suitability result, measured over all 850 scenes.** Two of the brief's
   three named manoeuvres do not exist in the dataset everyone benchmarks on (F-042). Reported
   with evidence — max ego speed 66.7 km/h, 3 reverses totalling under 5 m.
4. **An input-representation ablation held at fixed ground truth.** front / BEV / both × temporal,
   one label set, per family. R-21 injects learned BEV features; R-20 converts BEV to text. The
   controlled comparison of *rendered* BEV against the camera, on identical labels, is open — and
   R-22 says the expected answer may well be "it does not help", which is still a result.
5. **Auditability as an artefact rather than a claim.** 37 tags, each with its derivation,
   parameters, decision ID and scope, plus provenance columns on every row (R-27).

**Honest weakness to state before someone else does:** the reference labels are themselves
heuristics with researcher-chosen thresholds (L-002, L-011, L-022) and are validated against a
weak reference (L-007). R-23's human ego-action annotations, if released, are the cheapest
available fix.

---

## 12. Open bibliographic work

- [ ] Read R-01 and R-02 in full; write the taxonomy alignment table against nuPlan's ~75 types.
- [ ] Restate RQ1 in PROCESS.md §2 per §0 above, and add a Related Work section keyed to this file.
- [ ] Check whether R-23 releases its human ego/non-ego action labels for nuScenes.
  Still open 2026-09-22. The discussion's Future Research proposes collecting labels
  *of the kind* R-23 collected and does not claim theirs are available, so the thesis
  is not blocked on this; it would only become a defect if that paragraph asserted
  the labels can be obtained.
- [ ] Read R-22's two papers before committing to the Stage D BEV design.
- [ ] Verify R-30 and R-33 against publishers; drop or correct if they do not check out.
- [ ] Add licence statements: nuScenes CC BY-NC-SA 4.0, BDD100K, Qwen2.5-VL Apache 2.0.
- [x] ~~Decide citation style with Yoann~~ — **APA**, decided 2026-09-18. The ECE dissertation
  guide permits APA, IEEE or Harvard but requires the list **alphabetical by surname**, which
  IEEE numeric does not give. The bibliography is maintained in BibTeX and rendered with `apacite`.

---

## 12bis. Added by the novelty audit (F-081)

Each of the four below was verified directly against its arXiv abstract page, never against a
second-hand summary. ⚠️ on all four: each one changes a claim we made.

**⚠️ [R-35] Northcutt, C. G., Athalye, A., Mueller, J. (2021). *Pervasive Label Errors in Test
Sets Destabilize Machine Learning Benchmarks.* NeurIPS 2021, Datasets and Benchmarks Track.
arXiv:2103.14749 (v1 26 Mar 2021, v4 7 Nov 2021).**
Label errors across 10 vision, language and audio test sets: **≥3.3% mean, 6% of the ImageNet
validation set**, 51% of algorithmically flagged candidates confirmed erroneous by crowd
workers. Corrected references **reverse rankings**: ResNet-18 overtakes ResNet-50 at +6%
mislabelled prevalence, VGG-11 overtakes VGG-19 at +5% on CIFAR-10.
*Use.* The ancestor of Step F3. Cite it in the sentence that introduces the reference-cost
result, not in a footnote. It is what kills §11 item 2's old wording.

**⚠️ [R-36] Neuhaus, Y., Hein, M. (2025). *RePOPE: Impact of Annotation Errors on the POPE
Benchmark.* arXiv:2504.15707 (22 Apr 2025). Code and data: github.com/YanNeu/RePOPE.**
Re-annotates the MSCOCO images behind the POPE object-hallucination benchmark, identifies **an
imbalance in annotation errors across subsets**, re-evaluates multiple models on the revised
labels and observes **notable shifts in model rankings**.
*Use.* The closest published twin of F3: a VLM, binary presence probes, a swapped reference.
Its error *asymmetry* is F-067(b)'s "a bad reference can flatter", published first. Read the
full text before drafting §2.4; the abstract alone does not confirm the F1 mechanics, so the
per-subset numbers quoted anywhere must come from the paper itself.

**⚠️ [R-37] Schäfer, F. R., Gao, Y., Wang, D., Stauner, T., Günnemann, S., Piccinini, M.,
Schmidt, S., Betz, J. (2026). *EgoDyn-Bench: Evaluating Ego-Motion Understanding in
Vision-Centric Foundation Models for Autonomous Driving.* ECCV 2026. arXiv:2604.22851
(submitted 22 Apr 2026, revised 7 Jul 2026).**
Maps continuous vehicle kinematics to discrete motion concepts **via a deterministic oracle**,
then audits 20+ models (closed MLLMs, open VLMs at several scales, VLAs). Finds a *perception
bottleneck*: models hold the physical concepts but fail to align them with visual observations.
**Explicit trajectory encodings substantially restore physical consistency**, showing ego-motion
logic comes almost entirely from the language modality while vision contributes negligible
temporal signal.
*Use.* This is our instrument and our RQ3a/RQ3b result for the ego_maneuver family, published.
It is also **corroboration**: our BEV-text arm and our motion tags at 0.000 are the same
phenomenon measured independently. Cite it as the work we replicate and extend, and read it in
full before writing §2.6 and §2.7.

**⚠️ [R-38] Davidson, C., Ramanan, D., Peri, N. (2025). *RefAV: Towards Planning-Centric
Scenario Mining.* arXiv:2505.20981 (27 May 2025, rev. 27 Dec 2025).**
10,000 natural-language scenario queries over 1,000 Argoverse 2 sensor logs; referential
multi-object trackers as baselines; finds that **naively repurposing off-the-shelf VLMs yields
poor performance**.
*Use.* Same headline direction as our 0.389 macro F1, on a sibling dataset. Also the answer to
"which dataset instead" that §9 currently misses.
*Caution.* The audit reported a CVPR 2026 Oral venue; the arXiv page does not state it. Verify
before putting a venue in the .bib.

---

## 13. Verification queue: named but not yet checked, and therefore not citable

These were named during the novelty audit and reported as checked against publisher records,
but nothing in this file may rest on second-hand verification (the rule at the top of this
file). Each needs one direct check before it can be cited.

| work | why it matters | priority |
|---|---|---|
| ~~**STSBench**~~ VERIFIED 2026-09-22, arXiv:2506.06218, in refs.bib as `fruhwirth2025`: mines nuScenes scenarios from GT annotations, 971 human-verified MCQs over 43 scenarios | reported to auto-mine scenarios on nuScenes and use them to benchmark multimodal models. If accurate, it is the closest work to §11 item 1 and must be cited there | **highest** |
| **DriveBench**, arXiv:2501.04003 | 12 VLMs, 19,200 frames, clean/corrupted/**text-only** inputs; models answering from language priors. Bears directly on our majority-baseline nulls | high |
| **Justo Miro et al.**, arXiv:2601.14038, WACV 2026 | annotation-error impact on AV benchmarks reported to exceed SOTA margins. If they re-score detectors against corrected boxes, the AV instance of F3 exists | high |
| MMLU-Redux, arXiv:2406.04127; Vendrow et al., arXiv:2502.03461 | further reference-correction instances in LLM evaluation | medium |
| Li, Y. et al. (2023). *Evaluating Object Hallucination in LVLMs* (POPE). EMNLP 2023 | names our always-yes finding: yes-bias on binary presence probes | **highest** |
| Wei, J. et al. (2022) arXiv:2201.11903; Kojima, T. et al. (2022) arXiv:2205.11916 | chain of thought. The largest effect in the thesis currently has **no citation at all** | **highest** |
| Ulbrich, S., Menzel, T., Reschka, A., Schuldt, F., Maurer, M. (2015). ITSC 2015, pp. 982–988 | the definition of scene / situation / scenario. We cite its child (R-12) without it, and never define our central term | **highest** |
| Scholtes, M. et al. (2021). 6-layer model | the missing primary source for R-13 | high |
| Qi, C. R. et al. (2021). *Offboard 3D Object Detection from Point Cloud Sequences.* CVPR 2021 | "auto-labelling" already means offboard auto-labelling in AV. Title-level ambiguity | high |
| Lipton, Z. C., Elkan, C., Naryanaswamy, B. (2014). ECML PKDD | why always-yes beats a classifier on a high-prevalence tag under F1. Explains F-068 rather than leaving it a surprise | high |
| Dietterich (1998); Wilson (1927); Brown, Cai, DasGupta (2001); Opitz & Burst (2019); Field & Welsh (2007); Horvitz & Thompson (1952) | the statistics the thesis performs and does not cite: paired comparison, the interval estimator, which macro F1, clustered resampling, unequal selection probabilities | high, methodology chapter |
| Gebru, T. et al. (2021). *Datasheets for Datasets.* CACM | `label_schema.json` is a datasheet. Contribution 5 has prior art | medium |
| Ratner, A. et al. (2017). *Snorkel.* PVLDB 11(3) | 37 threshold rules over an unlabelled corpus are labelling functions. Names our paradigm and addresses L-002/L-011/L-022 | medium |
| Riedmaier, S. et al. (2020). *Survey on Scenario-Based Safety Assessment.* IEEE Access | the survey §2.2 should open on | medium |
| Argoverse 2 (Wilson, B. et al., NeurIPS 2021 D&B); ISO 21448 (SOTIF) | the dataset §9 omits; the standard that explains why scenario-based validation exists | medium |
