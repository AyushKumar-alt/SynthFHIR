# SynthFHIR — Engineering Documentation

This folder is the complete technical handbook for the SynthFHIR project.
It is written for readers with zero prior knowledge of healthcare, FHIR, or
synthetic data generation. Every concept is explained before it is used.
Every decision is justified.

---

## How to read this documentation

If you are completely new to the project, read in order from `00` to `10`.

If you have a specific question, use the table below.

| Document | Covers |
|---|---|
| [00_Project_Overview.md](00_Project_Overview.md) | What the project is, why it exists, what it will produce |
| [01_Background.md](01_Background.md) | FHIR, HL7, EHR, clinical concepts — everything you need to understand the data |
| [02_Dataset.md](02_Dataset.md) | Synthea, every table, every column, relationships, cardinality |
| [03_Preprocessing.md](03_Preprocessing.md) | Every preprocessing decision explained with full justification |
| [04_Model_Architecture.md](04_Model_Architecture.md) | GAN, VAE, Copula, CTGAN, PAR — how they work and why we selected them |
| [05_Training_Pipeline.md](05_Training_Pipeline.md) | Complete training lifecycle: checkpointing, GPU, fault tolerance, ZIP backup |
| [06_Evaluation.md](06_Evaluation.md) | Utility and privacy evaluation methodology (Phase 5) |
| [07_FHIR_Reconstruction.md](07_FHIR_Reconstruction.md) | Re-converting synthetic tables back to FHIR format (Phase 6) |
| [08_Dashboard.md](08_Dashboard.md) | Interactive visualisation dashboard (Phase 7) |
| [09_Engineering_Log.md](09_Engineering_Log.md) | Chronological incident log — crashes, fixes, lessons |
| [10_FAQ.md](10_FAQ.md) | Glossary, troubleshooting, frequently asked questions |

---

## Documentation Policy

- This documentation grows with the project. Every phase completion adds new sections.
- Previous content is never deleted — the engineering log is append-only.
- Every technical term is defined on first use.
- Every decision records the alternatives that were considered and rejected.
- Mathematical notation is included where it clarifies the explanation.
