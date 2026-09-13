# Original hackathon submission (archived)

This folder preserves the code exactly as it was submitted for the TotalEnergies
AI Hackathon 2026 — the Gemini + Monte Carlo + hill-climbing version described
in the top-level README's [Revision notes](../README.md#-revision-notes-post-contest).

It is kept here for reference and comparison only. **It is not maintained and
is not the current solution** — that's `src/` at the repository root. This
copy won't run as-is: it depends on `llama-index`/`google-genai` and a
`GOOGLE_API_KEY`, neither of which are in the top-level `requirements.txt`
anymore, and it expects to sit directly under a repo root with `inputs/`,
`outputs/`, and `images/` as siblings of its own parent folder.
