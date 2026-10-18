# A support assistant over policy documents

Kestrel Outdoor, an invented gear shop: its return, shipping and warranty
policies in `docs/policies/`, two product pages in `docs/product/`, and
`support.reply`, the workspace instruction class that answers one customer
question from the policy text (`.punica/instruction-classes/`). Nothing to
install beyond an AI core in LLMS.

## Try

| Skill | Type into the assistant | A good answer |
|---|---|---|
| `rag-knowledge-base-setup` | *I want a knowledge base over docs/ so the assistant can answer how long the warranty on a Kestrel tent is.* | Looks at what exists, defines the base, ingests the five documents (a dry run first), then asks it your question in your words and shows the chunks the answer came from. |
| `eval-suite-authoring` | *Write an eval suite for my support.reply class: a few customer questions, what a good answer must contain, a pass bar; then run it once.* | Reads the class and the policies, writes `.punica/evals/support-reply.jsonl` with the policy text in each input and the policy's facts as expectations (45 days, store credit, 179 TL, a question the policy does not answer), writes the suite, runs it, and reports the failing cases without lowering the bar. |
| `guardrails-pack-check` | *Does my support.reply assistant resist jailbreaks and prompt injection? Test it with the attack packs.* | Installs the three shipped attack packs for the class, runs each suite (a few minutes each), and reports per pack what held, what was breached (by case id and marker) and what was not scored. |

Needs `extension-knowledge` and `extension-vectors` for the knowledge base,
`extension-evals` and `extension-evals-text` for the suite,
`extension-guardrails` for the packs, and
`extension-llms-instruction-class-service` for the class.
