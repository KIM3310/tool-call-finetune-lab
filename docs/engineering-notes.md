# Keep the evaluator independent of the answer

The previous loop removed all assistant messages but kept later tool results. On multi-turn records this lost valid history, exposed future observations, and combined tool-call targets from several turns into one prediction. The evaluator now predicts only the last tool-call turn using the preceding conversation.

The old scorer accepted extra arguments and compared values through lowercased strings. It could treat a numeric string as a number or erase case-sensitive identifiers. The current local contract compares complete JSON argument structures, with explicit type and ordering rules. It is deliberately identified separately from official BFCL evaluation.

The old deduplication key discarded examples sharing a prompt and function name even when labels or tool schemas differed. Full-content deduplication preserves those distinctions, while grouped partitioning keeps matching request/tool inputs together across sources. Input-order independence, group separation, malformed-data rejection, and ratio validation are covered by regression tests.

Evaluation results record the scorer contract and dataset fingerprint. Comparisons reject mismatched modern fingerprints and label legacy metadata as unverified. Report labels identify the actual configured model instead of assigning every reference run a fixed provider/model name.

`make proof` runs synthetic CPU cases and binds the report to the changed source. It does not train or query a language model. A measured model comparison still requires actual model assets and a held-out evaluation protocol.
