# Annotation guidelines (agree on these BEFORE labelling; all annotators use the same file)

You label **raw extracted triples** `(head, relation, tail)` shown with the quote the model cited.
You do not see which repair steps kept or dropped a triple, nor the gold annotations. Keep it that way.

Read the full document in `docs.txt` (search for the `doc_id`). Judge against **the document text**,
not against your knowledge of the world.

| Label | Use it when |
|---|---|
| `correct` | the document states the fact, or directly implies it, **with the direction of the relation as described** (see the schema descriptions); the head and tail are the right entities |
| `incorrect` | the fact is not in the document (even if true in the world); the direction is reversed; an endpoint is the wrong entity or the wrong type of thing; the relation does not mean what the schema says |
| `uncertain` | you cannot decide after reading the whole document (ambiguous wording, unclear coreference). Use sparingly; the policy for `uncertain` is set in `config.yaml` (`annotation.uncertain`) |

Rules of thumb
1. **Quote problems are not label problems.** If the fact is right but the quote is paraphrased or wrong, label the *fact* `correct`. Quote validity is measured separately by step P.
2. **Duplicates:** if two entities clearly refer to the same thing, label each triple on its own merits.
3. **Type of the endpoint:** `P27` (country of citizenship) with tail "Berlin" is `incorrect` (a city, not a country).
4. **Dates:** a different granularity (year vs full date) is `correct` only if it does not contradict the text.
5. Write a short reason in `notes` for every `incorrect` and `uncertain`: this becomes the error taxonomy
   (suggested categories: hallucinated fact, wrong direction, wrong entity, wrong type, wrong relation, paraphrase-only).

Calibration: first, all three annotators independently label the separate 30-item sample in
`annotation/calibration/`. Compare those calibration labels, refine this file, and then freeze the
guidelines. The final 50-item sample in `annotation/` excludes those calibration raw IDs. All three
annotators then label that final sample independently; do not discuss those labels until all three
files are complete, or inter-annotator agreement is not meaningful. Score only the final sample.
