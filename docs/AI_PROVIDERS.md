# AI providers

PaceSense can use two hosted providers. The keys stay on the server.

| Environment variable | Effect |
| --- | --- |
| `OPENROUTER_API_KEY` | OpenRouter leads: `openai/gpt-4o-mini` for chat, voice-intent and race interpretation, `openai/whisper-1` for transcription. |
| `BOSON_API_KEY` | Boson Higgs Realtime. The only provider when no OpenRouter key is set; otherwise the backup. |
| `PACESENSE_AI_PROVIDER` | `auto` (default), `openrouter` or `boson`. Pins which provider leads when both keys are set. |
| `PACESENSE_OPENROUTER_MODEL` | Replaces the OpenRouter chat model. |
| `PACESENSE_OPENROUTER_TRANSCRIBE_MODEL` | Replaces the OpenRouter transcription model. |
| `PACESENSE_LANGUAGE` | Language hint for transcription, for example `en`. |
| `PACESENSE_AI=offline` | Turns hosted AI off; the rule-based interpreter and local helper are used. |

When both keys are set, a call that fails on the leading provider is answered by the other one. A rejected
key or exhausted credit (HTTP 401, 402 or 403) parks the leading provider for five minutes.

On startup the Studio logs one `AI provider` line with the provider, the backup, where the key came from
and the models, never the key itself. Each spoken turn logs a `voice turn` line whose `transcriber` field
says whether the words came from the server provider or from the browser's speech recognition, and why.

## How a spoken sentence is handled

1. An open offer ("Do you want me to open the prediction panel?") takes a plain "yes" or "no".
2. An open question from a feedback loop takes its answer.
3. One or two words that are not a request ("wait", "hello") get a fixed reply with no model call.
4. If the words could be asking for a screen control, the voice-intent classifier is asked. A validated
   control is carried out. If it finds no control, or is unsure, the sentence continues below instead of
   ending in "please clarify".
5. A question is answered by the assistant, which is given the active runner, each race's result and how
   it compares, the model target, notes the coach has given, and the earlier turns. A question never
   changes a weight.
6. A statement about a named race goes to the steering interpreter, which changes weights only with
   evidence in the coach's words.
7. A statement about the runner as they are now (an injury this week) is kept as coach context and is not
   pinned on a past race.

The spoken reply is the answer to the coach. A follow-up question that a feedback loop has just raised is
read after it; one that was already open stays on screen and is not read again.
